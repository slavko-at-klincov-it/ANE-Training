// ane_train.m — Implementation of ane_train.h high-level API
// Session-based wrappers around the proven training and generation pipelines.
// Compiled per model: make libane_train MODEL=stories110m
#include "ane_train.h"
#include "mil_dynamic.h"
#include "cpu_ops.h"
#include "hw_monitor.h"

// ===== Internal structs (not exposed in header) =====

// Dynamic kernel set per layer (10 kernels, compiled once)
typedef struct {
    Kern *sdpaFwd;
    Kern *woFwd;
    Kern *ffnFused;
    Kern *ffnBwdW2t;
    Kern *ffnBwdW13t;
    Kern *wotBwd;
    Kern *sdpaBwd1;
    Kern *sdpaBwd2;
    Kern *qBwd;
    Kern *kvBwd;
} DynLayerKernels;

struct ANETrainSession {
    ANETrainConfig config;
    int current_step, adam_t;
    float current_lr, best_loss, last_loss;
    int best_loss_step;

    LayerWeights lw[NLAYERS];
    LayerAdam la[NLAYERS];
    LayerActs acts[NLAYERS];
    LayerGrads grads[NLAYERS];
    float *rms_final, *embed;
    AdamState arms_final, aembed;

    float *Wqt[NLAYERS], *Wkt[NLAYERS], *Wvt[NLAYERS], *Wot[NLAYERS];
    float *W1t[NLAYERS], *W2t[NLAYERS], *W3t[NLAYERS];

    DynLayerKernels dk;
    PerLayerSurfaces pls[NLAYERS];
    PerLayerRequests plr[NLAYERS];

    // Work buffers
    float *dy, *dffn, *dx_ffn, *dx2, *dx_attn;
    float *dq, *dk_buf, *dv, *da_buf, *x_cur, *x_final, *xnorm_buf;
    float *logits, *dlogits, *dh1, *dh3, *dsilu, *silu_tmp, *silu_tmp2;
    float *k_tiled, *v_tiled, *dq_full, *dk_full, *dv_full;
    float *gate_buf;

    // Data
    uint16_t *token_data; size_t n_tokens; int data_fd; size_t data_len;
    VocabMap vm; int CV;
    float *cembed, *gcembed, *gembed, *grms_final;
    AdamState acembed;

    // Dispatch
    dispatch_queue_t dw_q; dispatch_group_t dw_grp;

    // Callbacks
    ANETrainProgressFn progress_fn; void *progress_ud;
    ANETrainCheckpointFn ckpt_fn; void *ckpt_ud;

    // Timing / stats
    uint64_t t_wall_start; double total_train_ms, compile_ms;
    int steps_done;
    float neg_max_act, max_act;

    // Cumulative from checkpoint resume
    double cum_train, cum_wall;
    int cum_steps;

    bool data_loaded;
    bool kernels_compiled;
};

// Forward-only kernel set for generation
typedef struct {
    Kern *sdpaFwd;
    Kern *woFwd;
    Kern *ffnFused;
} FwdKernels;

typedef struct {
    IOSurfaceRef sdpaFwd_in, woFwd_in, ffnFused_in;
} FwdLayerSurfaces;
typedef struct {
    void *sdpaFwd, *woFwd, *ffnFused;
} FwdLayerRequests;

// Tokenizer
typedef struct {
    char **vocab;
    float *scores;
    int vocab_size;
    int max_token_length;
} Tokenizer;

struct ANEGenSession {
    FwdKernels fk;
    FwdLayerSurfaces fls[NLAYERS];
    FwdLayerRequests flr[NLAYERS];
    LayerWeights lw[NLAYERS];
    float *rms_final, *embed;
    float *Wqt[NLAYERS], *Wkt[NLAYERS], *Wvt[NLAYERS], *Wot[NLAYERS];
    float *W1t[NLAYERS], *W3t[NLAYERS];
    Tokenizer tok;
    // Work buffers for forward
    float *x_cur, *xnorm_buf, *x_final;
    float *attn_out, *o_out, *x2, *x2norm;
    float *logits;
    int *context;
};

// ===== Helpers (same as train.m / generate.m) =====

static void transpose_weight(float *dst, const float *src, int rows, int cols) {
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            dst[c * rows + r] = src[r * cols + c];
}

// ===== Global init guard =====
static bool g_ane_inited = false;
static void ensure_ane_init(void) {
    if (!g_ane_inited) {
        ane_init();
        mach_timebase_info(&g_tb);
        g_ane_inited = true;
    }
}

// ===== Compile all 10 dynamic training kernels =====
static bool compile_dynamic_kernels(DynLayerKernels *dk) {
    NSDictionary *mask_w = @{@"@model_path/weights/mask.bin": @{@"offset":@0, @"data":get_mask_blob()}};
    NSDictionary *sdpa_fwd_w = @{
        @"@model_path/weights/mask.bin": @{@"offset":@0, @"data":get_mask_blob()},
        @"@model_path/weights/rope_cos.bin": @{@"offset":@0, @"data":get_rope_cos_blob()},
        @"@model_path/weights/rope_sin.bin": @{@"offset":@0, @"data":get_rope_sin_blob()}
    };

    int sdpa_out_ch = Q_DIM + Q_DIM + KV_DIM + KV_DIM + DIM;

    dk->sdpaFwd = compile_kern_mil_w(gen_sdpa_fwd_dynamic(), sdpa_fwd_w,
        DIM*SDPA_FWD_SP*2, sdpa_out_ch*SEQ*2);
    if (!dk->sdpaFwd) return false;

    dk->woFwd = compile_kern_mil_w(gen_wo_fwd_dynamic(), @{},
        Q_DIM*WO_FWD_SP*2, DIM*SEQ*2);
    if (!dk->woFwd) return false;

    int ffn_fused_och = DIM + 3*HIDDEN;
    dk->ffnFused = compile_kern_mil_w(gen_ffn_fused_dynamic(), @{},
        DIM*FFN_FUSED_SP*2, ffn_fused_och*SEQ*2);
    if (!dk->ffnFused) return false;

    dk->ffnBwdW2t = compile_kern_mil_w(gen_ffn_bwd_w2t_dynamic(), @{},
        DIM*FFN_BWD_W2T_SP*2, HIDDEN*SEQ*2);
    if (!dk->ffnBwdW2t) return false;

    dk->ffnBwdW13t = compile_kern_mil_w(gen_ffn_bwd_w13t_dynamic(), @{},
        HIDDEN*FFN_BWD_W13T_SP*2, DIM*SEQ*2);
    if (!dk->ffnBwdW13t) return false;

    dk->wotBwd = compile_kern_mil_w(gen_wot_dynamic(), @{},
        DIM*WOT_BWD_SP*2, Q_DIM*SEQ*2);
    if (!dk->wotBwd) return false;

    dk->sdpaBwd1 = compile_kern_mil_w(gen_sdpa_bwd1_noweight(), mask_w,
        4*Q_DIM*SEQ*2, (Q_DIM+2*SCORE_CH)*SEQ*2);
    if (!dk->sdpaBwd1) return false;

    dk->sdpaBwd2 = compile_kern_mil_w(gen_sdpa_bwd2(), @{},
        (2*SCORE_CH+2*Q_DIM)*SEQ*2, 2*Q_DIM*SEQ*2);
    if (!dk->sdpaBwd2) return false;

    dk->qBwd = compile_kern_mil_w(gen_q_bwd_dynamic(), @{},
        Q_DIM*Q_BWD_SP*2, DIM*SEQ*2);
    if (!dk->qBwd) return false;

    dk->kvBwd = compile_kern_mil_w(gen_kv_bwd_dynamic(), @{},
        KV_DIM*KV_BWD_SP*2, DIM*SEQ*2);
    if (!dk->kvBwd) return false;

    return true;
}

// ===== Checkpoint save/load (from train.m) =====

static void save_checkpoint_impl(const char *path, int step, int total_steps, float lr, float loss,
                                  double ct, double cw, int cs, int adam_t,
                                  LayerWeights *lw, LayerAdam *la, float *rms_final, AdamState *arms_final,
                                  float *embed, AdamState *aembed) {
    FILE *f = fopen(path, "wb");
    if (!f) return;
    CkptHdr h = {0};
    h.magic = 0x424C5A54; h.version = 4;
    h.step = step; h.total_steps = total_steps;
    h.n_layers = NLAYERS; h.vocab_size = VOCAB; h.dim = DIM;
    h.hidden_dim = HIDDEN; h.n_heads = HEADS; h.seq_len = SEQ;
    h.lr = lr; h.loss = loss;
    h.cum_train = ct; h.cum_wall = cw; h.cum_steps = cs; h.adam_t = adam_t;
    h.kv_heads = KV_HEADS; h.head_dim = HD; h.q_dim = Q_DIM;
    fwrite(&h, sizeof(h), 1, f);
    for (int L = 0; L < NLAYERS; L++) {
        fwrite(lw[L].Wq,4,WQ_SZ,f); fwrite(lw[L].Wk,4,WK_SZ,f);
        fwrite(lw[L].Wv,4,WV_SZ,f); fwrite(lw[L].Wo,4,WO_SZ,f);
        fwrite(lw[L].W1,4,W1_SZ,f); fwrite(lw[L].W2,4,W2_SZ,f); fwrite(lw[L].W3,4,W3_SZ,f);
        fwrite(lw[L].rms_att,4,DIM,f); fwrite(lw[L].rms_ffn,4,DIM,f);
        fwrite(la[L].Wq.m,4,WQ_SZ,f); fwrite(la[L].Wq.v,4,WQ_SZ,f);
        fwrite(la[L].Wk.m,4,WK_SZ,f); fwrite(la[L].Wk.v,4,WK_SZ,f);
        fwrite(la[L].Wv.m,4,WV_SZ,f); fwrite(la[L].Wv.v,4,WV_SZ,f);
        fwrite(la[L].Wo.m,4,WO_SZ,f); fwrite(la[L].Wo.v,4,WO_SZ,f);
        fwrite(la[L].W1.m,4,W1_SZ,f); fwrite(la[L].W1.v,4,W1_SZ,f);
        fwrite(la[L].W2.m,4,W2_SZ,f); fwrite(la[L].W2.v,4,W2_SZ,f);
        fwrite(la[L].W3.m,4,W3_SZ,f); fwrite(la[L].W3.v,4,W3_SZ,f);
        fwrite(la[L].rms_att.m,4,DIM,f); fwrite(la[L].rms_att.v,4,DIM,f);
        fwrite(la[L].rms_ffn.m,4,DIM,f); fwrite(la[L].rms_ffn.v,4,DIM,f);
    }
    fwrite(rms_final,4,DIM,f);
    fwrite(arms_final->m,4,DIM,f); fwrite(arms_final->v,4,DIM,f);
    fwrite(embed,4,(size_t)VOCAB*DIM,f);
    fwrite(aembed->m,4,(size_t)VOCAB*DIM,f); fwrite(aembed->v,4,(size_t)VOCAB*DIM,f);
    fclose(f);
}

static bool load_checkpoint_impl(const char *path, int *step, int *total_steps, float *lr, float *loss,
                                  double *ct, double *cw, int *cs, int *adam_t,
                                  LayerWeights *lw, LayerAdam *la, float *rms_final, AdamState *arms_final,
                                  float *embed, AdamState *aembed) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    CkptHdr h;
    fread(&h, sizeof(h), 1, f);
    if (h.magic != 0x424C5A54 || h.version != 4) { fclose(f); return false; }
    if (h.dim != DIM || h.hidden_dim != HIDDEN || h.n_heads != HEADS ||
        h.n_layers != NLAYERS || h.seq_len != SEQ || h.vocab_size != VOCAB) {
        fclose(f); return false;
    }
    *step = h.step; *total_steps = h.total_steps; *lr = h.lr; *loss = h.loss;
    *ct = h.cum_train; *cw = h.cum_wall; *cs = h.cum_steps; *adam_t = h.adam_t;
    for (int L = 0; L < NLAYERS; L++) {
        fread(lw[L].Wq,4,WQ_SZ,f); fread(lw[L].Wk,4,WK_SZ,f);
        fread(lw[L].Wv,4,WV_SZ,f); fread(lw[L].Wo,4,WO_SZ,f);
        fread(lw[L].W1,4,W1_SZ,f); fread(lw[L].W2,4,W2_SZ,f); fread(lw[L].W3,4,W3_SZ,f);
        fread(lw[L].rms_att,4,DIM,f); fread(lw[L].rms_ffn,4,DIM,f);
        fread(la[L].Wq.m,4,WQ_SZ,f); fread(la[L].Wq.v,4,WQ_SZ,f);
        fread(la[L].Wk.m,4,WK_SZ,f); fread(la[L].Wk.v,4,WK_SZ,f);
        fread(la[L].Wv.m,4,WV_SZ,f); fread(la[L].Wv.v,4,WV_SZ,f);
        fread(la[L].Wo.m,4,WO_SZ,f); fread(la[L].Wo.v,4,WO_SZ,f);
        fread(la[L].W1.m,4,W1_SZ,f); fread(la[L].W1.v,4,W1_SZ,f);
        fread(la[L].W2.m,4,W2_SZ,f); fread(la[L].W2.v,4,W2_SZ,f);
        fread(la[L].W3.m,4,W3_SZ,f); fread(la[L].W3.v,4,W3_SZ,f);
        fread(la[L].rms_att.m,4,DIM,f); fread(la[L].rms_att.v,4,DIM,f);
        fread(la[L].rms_ffn.m,4,DIM,f); fread(la[L].rms_ffn.v,4,DIM,f);
    }
    fread(rms_final,4,DIM,f);
    fread(arms_final->m,4,DIM,f); fread(arms_final->v,4,DIM,f);
    fread(embed,4,(size_t)VOCAB*DIM,f);
    fread(aembed->m,4,(size_t)VOCAB*DIM,f); fread(aembed->v,4,(size_t)VOCAB*DIM,f);
    fclose(f);
    return true;
}

// ===== Stage all weights into per-layer surfaces =====
static void stage_all_weights(ANETrainSession *s) {
    for (int L = 0; L < NLAYERS; L++) {
        stage_sdpa_fwd_weights(s->pls[L].sdpaFwd_in, s->Wqt[L], s->Wkt[L], s->Wvt[L]);
        stage_wo_fwd_weights(s->pls[L].woFwd_in, s->Wot[L]);
        stage_ffn_fused_weights(s->pls[L].ffnFused_in, s->W1t[L], s->W3t[L], s->lw[L].W2);
        stage_ffn_bwd_w2t_weights(s->pls[L].ffnBwdW2t_in, s->lw[L].W2);
        stage_ffn_bwd_w13t_weights(s->pls[L].ffnBwdW13t_in, s->lw[L].W1, s->lw[L].W3);
        stage_wot_bwd_weights(s->pls[L].wotBwd_in, s->lw[L].Wo);
        stage_q_bwd_weights(s->pls[L].qBwd_in, s->lw[L].Wq);
        stage_kv_bwd_weights(s->pls[L].kvBwd_in, s->lw[L].Wk, s->lw[L].Wv);
    }
}

// ===== Transpose all weights =====
static void transpose_all_weights(ANETrainSession *s) {
    for (int L = 0; L < NLAYERS; L++) {
        transpose_weight(s->Wqt[L], s->lw[L].Wq, Q_DIM, DIM);
        transpose_weight(s->Wkt[L], s->lw[L].Wk, KV_DIM, DIM);
        transpose_weight(s->Wvt[L], s->lw[L].Wv, KV_DIM, DIM);
        transpose_weight(s->Wot[L], s->lw[L].Wo, DIM, Q_DIM);
        transpose_weight(s->W1t[L], s->lw[L].W1, HIDDEN, DIM);
        transpose_weight(s->W2t[L], s->lw[L].W2, DIM, HIDDEN);
        transpose_weight(s->W3t[L], s->lw[L].W3, HIDDEN, DIM);
    }
}

// ===== Forward pass for one step =====
static void session_forward(ANETrainSession *s, uint16_t *input_tokens) {
    float res_alpha = 1.0f;
    embed_lookup(s->x_cur, s->embed, input_tokens, DIM, SEQ);

    for (int L = 0; L < NLAYERS; L++) {
        LayerActs *ac = &s->acts[L];
        memcpy(ac->layer_in, s->x_cur, SEQ*DIM*4);

        // RMSNorm (attention)
        rmsnorm(s->xnorm_buf, s->x_cur, s->lw[L].rms_att, DIM, SEQ);
        memcpy(ac->xnorm, s->xnorm_buf, SEQ*DIM*4);

        // Wait for pending dW cblas
        dispatch_group_wait(s->dw_grp, DISPATCH_TIME_FOREVER);

        // SDPA forward (ANE)
        write_sdpa_fwd_acts(s->pls[L].sdpaFwd_in, s->xnorm_buf);
        ane_eval_req(s->dk.sdpaFwd, s->plr[L].sdpaFwd);

        // Read SDPA output
        IOSurfaceLock(s->dk.sdpaFwd->ioOut, kIOSurfaceLockReadOnly, NULL);
        _Float16 *fwd_out = (_Float16*)IOSurfaceGetBaseAddress(s->dk.sdpaFwd->ioOut);
        int off = 0;
        cvt_f16_f32(ac->attn_out, fwd_out + off, Q_DIM*SEQ); off += Q_DIM*SEQ;
        cvt_f16_f32(ac->Q,        fwd_out + off, Q_DIM*SEQ); off += Q_DIM*SEQ;
        cvt_f16_f32(ac->K,        fwd_out + off, KV_DIM*SEQ); off += KV_DIM*SEQ;
        cvt_f16_f32(ac->V,        fwd_out + off, KV_DIM*SEQ);
        IOSurfaceUnlock(s->dk.sdpaFwd->ioOut, kIOSurfaceLockReadOnly, NULL);

        // Wo forward (ANE)
        write_wo_fwd_acts(s->pls[L].woFwd_in, ac->attn_out);
        ane_eval_req(s->dk.woFwd, s->plr[L].woFwd);
        io_read_dyn(s->dk.woFwd->ioOut, ac->o_out, DIM, SEQ);

        // Residual + RMSNorm (FFN)
        vDSP_vsma(ac->o_out, 1, &res_alpha, s->x_cur, 1, ac->x2, 1, (vDSP_Length)(SEQ*DIM));
        rmsnorm(ac->x2norm, ac->x2, s->lw[L].rms_ffn, DIM, SEQ);

        // Fused FFN (ANE)
        write_ffn_fused_acts(s->pls[L].ffnFused_in, ac->x2norm, ac->x2);
        ane_eval_req(s->dk.ffnFused, s->plr[L].ffnFused);

        // Read fused output
        IOSurfaceLock(s->dk.ffnFused->ioOut, kIOSurfaceLockReadOnly, NULL);
        _Float16 *ffn_out = (_Float16*)IOSurfaceGetBaseAddress(s->dk.ffnFused->ioOut);
        off = 0;
        cvt_f16_f32(s->x_cur,    ffn_out + off, DIM*SEQ);    off += DIM*SEQ;
        cvt_f16_f32(ac->h1,      ffn_out + off, HIDDEN*SEQ); off += HIDDEN*SEQ;
        cvt_f16_f32(ac->h3,      ffn_out + off, HIDDEN*SEQ); off += HIDDEN*SEQ;
        cvt_f16_f32(ac->silu_out, ffn_out + off, HIDDEN*SEQ);
        IOSurfaceUnlock(s->dk.ffnFused->ioOut, kIOSurfaceLockReadOnly, NULL);

        // Clamp residual
        if (s->max_act > 0)
            vDSP_vclip(s->x_cur, 1, &s->neg_max_act, &s->max_act, s->x_cur, 1, (vDSP_Length)(SEQ*DIM));
    }

    // Final RMSNorm
    rmsnorm(s->x_final, s->x_cur, s->rms_final, DIM, SEQ);
}

// ===== Backward pass for one step =====
static void session_backward(ANETrainSession *s, uint16_t *input_tokens, uint16_t *ctargets) {
    float res_alpha = 1.0f;
    float loss_scale = s->config.loss_scale;
    int CV = s->CV;

    // Classifier forward + loss
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                CV, SEQ, DIM, 1.0f, s->cembed, DIM, s->x_final, SEQ, 0.0f, s->logits, SEQ);
    float loss = cross_entropy_loss(s->dlogits, s->logits, ctargets, CV, SEQ);
    s->last_loss = loss;

    // Scale dlogits for FP16 stability
    vDSP_vsmul(s->dlogits, 1, &loss_scale, s->dlogits, 1, (vDSP_Length)(SEQ*CV));

    // Classifier backward
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                DIM, SEQ, CV, 1.0f, s->cembed, DIM, s->dlogits, SEQ, 0.0f, s->dy, SEQ);

    // dEmbed async
    dispatch_group_async(s->dw_grp, s->dw_q, ^{
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    CV, DIM, SEQ, 1.0f, s->dlogits, SEQ, s->x_final, SEQ, 1.0f, s->gcembed, DIM);
    });

    // Final RMSNorm backward
    float *dx_rms_final = (float*)calloc(SEQ*DIM, 4);
    rmsnorm_bwd(dx_rms_final, s->grms_final, s->dy, s->x_cur, s->rms_final, DIM, SEQ);
    memcpy(s->dy, dx_rms_final, SEQ*DIM*4);
    free(dx_rms_final);

    // Layer backward (reverse order)
    for (int L = NLAYERS-1; L >= 0; L--) {
        LayerActs *ac = &s->acts[L];
        LayerGrads *gr = &s->grads[L];

        // dffn = alpha * dy
        vDSP_vsmul(s->dy, 1, &res_alpha, s->dffn, 1, (vDSP_Length)(SEQ*DIM));

        // FFN backward W2^T (ANE)
        write_ffn_bwd_w2t_acts(s->pls[L].ffnBwdW2t_in, s->dffn);
        ane_eval_req(s->dk.ffnBwdW2t, s->plr[L].ffnBwdW2t);
        io_read_dyn(s->dk.ffnBwdW2t->ioOut, s->dsilu, HIDDEN, SEQ);

        // SiLU derivative
        {
            int n = HIDDEN*SEQ;
            float minus1 = -1.0f, one = 1.0f;
            vDSP_vsmul(ac->h1, 1, &minus1, s->silu_tmp, 1, (vDSP_Length)n);
            vvexpf(s->silu_tmp, s->silu_tmp, &n);
            vDSP_vsadd(s->silu_tmp, 1, &one, s->silu_tmp, 1, (vDSP_Length)n);
            vvrecf(s->silu_tmp, s->silu_tmp, &n);
            vDSP_vmul(ac->h1, 1, s->silu_tmp, 1, s->dh3, 1, (vDSP_Length)n);
            vDSP_vmul(s->dsilu, 1, s->dh3, 1, s->dh3, 1, (vDSP_Length)n);
            vDSP_vsadd(s->silu_tmp, 1, &minus1, s->silu_tmp2, 1, (vDSP_Length)n);
            vDSP_vneg(s->silu_tmp2, 1, s->silu_tmp2, 1, (vDSP_Length)n);
            vDSP_vmul(ac->h1, 1, s->silu_tmp2, 1, s->silu_tmp2, 1, (vDSP_Length)n);
            vDSP_vsadd(s->silu_tmp2, 1, &one, s->silu_tmp2, 1, (vDSP_Length)n);
            vDSP_vmul(s->silu_tmp, 1, s->silu_tmp2, 1, s->silu_tmp2, 1, (vDSP_Length)n);
            vDSP_vmul(s->dsilu, 1, ac->h3, 1, s->dh1, 1, (vDSP_Length)n);
            vDSP_vmul(s->dh1, 1, s->silu_tmp2, 1, s->dh1, 1, (vDSP_Length)n);
        }

        // dh1@W1^T + dh3@W3^T (ANE)
        write_ffn_bwd_w13t_acts(s->pls[L].ffnBwdW13t_in, s->dh1, s->dh3);
        ane_eval_req(s->dk.ffnBwdW13t, s->plr[L].ffnBwdW13t);
        io_read_dyn(s->dk.ffnBwdW13t->ioOut, s->dx_ffn, DIM, SEQ);

        // dW FFN async
        float *capt_dffn = (float*)malloc(SEQ*DIM*4); memcpy(capt_dffn, s->dffn, SEQ*DIM*4);
        float *capt_silu = (float*)malloc(SEQ*HIDDEN*4); memcpy(capt_silu, ac->silu_out, SEQ*HIDDEN*4);
        float *capt_dh1 = (float*)malloc(SEQ*HIDDEN*4); memcpy(capt_dh1, s->dh1, SEQ*HIDDEN*4);
        float *capt_dh3 = (float*)malloc(SEQ*HIDDEN*4); memcpy(capt_dh3, s->dh3, SEQ*HIDDEN*4);
        float *capt_x2n = (float*)malloc(SEQ*DIM*4); memcpy(capt_x2n, ac->x2norm, SEQ*DIM*4);
        dispatch_group_async(s->dw_grp, s->dw_q, ^{
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, DIM, HIDDEN, SEQ,
                        1.0f, capt_dffn, SEQ, capt_silu, SEQ, 1.0f, gr->W2, HIDDEN);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, HIDDEN, DIM, SEQ,
                        1.0f, capt_dh1, SEQ, capt_x2n, SEQ, 1.0f, gr->W1, DIM);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, HIDDEN, DIM, SEQ,
                        1.0f, capt_dh3, SEQ, capt_x2n, SEQ, 1.0f, gr->W3, DIM);
            free(capt_dffn); free(capt_silu); free(capt_dh1); free(capt_dh3); free(capt_x2n);
        });

        // RMSNorm2 backward
        memset(s->dx2, 0, SEQ*DIM*4);
        rmsnorm_bwd(s->dx2, gr->rms_ffn, s->dx_ffn, ac->x2, s->lw[L].rms_ffn, DIM, SEQ);
        for (int i = 0; i < SEQ*DIM; i++) s->dx2[i] += s->dy[i];

        // Wo^T backward (ANE)
        float *dx2_scaled = (float*)malloc(SEQ*DIM*4);
        vDSP_vsmul(s->dx2, 1, &res_alpha, dx2_scaled, 1, (vDSP_Length)(SEQ*DIM));
        write_wot_bwd_acts(s->pls[L].wotBwd_in, dx2_scaled);
        ane_eval_req(s->dk.wotBwd, s->plr[L].wotBwd);
        io_read_dyn(s->dk.wotBwd->ioOut, s->da_buf, Q_DIM, SEQ);

        // dWo async
        float *capt_do = (float*)malloc(SEQ*DIM*4); memcpy(capt_do, dx2_scaled, SEQ*DIM*4);
        free(dx2_scaled);
        float *capt_attn = (float*)malloc(SEQ*Q_DIM*4); memcpy(capt_attn, ac->attn_out, SEQ*Q_DIM*4);
        dispatch_group_async(s->dw_grp, s->dw_q, ^{
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, DIM, Q_DIM, SEQ,
                        1.0f, capt_do, SEQ, capt_attn, SEQ, 1.0f, gr->Wo, Q_DIM);
            free(capt_do); free(capt_attn);
        });

        // GQA tile
        gqa_tile_kv(s->k_tiled, ac->K, SEQ);
        gqa_tile_kv(s->v_tiled, ac->V, SEQ);

        // SDPA backward part 1
        io_write_fp16_at(s->dk.sdpaBwd1->ioIn, 0,       ac->Q,      Q_DIM, SEQ);
        io_write_fp16_at(s->dk.sdpaBwd1->ioIn, Q_DIM,   s->k_tiled, Q_DIM, SEQ);
        io_write_fp16_at(s->dk.sdpaBwd1->ioIn, 2*Q_DIM, s->v_tiled, Q_DIM, SEQ);
        io_write_fp16_at(s->dk.sdpaBwd1->ioIn, 3*Q_DIM, s->da_buf,  Q_DIM, SEQ);
        ane_eval(s->dk.sdpaBwd1);

        // SDPA backward part 2
        io_copy(s->dk.sdpaBwd2->ioIn, 0, s->dk.sdpaBwd1->ioOut, Q_DIM, 2*SCORE_CH, SEQ);
        io_write_fp16_at(s->dk.sdpaBwd2->ioIn, 2*SCORE_CH,       ac->Q,      Q_DIM, SEQ);
        io_write_fp16_at(s->dk.sdpaBwd2->ioIn, 2*SCORE_CH+Q_DIM, s->k_tiled, Q_DIM, SEQ);
        ane_eval(s->dk.sdpaBwd2);

        // Read SDPA backward outputs
        io_read_fp16(s->dk.sdpaBwd2->ioOut, s->dq_full, 0,     Q_DIM, SEQ);
        io_read_fp16(s->dk.sdpaBwd2->ioOut, s->dk_full, Q_DIM, Q_DIM, SEQ);
        io_read_fp16(s->dk.sdpaBwd1->ioOut, s->dv_full, 0,     Q_DIM, SEQ);

        // GQA reduce
        gqa_reduce_kv(s->dk_buf, s->dk_full, SEQ);
        gqa_reduce_kv(s->dv, s->dv_full, SEQ);
        memcpy(s->dq, s->dq_full, SEQ*Q_DIM*4);

        // RoPE backward
        rope_backward_inplace(s->dq, SEQ, Q_DIM, HD);
        rope_backward_inplace(s->dk_buf, SEQ, KV_DIM, HD);

        // dWq/dWk/dWv async
        float *capt_dq = (float*)malloc(SEQ*Q_DIM*4); memcpy(capt_dq, s->dq, SEQ*Q_DIM*4);
        float *capt_dk = (float*)malloc(SEQ*KV_DIM*4); memcpy(capt_dk, s->dk_buf, SEQ*KV_DIM*4);
        float *capt_dv = (float*)malloc(SEQ*KV_DIM*4); memcpy(capt_dv, s->dv, SEQ*KV_DIM*4);
        float *capt_xn = (float*)malloc(SEQ*DIM*4); memcpy(capt_xn, ac->xnorm, SEQ*DIM*4);
        dispatch_group_async(s->dw_grp, s->dw_q, ^{
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Q_DIM, DIM, SEQ,
                        1.0f, capt_dq, SEQ, capt_xn, SEQ, 1.0f, gr->Wq, DIM);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, KV_DIM, DIM, SEQ,
                        1.0f, capt_dk, SEQ, capt_xn, SEQ, 1.0f, gr->Wk, DIM);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, KV_DIM, DIM, SEQ,
                        1.0f, capt_dv, SEQ, capt_xn, SEQ, 1.0f, gr->Wv, DIM);
            free(capt_dq); free(capt_dk); free(capt_dv); free(capt_xn);
        });

        // Q backward (ANE)
        write_q_bwd_acts(s->pls[L].qBwd_in, s->dq);
        ane_eval_req(s->dk.qBwd, s->plr[L].qBwd);
        io_read_dyn(s->dk.qBwd->ioOut, s->dx_attn, DIM, SEQ);

        // KV backward (ANE)
        float *dx_kv = (float*)malloc(SEQ*DIM*4);
        write_kv_bwd_acts(s->pls[L].kvBwd_in, s->dk_buf, s->dv);
        ane_eval_req(s->dk.kvBwd, s->plr[L].kvBwd);
        io_read_dyn(s->dk.kvBwd->ioOut, dx_kv, DIM, SEQ);

        // dx_attn = dx_q + dx_kv
        for (int i = 0; i < SEQ*DIM; i++) s->dx_attn[i] += dx_kv[i];
        free(dx_kv);

        // RMSNorm1 backward
        float *dx_rms1 = (float*)calloc(SEQ*DIM, 4);
        rmsnorm_bwd(dx_rms1, gr->rms_att, s->dx_attn, ac->layer_in, s->lw[L].rms_att, DIM, SEQ);
        for (int i = 0; i < SEQ*DIM; i++) s->dy[i] = dx_rms1[i] + s->dx2[i];
        free(dx_rms1);
    }

    // Embedding backward
    dispatch_group_wait(s->dw_grp, DISPATCH_TIME_FOREVER);
    embed_backward(s->gembed, s->dy, input_tokens, DIM, SEQ);
}

// ===== Adam update step =====
static float session_adam_update(ANETrainSession *s) {
    float loss_scale = s->config.loss_scale;
    int accum = s->config.accum_steps;
    float grad_clip = s->config.grad_clip;
    float gsc = 1.0f / (accum * loss_scale);
    int CV = s->CV;

    s->adam_t++;

    // Scale gradients
    for (int L = 0; L < NLAYERS; L++) {
        LayerGrads *g = &s->grads[L];
        for (size_t i=0;i<WQ_SZ;i++) g->Wq[i]*=gsc;
        for (size_t i=0;i<WK_SZ;i++) g->Wk[i]*=gsc;
        for (size_t i=0;i<WV_SZ;i++) g->Wv[i]*=gsc;
        for (size_t i=0;i<WO_SZ;i++) g->Wo[i]*=gsc;
        for (size_t i=0;i<W1_SZ;i++) g->W1[i]*=gsc;
        for (size_t i=0;i<W2_SZ;i++) g->W2[i]*=gsc;
        for (size_t i=0;i<W3_SZ;i++) g->W3[i]*=gsc;
        for (int i=0;i<DIM;i++){g->rms_att[i]*=gsc; g->rms_ffn[i]*=gsc;}
    }
    for (int i=0;i<DIM;i++) s->grms_final[i]*=gsc;
    vocab_scatter_grads(s->gembed, s->gcembed, &s->vm, DIM);
    for (size_t i=0;i<(size_t)VOCAB*DIM;i++) s->gembed[i]*=gsc;

    // Global gradient norm
    float grad_norm_sq = 0;
    for (int L = 0; L < NLAYERS; L++) {
        LayerGrads *g = &s->grads[L]; float sn;
        vDSP_dotpr(g->Wq,1,g->Wq,1,&sn,(vDSP_Length)WQ_SZ); grad_norm_sq+=sn;
        vDSP_dotpr(g->Wk,1,g->Wk,1,&sn,(vDSP_Length)WK_SZ); grad_norm_sq+=sn;
        vDSP_dotpr(g->Wv,1,g->Wv,1,&sn,(vDSP_Length)WV_SZ); grad_norm_sq+=sn;
        vDSP_dotpr(g->Wo,1,g->Wo,1,&sn,(vDSP_Length)WO_SZ); grad_norm_sq+=sn;
        vDSP_dotpr(g->W1,1,g->W1,1,&sn,(vDSP_Length)W1_SZ); grad_norm_sq+=sn;
        vDSP_dotpr(g->W2,1,g->W2,1,&sn,(vDSP_Length)W2_SZ); grad_norm_sq+=sn;
        vDSP_dotpr(g->W3,1,g->W3,1,&sn,(vDSP_Length)W3_SZ); grad_norm_sq+=sn;
        vDSP_dotpr(g->rms_att,1,g->rms_att,1,&sn,(vDSP_Length)DIM); grad_norm_sq+=sn;
        vDSP_dotpr(g->rms_ffn,1,g->rms_ffn,1,&sn,(vDSP_Length)DIM); grad_norm_sq+=sn;
    }
    {
        float sn;
        vDSP_dotpr(s->grms_final,1,s->grms_final,1,&sn,(vDSP_Length)DIM); grad_norm_sq+=sn;
        vDSP_dotpr(s->gembed,1,s->gembed,1,&sn,(vDSP_Length)((size_t)VOCAB*DIM)); grad_norm_sq+=sn;
    }
    float grad_norm = sqrtf(grad_norm_sq);

    // Gradient clipping
    if (grad_clip > 0 && grad_norm > grad_clip) {
        float clip_scale = grad_clip / grad_norm;
        for (int L = 0; L < NLAYERS; L++) {
            LayerGrads *g = &s->grads[L];
            vDSP_vsmul(g->Wq,1,&clip_scale,g->Wq,1,(vDSP_Length)WQ_SZ);
            vDSP_vsmul(g->Wk,1,&clip_scale,g->Wk,1,(vDSP_Length)WK_SZ);
            vDSP_vsmul(g->Wv,1,&clip_scale,g->Wv,1,(vDSP_Length)WV_SZ);
            vDSP_vsmul(g->Wo,1,&clip_scale,g->Wo,1,(vDSP_Length)WO_SZ);
            vDSP_vsmul(g->W1,1,&clip_scale,g->W1,1,(vDSP_Length)W1_SZ);
            vDSP_vsmul(g->W2,1,&clip_scale,g->W2,1,(vDSP_Length)W2_SZ);
            vDSP_vsmul(g->W3,1,&clip_scale,g->W3,1,(vDSP_Length)W3_SZ);
            vDSP_vsmul(g->rms_att,1,&clip_scale,g->rms_att,1,(vDSP_Length)DIM);
            vDSP_vsmul(g->rms_ffn,1,&clip_scale,g->rms_ffn,1,(vDSP_Length)DIM);
        }
        vDSP_vsmul(s->grms_final,1,&clip_scale,s->grms_final,1,(vDSP_Length)DIM);
        vDSP_vsmul(s->gembed,1,&clip_scale,s->gembed,1,(vDSP_Length)((size_t)VOCAB*DIM));
    }

    // Cosine LR schedule with warmup
    float max_lr = s->config.learning_rate;
    float min_lr_frac = (s->config.min_lr > 0 && max_lr > 0) ? (s->config.min_lr / max_lr) : 0.1f;
    int warmup_steps = s->config.warmup_steps;
    int total_steps = s->config.max_steps;
    int step = s->current_step;

    if (step < warmup_steps) {
        s->current_lr = max_lr * ((float)(step + 1)) / warmup_steps;
    } else if (total_steps > warmup_steps) {
        float decay_ratio = (float)(step - warmup_steps) / (float)(total_steps - warmup_steps);
        float min_lr = max_lr * min_lr_frac;
        s->current_lr = min_lr + 0.5f * (1.0f + cosf(M_PI * decay_ratio)) * (max_lr - min_lr);
    } else {
        s->current_lr = max_lr;
    }

    float lr = s->current_lr;
    float b1 = s->config.beta1, b2 = s->config.beta2;
    float eps = s->config.eps, wd = s->config.weight_decay;

    // Adam update all parameters
    for (int L = 0; L < NLAYERS; L++) {
        LayerGrads *g = &s->grads[L];
        adam_update(s->lw[L].Wq, g->Wq, &s->la[L].Wq, s->adam_t, lr, b1, b2, eps, wd);
        adam_update(s->lw[L].Wk, g->Wk, &s->la[L].Wk, s->adam_t, lr, b1, b2, eps, wd);
        adam_update(s->lw[L].Wv, g->Wv, &s->la[L].Wv, s->adam_t, lr, b1, b2, eps, wd);
        adam_update(s->lw[L].Wo, g->Wo, &s->la[L].Wo, s->adam_t, lr, b1, b2, eps, wd);
        adam_update(s->lw[L].W1, g->W1, &s->la[L].W1, s->adam_t, lr, b1, b2, eps, wd);
        adam_update(s->lw[L].W2, g->W2, &s->la[L].W2, s->adam_t, lr, b1, b2, eps, wd);
        adam_update(s->lw[L].W3, g->W3, &s->la[L].W3, s->adam_t, lr, b1, b2, eps, wd);
        adam_update(s->lw[L].rms_att, g->rms_att, &s->la[L].rms_att, s->adam_t, lr, b1, b2, eps, 0.0f);
        adam_update(s->lw[L].rms_ffn, g->rms_ffn, &s->la[L].rms_ffn, s->adam_t, lr, b1, b2, eps, 0.0f);

        // Update transposed weight buffers and re-stage
        transpose_weight(s->Wqt[L], s->lw[L].Wq, Q_DIM, DIM);
        transpose_weight(s->Wkt[L], s->lw[L].Wk, KV_DIM, DIM);
        transpose_weight(s->Wvt[L], s->lw[L].Wv, KV_DIM, DIM);
        transpose_weight(s->Wot[L], s->lw[L].Wo, DIM, Q_DIM);
        transpose_weight(s->W1t[L], s->lw[L].W1, HIDDEN, DIM);
        transpose_weight(s->W2t[L], s->lw[L].W2, DIM, HIDDEN);
        transpose_weight(s->W3t[L], s->lw[L].W3, HIDDEN, DIM);

        stage_sdpa_fwd_weights(s->pls[L].sdpaFwd_in, s->Wqt[L], s->Wkt[L], s->Wvt[L]);
        stage_wo_fwd_weights(s->pls[L].woFwd_in, s->Wot[L]);
        stage_ffn_fused_weights(s->pls[L].ffnFused_in, s->W1t[L], s->W3t[L], s->lw[L].W2);
        stage_ffn_bwd_w2t_weights(s->pls[L].ffnBwdW2t_in, s->lw[L].W2);
        stage_ffn_bwd_w13t_weights(s->pls[L].ffnBwdW13t_in, s->lw[L].W1, s->lw[L].W3);
        stage_wot_bwd_weights(s->pls[L].wotBwd_in, s->lw[L].Wo);
        stage_q_bwd_weights(s->pls[L].qBwd_in, s->lw[L].Wq);
        stage_kv_bwd_weights(s->pls[L].kvBwd_in, s->lw[L].Wk, s->lw[L].Wv);
    }
    adam_update(s->rms_final, s->grms_final, &s->arms_final, s->adam_t, lr, b1, b2, eps, 0.0f);
    adam_update(s->embed, s->gembed, &s->aembed, s->adam_t, lr, b1, b2, eps, wd);
    free(s->cembed);
    s->cembed = vocab_compact_embed(s->embed, &s->vm, DIM);

    // Zero grads
    for (int L = 0; L < NLAYERS; L++) layer_grads_zero(&s->grads[L]);
    memset(s->grms_final, 0, DIM*4);
    memset(s->gembed, 0, (size_t)VOCAB*DIM*4);
    memset(s->gcembed, 0, (size_t)s->CV*DIM*4);

    return grad_norm;
}

// ===== Tokenizer (from generate.m) =====

static bool tokenizer_load(Tokenizer *t, const char *path, int vocab_size) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    t->vocab_size = vocab_size;
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->scores = (float*)malloc(vocab_size * sizeof(float));
    fread(&t->max_token_length, sizeof(int), 1, f);
    for (int i = 0; i < vocab_size; i++) {
        fread(&t->scores[i], sizeof(float), 1, f);
        int len;
        fread(&len, sizeof(int), 1, f);
        t->vocab[i] = (char*)malloc(len + 1);
        fread(t->vocab[i], 1, len, f);
        t->vocab[i][len] = '\0';
    }
    fclose(f);
    return true;
}

static void tokenizer_free(Tokenizer *t) {
    if (!t->vocab) return;
    for (int i = 0; i < t->vocab_size; i++) free(t->vocab[i]);
    free(t->vocab);
    free(t->scores);
    t->vocab = NULL;
    t->scores = NULL;
}

static const char *tokenizer_decode(Tokenizer *t, int token) {
    if (token < 0 || token >= t->vocab_size) return "";
    return t->vocab[token];
}

static int tokenizer_encode(Tokenizer *t, const char *text, int *tokens, int max_tokens) {
    if (!text || !*text) return 0;
    int *char_tokens = (int*)malloc(strlen(text) * sizeof(int));
    int n_chars = 0;
    const char *p = text;
    while (*p && n_chars < max_tokens) {
        char_tokens[n_chars] = -1;
        for (int v = 0; v < t->vocab_size; v++) {
            if (t->vocab[v][0] == *p && t->vocab[v][1] == '\0') {
                char_tokens[n_chars] = v;
                break;
            }
        }
        if (char_tokens[n_chars] == -1) { p++; continue; }
        n_chars++;
        p++;
    }
    while (n_chars > 1) {
        float best_score = -1e10f;
        int best_idx = -1, best_token = -1;
        for (int i = 0; i < n_chars - 1; i++) {
            char merged[256];
            snprintf(merged, sizeof(merged), "%s%s",
                     t->vocab[char_tokens[i]], t->vocab[char_tokens[i+1]]);
            for (int v = 0; v < t->vocab_size; v++) {
                if (strcmp(t->vocab[v], merged) == 0 && t->scores[v] > best_score) {
                    best_score = t->scores[v];
                    best_idx = i;
                    best_token = v;
                }
            }
        }
        if (best_idx == -1) break;
        char_tokens[best_idx] = best_token;
        for (int i = best_idx + 1; i < n_chars - 1; i++)
            char_tokens[i] = char_tokens[i + 1];
        n_chars--;
    }
    int n = n_chars < max_tokens ? n_chars : max_tokens;
    memcpy(tokens, char_tokens, n * sizeof(int));
    free(char_tokens);
    return n;
}

// ===== Sampling (from generate.m) =====

static int sample_argmax(const float *logits, int n) {
    int best = 0; float best_val = logits[0];
    for (int i = 1; i < n; i++)
        if (logits[i] > best_val) { best_val = logits[i]; best = i; }
    return best;
}

static int sample_temperature(const float *logits, int n, float temp) {
    float *probs = (float*)malloc(n * sizeof(float));
    float maxv = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > maxv) maxv = logits[i];
    float sum = 0;
    for (int i = 0; i < n; i++) {
        probs[i] = expf((logits[i] - maxv) / temp);
        sum += probs[i];
    }
    for (int i = 0; i < n; i++) probs[i] /= sum;
    float r = (float)drand48();
    float cumsum = 0;
    for (int i = 0; i < n; i++) {
        cumsum += probs[i];
        if (cumsum >= r) { free(probs); return i; }
    }
    free(probs);
    return n - 1;
}

static int sample_topp(const float *logits, int n, float temp, float topp) {
    float *probs = (float*)malloc(n * sizeof(float));
    float maxv = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > maxv) maxv = logits[i];
    float sum = 0;
    for (int i = 0; i < n; i++) {
        probs[i] = expf((logits[i] - maxv) / temp);
        sum += probs[i];
    }
    for (int i = 0; i < n; i++) probs[i] /= sum;
    int *idx = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) idx[i] = i;
    for (int i = 0; i < n - 1; i++) {
        int best = i;
        for (int j = i + 1; j < n; j++)
            if (probs[idx[j]] > probs[idx[best]]) best = j;
        if (best != i) { int tmp = idx[i]; idx[i] = idx[best]; idx[best] = tmp; }
        float cum = 0;
        for (int k = 0; k <= i; k++) cum += probs[idx[k]];
        if (cum >= topp) break;
    }
    float cum = 0, cutoff_sum = 0;
    int cutoff = 0;
    for (int i = 0; i < n; i++) {
        cum += probs[idx[i]];
        cutoff = i + 1;
        cutoff_sum = cum;
        if (cum >= topp) break;
    }
    float r = (float)drand48() * cutoff_sum;
    float cumsum = 0;
    int result = idx[0];
    for (int i = 0; i < cutoff; i++) {
        cumsum += probs[idx[i]];
        if (cumsum >= r) { result = idx[i]; break; }
    }
    free(probs);
    free(idx);
    return result;
}

// ===== Checkpoint loading for generation (weights only) =====
static bool load_gen_checkpoint(const char *path, CkptHdr *hdr,
                                 LayerWeights *lw, float *rms_final, float *embed) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    fread(hdr, sizeof(CkptHdr), 1, f);
    if (hdr->magic != 0x424C5A54 || hdr->version != 4) { fclose(f); return false; }
    if (hdr->dim != DIM || hdr->hidden_dim != HIDDEN || hdr->n_heads != HEADS ||
        hdr->n_layers != NLAYERS || hdr->seq_len != SEQ || hdr->vocab_size != VOCAB) {
        fclose(f); return false;
    }
    for (int L = 0; L < NLAYERS; L++) {
        fread(lw[L].Wq,4,WQ_SZ,f); fread(lw[L].Wk,4,WK_SZ,f);
        fread(lw[L].Wv,4,WV_SZ,f); fread(lw[L].Wo,4,WO_SZ,f);
        fread(lw[L].W1,4,W1_SZ,f); fread(lw[L].W2,4,W2_SZ,f); fread(lw[L].W3,4,W3_SZ,f);
        fread(lw[L].rms_att,4,DIM,f); fread(lw[L].rms_ffn,4,DIM,f);
        fseek(f, (WQ_SZ+WQ_SZ + WK_SZ+WK_SZ + WV_SZ+WV_SZ + WO_SZ+WO_SZ +
                   W1_SZ+W1_SZ + W2_SZ+W2_SZ + W3_SZ+W3_SZ + DIM+DIM + DIM+DIM)*4, SEEK_CUR);
    }
    fread(rms_final,4,DIM,f);
    fseek(f, (DIM+DIM)*4, SEEK_CUR);
    fread(embed,4,(size_t)VOCAB*DIM,f);
    fclose(f);
    return true;
}

// ===== Compile forward-only kernels (3 total) =====
static bool compile_fwd_kernels(FwdKernels *fk) {
    NSDictionary *sdpa_fwd_w = @{
        @"@model_path/weights/mask.bin": @{@"offset":@0, @"data":get_mask_blob()},
        @"@model_path/weights/rope_cos.bin": @{@"offset":@0, @"data":get_rope_cos_blob()},
        @"@model_path/weights/rope_sin.bin": @{@"offset":@0, @"data":get_rope_sin_blob()}
    };
    int sdpa_out_ch = Q_DIM + Q_DIM + KV_DIM + KV_DIM + DIM;
    fk->sdpaFwd = compile_kern_mil_w(gen_sdpa_fwd_dynamic(), sdpa_fwd_w,
        DIM*SDPA_FWD_SP*2, sdpa_out_ch*SEQ*2);
    if (!fk->sdpaFwd) return false;
    fk->woFwd = compile_kern_mil_w(gen_wo_fwd_dynamic(), @{},
        Q_DIM*WO_FWD_SP*2, DIM*SEQ*2);
    if (!fk->woFwd) return false;
    int ffn_fused_och = DIM + 3*HIDDEN;
    fk->ffnFused = compile_kern_mil_w(gen_ffn_fused_dynamic(), @{},
        DIM*FFN_FUSED_SP*2, ffn_fused_och*SEQ*2);
    if (!fk->ffnFused) return false;
    return true;
}

// =====================================================================
// PUBLIC API — Error / Info
// =====================================================================

const char *ane_train_error_str(ANETrainError err) {
    switch (err) {
        case ANE_TRAIN_OK:             return "OK";
        case ANE_TRAIN_ERR_ANE:        return "ANE device/driver error";
        case ANE_TRAIN_ERR_COMPILE:    return "Model compilation failed";
        case ANE_TRAIN_ERR_MEMORY:     return "Memory allocation failed";
        case ANE_TRAIN_ERR_CONFIG:     return "Invalid configuration";
        case ANE_TRAIN_ERR_CHECKPOINT: return "Checkpoint load/save failure";
        case ANE_TRAIN_ERR_DATA:       return "Training data load failure";
    }
    return "Unknown error";
}

ANEModelInfo ane_train_model_info(void) {
    size_t xformer = (size_t)NLAYERS * (WQ_SZ + WK_SZ + WV_SZ + WO_SZ + W1_SZ + W2_SZ + W3_SZ + 2*DIM);
    size_t embed_params = (size_t)VOCAB * DIM;
    ANEModelInfo info = {
        .name = MODEL_NAME,
        .dim = DIM,
        .n_layers = NLAYERS,
        .n_heads = HEADS,
        .n_kv_heads = KV_HEADS,
        .vocab_size = VOCAB,
        .seq_len = SEQ,
        .hidden_dim = HIDDEN,
        .param_count = xformer + embed_params + DIM, // +DIM for rms_final
    };
    return info;
}

ANETrainConfig ane_train_default_config(void) {
    ANETrainConfig cfg = {
        .learning_rate = 3e-4f,
        .min_lr = 3e-5f,
        .warmup_steps = 100,
        .max_steps = 10000,
        .batch_size = 1,
        .accum_steps = 10,
        .beta1 = 0.9f,
        .beta2 = 0.95f,
        .eps = 1e-8f,
        .weight_decay = 0.1f,
        .loss_scale = 256.0f,
        .grad_clip = 1.0f,
        .checkpoint_every = 100,
        .checkpoint_dir = NULL,
        .log_every = 10,
    };
    return cfg;
}

ANEHWSnapshot ane_hw_snapshot(void) {
    ensure_ane_init();
    // Ensure hw monitor globals are initialized for snapshot
    if (g_hwmon.tb.denom == 0) {
        mach_timebase_info(&g_hwmon.tb);
        g_hwmon.start_time = mach_absolute_time();
    }

    HWSnapshot raw = hw_snapshot();
    ANEHWSnapshot snap = {0};
    // CPU usage: approximate from cumulative times
    snap.cpu_usage = -1;  // Not available as instantaneous %
    snap.gpu_usage = (float)raw.gpu_util_pct;
    snap.mem_used_bytes = (size_t)(raw.mem_rss_mb * 1024 * 1024);
    // Total memory via sysctl
    int mib[2] = {CTL_HW, HW_MEMSIZE};
    uint64_t mem_total = 0;
    size_t len = sizeof(mem_total);
    sysctl(mib, 2, &mem_total, &len, NULL, 0);
    snap.mem_total_bytes = (size_t)mem_total;
    snap.thermal_state = raw.thermal_state;
    snap.timestamp = (double)(mach_absolute_time()) * g_tb.numer / g_tb.denom / 1e9;
    return snap;
}

// =====================================================================
// PUBLIC API — Training Session
// =====================================================================

ANETrainSession *ane_train_create(const ANETrainConfig *cfg, ANETrainError *err) {
    @autoreleasepool {
    ensure_ane_init();

    ANETrainSession *s = (ANETrainSession*)calloc(1, sizeof(ANETrainSession));
    if (!s) { if (err) *err = ANE_TRAIN_ERR_MEMORY; return NULL; }

    s->config = *cfg;
    s->current_step = 0;
    s->adam_t = 0;
    s->current_lr = cfg->learning_rate;
    s->best_loss = 999.0f;
    s->last_loss = 999.0f;
    s->best_loss_step = 0;
    s->max_act = 0;
    s->neg_max_act = 0;
    s->data_loaded = false;
    s->data_fd = -1;

    // Allocate per-layer state
    for (int L = 0; L < NLAYERS; L++) {
        s->lw[L] = layer_weights_alloc();
        s->la[L] = layer_adam_alloc();
        s->acts[L] = layer_acts_alloc();
        s->grads[L] = layer_grads_alloc();
    }
    s->rms_final = (float*)malloc(DIM*4);
    s->embed = (float*)malloc((size_t)VOCAB*DIM*4);
    s->grms_final = (float*)calloc(DIM, 4);
    s->gembed = (float*)calloc((size_t)VOCAB*DIM, 4);
    s->arms_final = adam_alloc(DIM);
    s->aembed = adam_alloc((size_t)VOCAB*DIM);

    // Random init (from scratch)
    srand48(42);
    float scale_d = 1.0f/sqrtf(DIM), scale_qd = 1.0f/sqrtf(Q_DIM), scale_h = 1.0f/sqrtf(HIDDEN);
    float out_scale = 1.0f/sqrtf((float)NLAYERS);
    for (int L = 0; L < NLAYERS; L++) {
        for (size_t i=0;i<WQ_SZ;i++) s->lw[L].Wq[i]=scale_d*(2*drand48()-1);
        for (size_t i=0;i<WK_SZ;i++) s->lw[L].Wk[i]=scale_d*(2*drand48()-1);
        for (size_t i=0;i<WV_SZ;i++) s->lw[L].Wv[i]=scale_d*(2*drand48()-1);
        for (size_t i=0;i<WO_SZ;i++) s->lw[L].Wo[i]=scale_qd*out_scale*(2*drand48()-1);
        for (size_t i=0;i<W1_SZ;i++) s->lw[L].W1[i]=scale_h*(2*drand48()-1);
        for (size_t i=0;i<W2_SZ;i++) s->lw[L].W2[i]=scale_d*out_scale*(2*drand48()-1);
        for (size_t i=0;i<W3_SZ;i++) s->lw[L].W3[i]=scale_h*(2*drand48()-1);
        for (int i=0;i<DIM;i++){s->lw[L].rms_att[i]=1.0f; s->lw[L].rms_ffn[i]=1.0f;}
    }
    for (int i=0;i<DIM;i++) s->rms_final[i]=1.0f;
    float escale = 0.02f;
    for (size_t i=0;i<(size_t)VOCAB*DIM;i++) s->embed[i]=escale*(2*drand48()-1);

    // Allocate transposed weight buffers
    for (int L = 0; L < NLAYERS; L++) {
        s->Wqt[L] = (float*)malloc(WQ_SZ*4);
        s->Wkt[L] = (float*)malloc(WK_SZ*4);
        s->Wvt[L] = (float*)malloc(WV_SZ*4);
        s->Wot[L] = (float*)malloc(WO_SZ*4);
        s->W1t[L] = (float*)malloc(W1_SZ*4);
        s->W2t[L] = (float*)malloc(W2_SZ*4);
        s->W3t[L] = (float*)malloc(W3_SZ*4);
    }
    transpose_all_weights(s);

    // Compile 10 dynamic kernels
    uint64_t tc = mach_absolute_time();
    if (!compile_dynamic_kernels(&s->dk)) {
        // Cleanup on failure
        for (int L = 0; L < NLAYERS; L++) {
            layer_weights_free(&s->lw[L]); layer_adam_free(&s->la[L]);
            layer_acts_free(&s->acts[L]); layer_grads_free(&s->grads[L]);
            free(s->Wqt[L]); free(s->Wkt[L]); free(s->Wvt[L]); free(s->Wot[L]);
            free(s->W1t[L]); free(s->W2t[L]); free(s->W3t[L]);
        }
        free(s->rms_final); free(s->embed);
        free(s->grms_final); free(s->gembed);
        adam_free(&s->arms_final); adam_free(&s->aembed);
        free(s);
        if (err) *err = ANE_TRAIN_ERR_COMPILE;
        return NULL;
    }
    s->compile_ms = tb_ms(mach_absolute_time() - tc);
    s->kernels_compiled = true;

    // Allocate per-layer IOSurfaces + requests
    for (int L = 0; L < NLAYERS; L++) {
        s->pls[L].sdpaFwd_in    = make_surface(DIM*SDPA_FWD_SP*2);
        s->pls[L].woFwd_in      = make_surface(Q_DIM*WO_FWD_SP*2);
        s->pls[L].ffnFused_in   = make_surface(DIM*FFN_FUSED_SP*2);
        s->pls[L].ffnBwdW2t_in  = make_surface(DIM*FFN_BWD_W2T_SP*2);
        s->pls[L].ffnBwdW13t_in = make_surface(HIDDEN*FFN_BWD_W13T_SP*2);
        s->pls[L].wotBwd_in     = make_surface(DIM*WOT_BWD_SP*2);
        s->pls[L].qBwd_in       = make_surface(Q_DIM*Q_BWD_SP*2);
        s->pls[L].kvBwd_in      = make_surface(KV_DIM*KV_BWD_SP*2);

        s->plr[L].sdpaFwd   = make_request(s->dk.sdpaFwd,    s->pls[L].sdpaFwd_in);
        s->plr[L].woFwd     = make_request(s->dk.woFwd,      s->pls[L].woFwd_in);
        s->plr[L].ffnFused  = make_request(s->dk.ffnFused,   s->pls[L].ffnFused_in);
        s->plr[L].ffnBwdW2t = make_request(s->dk.ffnBwdW2t,  s->pls[L].ffnBwdW2t_in);
        s->plr[L].ffnBwdW13t= make_request(s->dk.ffnBwdW13t, s->pls[L].ffnBwdW13t_in);
        s->plr[L].wotBwd    = make_request(s->dk.wotBwd,     s->pls[L].wotBwd_in);
        s->plr[L].qBwd      = make_request(s->dk.qBwd,       s->pls[L].qBwd_in);
        s->plr[L].kvBwd     = make_request(s->dk.kvBwd,      s->pls[L].kvBwd_in);
    }

    // Stage initial weights
    stage_all_weights(s);

    // Allocate work buffers
    s->dy       = (float*)malloc(SEQ*DIM*4);
    s->dffn     = (float*)malloc(SEQ*DIM*4);
    s->dx_ffn   = (float*)malloc(SEQ*DIM*4);
    s->dx2      = (float*)malloc(SEQ*DIM*4);
    s->dx_attn  = (float*)malloc(SEQ*DIM*4);
    s->dq       = (float*)malloc(SEQ*Q_DIM*4);
    s->dk_buf   = (float*)malloc(SEQ*KV_DIM*4);
    s->dv       = (float*)malloc(SEQ*KV_DIM*4);
    s->da_buf   = (float*)malloc(SEQ*Q_DIM*4);
    s->x_cur    = (float*)malloc(SEQ*DIM*4);
    s->x_final  = (float*)malloc(SEQ*DIM*4);
    s->xnorm_buf= (float*)malloc(SEQ*DIM*4);
    s->dh1      = (float*)malloc(SEQ*HIDDEN*4);
    s->dh3      = (float*)malloc(SEQ*HIDDEN*4);
    s->dsilu    = (float*)malloc(SEQ*HIDDEN*4);
    s->silu_tmp = (float*)malloc(SEQ*HIDDEN*4);
    s->silu_tmp2= (float*)malloc(SEQ*HIDDEN*4);
    s->gate_buf = (float*)malloc(SEQ*HIDDEN*4);
    s->k_tiled  = (float*)malloc(SEQ*Q_DIM*4);
    s->v_tiled  = (float*)malloc(SEQ*Q_DIM*4);
    s->dq_full  = (float*)malloc(SEQ*Q_DIM*4);
    s->dk_full  = (float*)malloc(SEQ*Q_DIM*4);
    s->dv_full  = (float*)malloc(SEQ*Q_DIM*4);

    // logits/dlogits are allocated after data is loaded (needs CV)

    s->dw_q = dispatch_queue_create("ane_train_cblas", DISPATCH_QUEUE_SERIAL);
    s->dw_grp = dispatch_group_create();

    s->progress_fn = NULL;
    s->ckpt_fn = NULL;
    s->t_wall_start = mach_absolute_time();

    if (err) *err = ANE_TRAIN_OK;
    return s;
    }
}

ANETrainError ane_train_load_data(ANETrainSession *s, const char *path) {
    if (!s) return ANE_TRAIN_ERR_CONFIG;

    s->data_fd = open(path, O_RDONLY);
    if (s->data_fd < 0) return ANE_TRAIN_ERR_DATA;

    struct stat st;
    fstat(s->data_fd, &st);
    s->data_len = st.st_size;
    s->token_data = (uint16_t*)mmap(NULL, s->data_len, PROT_READ, MAP_PRIVATE, s->data_fd, 0);
    if (s->token_data == MAP_FAILED) {
        close(s->data_fd);
        s->data_fd = -1;
        return ANE_TRAIN_ERR_DATA;
    }
    s->n_tokens = s->data_len / 2;

    // Build vocab map
    s->vm = vocab_map_build(s->token_data, s->n_tokens, VOCAB);
    s->CV = s->vm.compact_vocab;

    s->cembed = vocab_compact_embed(s->embed, &s->vm, DIM);
    s->gcembed = (float*)calloc((size_t)s->CV*DIM, 4);
    s->acembed = adam_alloc((size_t)s->CV*DIM);

    // Allocate logits buffers now that CV is known
    s->logits  = (float*)malloc(SEQ*s->CV*4);
    s->dlogits = (float*)malloc(SEQ*s->CV*4);

    s->data_loaded = true;
    return ANE_TRAIN_OK;
}

ANETrainError ane_train_resume(ANETrainSession *s, const char *checkpoint_path) {
    if (!s) return ANE_TRAIN_ERR_CONFIG;

    int step, total_steps, adam_t;
    float lr, loss;
    double ct, cw;
    int cs;

    if (!load_checkpoint_impl(checkpoint_path, &step, &total_steps, &lr, &loss,
                               &ct, &cw, &cs, &adam_t,
                               s->lw, s->la, s->rms_final, &s->arms_final,
                               s->embed, &s->aembed)) {
        return ANE_TRAIN_ERR_CHECKPOINT;
    }

    s->current_step = step;
    s->adam_t = adam_t;
    s->current_lr = lr;
    s->last_loss = loss;
    s->best_loss = loss;
    s->best_loss_step = step;
    s->cum_train = ct;
    s->cum_wall = cw;
    s->cum_steps = cs;

    // Update transposed weights and re-stage
    transpose_all_weights(s);
    stage_all_weights(s);

    // Rebuild compact embed if data loaded
    if (s->data_loaded) {
        free(s->cembed);
        s->cembed = vocab_compact_embed(s->embed, &s->vm, DIM);
    }

    return ANE_TRAIN_OK;
}

void ane_train_set_progress_callback(ANETrainSession *s, ANETrainProgressFn fn, void *ctx) {
    if (!s) return;
    s->progress_fn = fn;
    s->progress_ud = ctx;
}

void ane_train_set_checkpoint_callback(ANETrainSession *s, ANETrainCheckpointFn fn, void *ctx) {
    if (!s) return;
    s->ckpt_fn = fn;
    s->ckpt_ud = ctx;
}

ANETrainStepResult ane_train_step(ANETrainSession *s) {
    ANETrainStepResult res = {0};
    if (!s || !s->data_loaded) return res;

    uint64_t t_step = mach_absolute_time();

    // Sample data
    size_t max_pos = s->n_tokens - SEQ - 1;
    size_t pos = (size_t)(drand48() * max_pos);
    uint16_t *input_tokens = s->token_data + pos;
    uint16_t *target_tokens_raw = s->token_data + pos + 1;

    uint16_t ctargets[SEQ];
    for (int t = 0; t < SEQ; t++)
        ctargets[t] = (uint16_t)s->vm.full_to_compact[target_tokens_raw[t]];

    // Forward
    uint64_t t_fwd = mach_absolute_time();
    session_forward(s, input_tokens);
    double fwd_ms = tb_ms(mach_absolute_time() - t_fwd);

    // Backward
    uint64_t t_bwd = mach_absolute_time();
    session_backward(s, input_tokens, ctargets);
    double bwd_ms = tb_ms(mach_absolute_time() - t_bwd);

    s->steps_done++;
    double update_ms = 0;
    float grad_norm = 0;

    // Adam update every accum_steps
    bool do_update = ((s->current_step + 1) % s->config.accum_steps == 0) ||
                     (s->config.max_steps > 0 && s->current_step == s->config.max_steps - 1);
    if (do_update) {
        uint64_t t_upd = mach_absolute_time();
        grad_norm = session_adam_update(s);
        update_ms = tb_ms(mach_absolute_time() - t_upd);

        // Auto-checkpoint
        if (s->config.checkpoint_every > 0 &&
            (s->current_step + 1) % s->config.checkpoint_every == 0 &&
            s->last_loss < s->best_loss) {
            s->best_loss = s->last_loss;
            s->best_loss_step = s->current_step + 1;

            char ckpt_path[512];
            if (s->config.checkpoint_dir)
                snprintf(ckpt_path, sizeof(ckpt_path), "%s/%s", s->config.checkpoint_dir, CKPT_PATH);
            else
                snprintf(ckpt_path, sizeof(ckpt_path), "%s", CKPT_PATH);

            bool do_save = true;
            if (s->ckpt_fn)
                do_save = s->ckpt_fn(s->current_step + 1, ckpt_path, s->ckpt_ud);

            if (do_save) {
                double wall = tb_ms(mach_absolute_time() - s->t_wall_start);
                save_checkpoint_impl(ckpt_path, s->current_step + 1, s->config.max_steps,
                    s->current_lr, s->last_loss,
                    s->total_train_ms + s->cum_train, wall + s->cum_wall,
                    s->steps_done + s->cum_steps, s->adam_t,
                    s->lw, s->la, s->rms_final, &s->arms_final, s->embed, &s->aembed);
            }
        }
    }

    double step_ms = tb_ms(mach_absolute_time() - t_step);
    s->total_train_ms += step_ms;

    res.step = s->current_step + 1;
    res.loss = s->last_loss;
    res.lr = s->current_lr;
    res.grad_norm = grad_norm;
    res.fwd_ms = fwd_ms;
    res.bwd_ms = bwd_ms;
    res.update_ms = update_ms;
    res.total_ms = step_ms;

    s->current_step++;
    return res;
}

ANETrainError ane_train_run(ANETrainSession *s) {
    if (!s) return ANE_TRAIN_ERR_CONFIG;
    if (!s->data_loaded) return ANE_TRAIN_ERR_DATA;

    srand48(42 + s->current_step);

    int max_steps = s->config.max_steps;
    for (int step = s->current_step; step < max_steps; step++) {
        ANETrainStepResult res = ane_train_step(s);

        if (s->progress_fn) {
            if (!s->progress_fn(&res, s->progress_ud))
                break;
        }
    }
    return ANE_TRAIN_OK;
}

ANETrainStats ane_train_stats(const ANETrainSession *s) {
    ANETrainStats st = {0};
    if (!s) return st;

    st.steps_done = s->steps_done;
    st.best_loss = s->best_loss;
    st.best_loss_step = s->best_loss_step;
    st.compile_ms = s->compile_ms;

    double wall = tb_ms(mach_absolute_time() - s->t_wall_start);
    st.elapsed_s = (wall + s->cum_wall) / 1000.0;
    st.avg_step_ms = s->steps_done > 0 ? s->total_train_ms / s->steps_done : 0;

    // Estimate TFLOPS
    double fwd_flops = 2.0 * NLAYERS * ((double)WQ_SZ + WK_SZ + WV_SZ + WO_SZ + W1_SZ + W2_SZ + W3_SZ) * SEQ;
    double total_flops = 3.0 * fwd_flops;  // forward + backward ~= 3x forward
    if (st.avg_step_ms > 0)
        st.tflops = (float)(total_flops / (st.avg_step_ms * 1e9));
    else
        st.tflops = 0;

    return st;
}

ANETrainError ane_train_save(const ANETrainSession *s, const char *path) {
    if (!s) return ANE_TRAIN_ERR_CONFIG;

    double wall = tb_ms(mach_absolute_time() - s->t_wall_start);
    // Cast away const for save — save_checkpoint_impl does not modify data
    ANETrainSession *ms = (ANETrainSession*)s;
    save_checkpoint_impl(path, s->current_step, s->config.max_steps,
        s->current_lr, s->last_loss,
        s->total_train_ms + s->cum_train, wall + s->cum_wall,
        s->steps_done + s->cum_steps, s->adam_t,
        ms->lw, ms->la, ms->rms_final, &ms->arms_final, ms->embed, &ms->aembed);
    return ANE_TRAIN_OK;
}

void ane_train_destroy(ANETrainSession *s) {
    if (!s) return;

    // Wait for pending async work
    dispatch_group_wait(s->dw_grp, DISPATCH_TIME_FOREVER);

    for (int L = 0; L < NLAYERS; L++) {
        layer_weights_free(&s->lw[L]); layer_adam_free(&s->la[L]);
        layer_acts_free(&s->acts[L]); layer_grads_free(&s->grads[L]);
        free(s->Wqt[L]); free(s->Wkt[L]); free(s->Wvt[L]); free(s->Wot[L]);
        free(s->W1t[L]); free(s->W2t[L]); free(s->W3t[L]);
    }

    if (s->kernels_compiled) {
        free_per_layer(s->pls, s->plr);
        free_kern(s->dk.sdpaFwd); free_kern(s->dk.woFwd); free_kern(s->dk.ffnFused);
        free_kern(s->dk.ffnBwdW2t); free_kern(s->dk.ffnBwdW13t); free_kern(s->dk.wotBwd);
        free_kern(s->dk.sdpaBwd1); free_kern(s->dk.sdpaBwd2);
        free_kern(s->dk.qBwd); free_kern(s->dk.kvBwd);
    }

    free(s->rms_final); free(s->embed);
    free(s->grms_final); free(s->gembed);
    adam_free(&s->arms_final); adam_free(&s->aembed);

    free(s->dy); free(s->dffn); free(s->dx_ffn);
    free(s->dx2); free(s->dx_attn);
    free(s->dq); free(s->dk_buf); free(s->dv); free(s->da_buf);
    free(s->x_cur); free(s->x_final); free(s->xnorm_buf);
    free(s->logits); free(s->dlogits);
    free(s->dh1); free(s->dh3); free(s->dsilu);
    free(s->silu_tmp); free(s->silu_tmp2); free(s->gate_buf);
    free(s->k_tiled); free(s->v_tiled);
    free(s->dq_full); free(s->dk_full); free(s->dv_full);

    if (s->data_loaded) {
        munmap(s->token_data, s->data_len);
        close(s->data_fd);
        free(s->vm.full_to_compact);
        free(s->vm.compact_to_full);
        free(s->cembed); free(s->gcembed);
        adam_free(&s->acembed);
    }

    free(s);
}

// =====================================================================
// PUBLIC API — Generation Session
// =====================================================================

ANEGenConfig ane_gen_default_config(void) {
    ANEGenConfig cfg = {
        .temperature = 0.8f,
        .top_p = 0.9f,
        .max_tokens = 256,
        .seed = 0,
    };
    return cfg;
}

ANEGenSession *ane_gen_create(const char *checkpoint_path,
                               const char *tokenizer_path,
                               ANETrainError *err) {
    @autoreleasepool {
    ensure_ane_init();

    ANEGenSession *g = (ANEGenSession*)calloc(1, sizeof(ANEGenSession));
    if (!g) { if (err) *err = ANE_TRAIN_ERR_MEMORY; return NULL; }

    // Load tokenizer
    if (!tokenizer_load(&g->tok, tokenizer_path, VOCAB)) {
        free(g);
        if (err) *err = ANE_TRAIN_ERR_DATA;
        return NULL;
    }

    // Allocate weights
    for (int L = 0; L < NLAYERS; L++) g->lw[L] = layer_weights_alloc();
    g->rms_final = (float*)malloc(DIM*4);
    g->embed = (float*)malloc((size_t)VOCAB*DIM*4);

    // Load checkpoint
    CkptHdr hdr;
    if (!load_gen_checkpoint(checkpoint_path, &hdr, g->lw, g->rms_final, g->embed)) {
        for (int L = 0; L < NLAYERS; L++) layer_weights_free(&g->lw[L]);
        free(g->rms_final); free(g->embed);
        tokenizer_free(&g->tok);
        free(g);
        if (err) *err = ANE_TRAIN_ERR_CHECKPOINT;
        return NULL;
    }

    // Transpose weights for forward kernels
    for (int L = 0; L < NLAYERS; L++) {
        g->Wqt[L] = (float*)malloc(WQ_SZ*4);
        g->Wkt[L] = (float*)malloc(WK_SZ*4);
        g->Wvt[L] = (float*)malloc(WV_SZ*4);
        g->Wot[L] = (float*)malloc(WO_SZ*4);
        g->W1t[L] = (float*)malloc(W1_SZ*4);
        g->W3t[L] = (float*)malloc(W3_SZ*4);
        transpose_weight(g->Wqt[L], g->lw[L].Wq, Q_DIM, DIM);
        transpose_weight(g->Wkt[L], g->lw[L].Wk, KV_DIM, DIM);
        transpose_weight(g->Wvt[L], g->lw[L].Wv, KV_DIM, DIM);
        transpose_weight(g->Wot[L], g->lw[L].Wo, DIM, Q_DIM);
        transpose_weight(g->W1t[L], g->lw[L].W1, HIDDEN, DIM);
        transpose_weight(g->W3t[L], g->lw[L].W3, HIDDEN, DIM);
    }

    // Compile 3 forward kernels
    if (!compile_fwd_kernels(&g->fk)) {
        for (int L = 0; L < NLAYERS; L++) {
            layer_weights_free(&g->lw[L]);
            free(g->Wqt[L]); free(g->Wkt[L]); free(g->Wvt[L]);
            free(g->Wot[L]); free(g->W1t[L]); free(g->W3t[L]);
        }
        free(g->rms_final); free(g->embed);
        tokenizer_free(&g->tok);
        free(g);
        if (err) *err = ANE_TRAIN_ERR_COMPILE;
        return NULL;
    }

    // Allocate per-layer IOSurfaces + requests
    for (int L = 0; L < NLAYERS; L++) {
        g->fls[L].sdpaFwd_in  = make_surface(DIM*SDPA_FWD_SP*2);
        g->fls[L].woFwd_in    = make_surface(Q_DIM*WO_FWD_SP*2);
        g->fls[L].ffnFused_in = make_surface(DIM*FFN_FUSED_SP*2);
        g->flr[L].sdpaFwd  = make_request(g->fk.sdpaFwd,  g->fls[L].sdpaFwd_in);
        g->flr[L].woFwd    = make_request(g->fk.woFwd,    g->fls[L].woFwd_in);
        g->flr[L].ffnFused = make_request(g->fk.ffnFused,  g->fls[L].ffnFused_in);
    }

    // Stage weights
    for (int L = 0; L < NLAYERS; L++) {
        stage_sdpa_fwd_weights(g->fls[L].sdpaFwd_in, g->Wqt[L], g->Wkt[L], g->Wvt[L]);
        stage_wo_fwd_weights(g->fls[L].woFwd_in, g->Wot[L]);
        stage_ffn_fused_weights(g->fls[L].ffnFused_in, g->W1t[L], g->W3t[L], g->lw[L].W2);
    }

    // Allocate work buffers
    g->x_cur    = (float*)malloc(SEQ*DIM*4);
    g->xnorm_buf= (float*)malloc(SEQ*DIM*4);
    g->x_final  = (float*)malloc(SEQ*DIM*4);
    g->attn_out = (float*)malloc(SEQ*Q_DIM*4);
    g->o_out    = (float*)malloc(SEQ*DIM*4);
    g->x2       = (float*)malloc(SEQ*DIM*4);
    g->x2norm   = (float*)malloc(SEQ*DIM*4);
    g->logits   = (float*)malloc(VOCAB*sizeof(float));
    g->context  = (int*)calloc(SEQ, sizeof(int));

    if (err) *err = ANE_TRAIN_OK;
    return g;
    }
}

ANEGenResult ane_gen_run(ANEGenSession *g, const char *prompt,
                          const ANEGenConfig *cfg,
                          ANEGenTokenFn on_token, void *ctx) {
    ANEGenResult result = {0};
    if (!g) { result.stop_reason = ANE_GEN_STOP_ERROR; return result; }

    float temperature = cfg ? cfg->temperature : 0.8f;
    float topp = cfg ? cfg->top_p : 0.9f;
    int max_tokens = cfg ? cfg->max_tokens : 256;
    uint64_t seed = cfg ? cfg->seed : 0;

    if (seed) srand48(seed); else srand48(time(NULL));

    // Build prompt tokens
    int ctx_len = 0;
    g->context[ctx_len++] = 1; // BOS

    if (prompt) {
        int prompt_tokens[SEQ];
        int n_prompt = tokenizer_encode(&g->tok, prompt, prompt_tokens, SEQ - 2);
        for (int i = 0; i < n_prompt && ctx_len < SEQ; i++)
            g->context[ctx_len++] = prompt_tokens[i];
    }

    // Accumulate generated text
    size_t text_cap = 4096;
    size_t text_len = 0;
    char *text = (char*)malloc(text_cap);
    text[0] = '\0';

    // Copy prompt text to output
    for (int i = 1; i < ctx_len; i++) {
        const char *piece = tokenizer_decode(&g->tok, g->context[i]);
        size_t plen = strlen(piece);
        while (text_len + plen + 1 > text_cap) { text_cap *= 2; text = (char*)realloc(text, text_cap); }
        memcpy(text + text_len, piece, plen);
        text_len += plen;
        text[text_len] = '\0';
    }

    float res_alpha = 1.0f;
    uint64_t t_gen_start = mach_absolute_time();
    int tokens_generated = 0;

    for (int gen = 0; gen < max_tokens; gen++) {
        // Build input
        memset(g->x_cur, 0, SEQ*DIM*4);
        uint16_t input_u16[SEQ];
        for (int t = 0; t < ctx_len; t++) input_u16[t] = (uint16_t)g->context[t];
        for (int t = ctx_len; t < SEQ; t++) input_u16[t] = 0;
        embed_lookup(g->x_cur, g->embed, input_u16, DIM, SEQ);

        // Forward pass
        for (int L = 0; L < NLAYERS; L++) {
            rmsnorm(g->xnorm_buf, g->x_cur, g->lw[L].rms_att, DIM, SEQ);
            write_sdpa_fwd_acts(g->fls[L].sdpaFwd_in, g->xnorm_buf);
            ane_eval_req(g->fk.sdpaFwd, g->flr[L].sdpaFwd);

            IOSurfaceLock(g->fk.sdpaFwd->ioOut, kIOSurfaceLockReadOnly, NULL);
            _Float16 *fwd_out = (_Float16*)IOSurfaceGetBaseAddress(g->fk.sdpaFwd->ioOut);
            cvt_f16_f32(g->attn_out, fwd_out, Q_DIM*SEQ);
            IOSurfaceUnlock(g->fk.sdpaFwd->ioOut, kIOSurfaceLockReadOnly, NULL);

            write_wo_fwd_acts(g->fls[L].woFwd_in, g->attn_out);
            ane_eval_req(g->fk.woFwd, g->flr[L].woFwd);
            io_read_dyn(g->fk.woFwd->ioOut, g->o_out, DIM, SEQ);

            vDSP_vsma(g->o_out, 1, &res_alpha, g->x_cur, 1, g->x2, 1, (vDSP_Length)(SEQ*DIM));
            rmsnorm(g->x2norm, g->x2, g->lw[L].rms_ffn, DIM, SEQ);

            write_ffn_fused_acts(g->fls[L].ffnFused_in, g->x2norm, g->x2);
            ane_eval_req(g->fk.ffnFused, g->flr[L].ffnFused);

            IOSurfaceLock(g->fk.ffnFused->ioOut, kIOSurfaceLockReadOnly, NULL);
            _Float16 *ffn_out = (_Float16*)IOSurfaceGetBaseAddress(g->fk.ffnFused->ioOut);
            cvt_f16_f32(g->x_cur, ffn_out, DIM*SEQ);
            IOSurfaceUnlock(g->fk.ffnFused->ioOut, kIOSurfaceLockReadOnly, NULL);
        }

        // Final RMSNorm
        rmsnorm(g->x_final, g->x_cur, g->rms_final, DIM, SEQ);

        // Classifier at last position
        int last_pos = ctx_len - 1;
        float *x_last = (float*)malloc(DIM * sizeof(float));
        for (int d = 0; d < DIM; d++)
            x_last[d] = g->x_final[d * SEQ + last_pos];
        cblas_sgemv(CblasRowMajor, CblasNoTrans, VOCAB, DIM,
                    1.0f, g->embed, DIM, x_last, 1, 0.0f, g->logits, 1);
        free(x_last);

        // Sample
        int next_token;
        if (temperature < 1e-6f) {
            next_token = sample_argmax(g->logits, VOCAB);
        } else if (topp > 0.0f && topp < 1.0f) {
            next_token = sample_topp(g->logits, VOCAB, temperature, topp);
        } else {
            next_token = sample_temperature(g->logits, VOCAB, temperature);
        }

        // EOS
        if (next_token == 2) {
            result.stop_reason = ANE_GEN_STOP_EOS;
            break;
        }

        // Append to context
        if (ctx_len >= SEQ) {
            memmove(g->context + 1, g->context + 2, (SEQ - 2) * sizeof(int));
            ctx_len = SEQ - 1;
        }
        g->context[ctx_len++] = next_token;
        tokens_generated++;

        // Decode and accumulate
        const char *piece = tokenizer_decode(&g->tok, next_token);
        size_t plen = strlen(piece);
        while (text_len + plen + 1 > text_cap) { text_cap *= 2; text = (char*)realloc(text, text_cap); }
        memcpy(text + text_len, piece, plen);
        text_len += plen;
        text[text_len] = '\0';

        // Token callback
        if (on_token) {
            if (!on_token(piece, next_token, ctx)) {
                result.stop_reason = ANE_GEN_STOP_CALLBACK;
                tokens_generated++;  // count the token that caused stop
                break;
            }
        }

        if (gen == max_tokens - 1) {
            result.stop_reason = ANE_GEN_STOP_MAX;
        }
    }

    double gen_ms = tb_ms(mach_absolute_time() - t_gen_start);
    result.text = text;
    result.tokens_generated = tokens_generated;
    result.total_ms = gen_ms;
    result.ms_per_token = tokens_generated > 0 ? gen_ms / tokens_generated : 0;
    return result;
}

void ane_gen_result_free(ANEGenResult *r) {
    if (r && r->text) {
        free(r->text);
        r->text = NULL;
    }
}

void ane_gen_destroy(ANEGenSession *g) {
    if (!g) return;

    free(g->x_cur); free(g->xnorm_buf); free(g->x_final);
    free(g->attn_out); free(g->o_out); free(g->x2); free(g->x2norm);
    free(g->logits); free(g->context);

    for (int L = 0; L < NLAYERS; L++) {
        CFRelease(g->fls[L].sdpaFwd_in);
        CFRelease(g->fls[L].woFwd_in);
        CFRelease(g->fls[L].ffnFused_in);
        CFRelease(g->flr[L].sdpaFwd);
        CFRelease(g->flr[L].woFwd);
        CFRelease(g->flr[L].ffnFused);
        layer_weights_free(&g->lw[L]);
        free(g->Wqt[L]); free(g->Wkt[L]); free(g->Wvt[L]);
        free(g->Wot[L]); free(g->W1t[L]); free(g->W3t[L]);
    }
    free(g->rms_final); free(g->embed);
    free_kern(g->fk.sdpaFwd); free_kern(g->fk.woFwd); free_kern(g->fk.ffnFused);
    tokenizer_free(&g->tok);

    free(g);
}
