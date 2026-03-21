// generate.m — Autoregressive text generation using ANE forward kernels
// Loads a trained checkpoint, compiles 3 forward kernels, generates text.
// Build: make generate_tiny_ane (or generate_stories110m, etc.)
#include "mil_dynamic.h"
#include "cpu_ops.h"

// Forward-only kernel set (3 of 10 from training)
typedef struct {
    Kern *sdpaFwd;
    Kern *woFwd;
    Kern *ffnFused;
} FwdKernels;

// Per-layer IOSurfaces + requests for forward only
typedef struct {
    IOSurfaceRef sdpaFwd_in, woFwd_in, ffnFused_in;
} FwdLayerSurfaces;
typedef struct {
    void *sdpaFwd, *woFwd, *ffnFused;
} FwdLayerRequests;

// Transpose W[rows,cols] -> W^T[cols,rows]
static void transpose_weight(float *dst, const float *src, int rows, int cols) {
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            dst[c * rows + r] = src[r * cols + c];
}

// ===== Tokenizer =====
typedef struct {
    char **vocab;
    float *scores;
    int vocab_size;
    int max_token_length;
} Tokenizer;

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
    for (int i = 0; i < t->vocab_size; i++) free(t->vocab[i]);
    free(t->vocab);
    free(t->scores);
}

static const char *tokenizer_decode(Tokenizer *t, int token) {
    if (token < 0 || token >= t->vocab_size) return "";
    return t->vocab[token];
}

// Simple BPE encode: greedy longest-match
static int tokenizer_encode(Tokenizer *t, const char *text, int *tokens, int max_tokens) {
    if (!text || !*text) return 0;
    int n = 0;
    // First: encode each character as its own token
    int *char_tokens = (int*)malloc(strlen(text) * sizeof(int));
    int n_chars = 0;
    const char *p = text;
    while (*p && n_chars < max_tokens) {
        // Find longest single-char match
        char_tokens[n_chars] = -1;
        for (int v = 0; v < t->vocab_size; v++) {
            if (t->vocab[v][0] == *p && t->vocab[v][1] == '\0') {
                char_tokens[n_chars] = v;
                break;
            }
        }
        if (char_tokens[n_chars] == -1) {
            // Unknown char, skip
            p++;
            continue;
        }
        n_chars++;
        p++;
    }

    // BPE merge: repeatedly find the best pair to merge
    while (n_chars > 1) {
        float best_score = -1e10f;
        int best_idx = -1;
        int best_token = -1;

        for (int i = 0; i < n_chars - 1; i++) {
            // Build merged string
            char merged[256];
            snprintf(merged, sizeof(merged), "%s%s",
                     t->vocab[char_tokens[i]], t->vocab[char_tokens[i+1]]);
            // Find in vocab
            for (int v = 0; v < t->vocab_size; v++) {
                if (strcmp(t->vocab[v], merged) == 0 && t->scores[v] > best_score) {
                    best_score = t->scores[v];
                    best_idx = i;
                    best_token = v;
                }
            }
        }
        if (best_idx == -1) break; // No more merges possible

        // Apply merge
        char_tokens[best_idx] = best_token;
        for (int i = best_idx + 1; i < n_chars - 1; i++)
            char_tokens[i] = char_tokens[i + 1];
        n_chars--;
    }

    n = n_chars < max_tokens ? n_chars : max_tokens;
    memcpy(tokens, char_tokens, n * sizeof(int));
    free(char_tokens);
    return n;
}

// ===== Sampling =====
static int sample_argmax(const float *logits, int n) {
    int best = 0;
    float best_val = logits[0];
    for (int i = 1; i < n; i++) {
        if (logits[i] > best_val) { best_val = logits[i]; best = i; }
    }
    return best;
}

static int sample_temperature(const float *logits, int n, float temp) {
    // Apply temperature and softmax
    float *probs = (float*)malloc(n * sizeof(float));
    float maxv = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > maxv) maxv = logits[i];
    float sum = 0;
    for (int i = 0; i < n; i++) {
        probs[i] = expf((logits[i] - maxv) / temp);
        sum += probs[i];
    }
    for (int i = 0; i < n; i++) probs[i] /= sum;

    // Sample from distribution
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
    // Apply temperature
    float *probs = (float*)malloc(n * sizeof(float));
    float maxv = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > maxv) maxv = logits[i];
    float sum = 0;
    for (int i = 0; i < n; i++) {
        probs[i] = expf((logits[i] - maxv) / temp);
        sum += probs[i];
    }
    for (int i = 0; i < n; i++) probs[i] /= sum;

    // Sort indices by probability (descending)
    int *idx = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) idx[i] = i;
    // Simple selection sort for top-p (only need top portion)
    for (int i = 0; i < n - 1; i++) {
        int best = i;
        for (int j = i + 1; j < n; j++)
            if (probs[idx[j]] > probs[idx[best]]) best = j;
        if (best != i) { int tmp = idx[i]; idx[i] = idx[best]; idx[best] = tmp; }
        // Early exit: if cumulative prob exceeds topp, stop sorting
        float cum = 0;
        for (int k = 0; k <= i; k++) cum += probs[idx[k]];
        if (cum >= topp) break;
    }

    // Re-normalize within top-p nucleus
    float cum = 0, cutoff_sum = 0;
    int cutoff = 0;
    for (int i = 0; i < n; i++) {
        cum += probs[idx[i]];
        cutoff = i + 1;
        cutoff_sum = cum;
        if (cum >= topp) break;
    }

    // Sample from nucleus
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

// ===== Checkpoint loading (weights only, skip adam state) =====
static bool load_gen_checkpoint(const char *path, CkptHdr *hdr,
                                 LayerWeights *lw, float *rms_final, float *embed) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    fread(hdr, sizeof(CkptHdr), 1, f);
    if (hdr->magic != 0x424C5A54 || hdr->version != 4) { fclose(f); return false; }

    // Verify model dimensions match
    if (hdr->dim != DIM || hdr->hidden_dim != HIDDEN || hdr->n_heads != HEADS ||
        hdr->n_layers != NLAYERS || hdr->seq_len != SEQ || hdr->vocab_size != VOCAB) {
        printf("ERROR: Checkpoint dims don't match compiled model!\n");
        printf("  Checkpoint: dim=%d hidden=%d heads=%d layers=%d seq=%d vocab=%d\n",
               hdr->dim, hdr->hidden_dim, hdr->n_heads, hdr->n_layers, hdr->seq_len, hdr->vocab_size);
        printf("  Compiled:   dim=%d hidden=%d heads=%d layers=%d seq=%d vocab=%d\n",
               DIM, HIDDEN, HEADS, NLAYERS, SEQ, VOCAB);
        fclose(f);
        return false;
    }

    for (int L = 0; L < NLAYERS; L++) {
        // Read weights
        fread(lw[L].Wq,4,WQ_SZ,f); fread(lw[L].Wk,4,WK_SZ,f);
        fread(lw[L].Wv,4,WV_SZ,f); fread(lw[L].Wo,4,WO_SZ,f);
        fread(lw[L].W1,4,W1_SZ,f); fread(lw[L].W2,4,W2_SZ,f); fread(lw[L].W3,4,W3_SZ,f);
        fread(lw[L].rms_att,4,DIM,f); fread(lw[L].rms_ffn,4,DIM,f);
        // Skip adam state (m and v for each weight)
        fseek(f, (WQ_SZ+WQ_SZ + WK_SZ+WK_SZ + WV_SZ+WV_SZ + WO_SZ+WO_SZ +
                   W1_SZ+W1_SZ + W2_SZ+W2_SZ + W3_SZ+W3_SZ + DIM+DIM + DIM+DIM)*4, SEEK_CUR);
    }
    fread(rms_final,4,DIM,f);
    // Skip adam state for rms_final
    fseek(f, (DIM+DIM)*4, SEEK_CUR);
    fread(embed,4,(size_t)VOCAB*DIM,f);
    // Don't need adam state for embed
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

    printf("  Compiling sdpaFwd...\n");
    fk->sdpaFwd = compile_kern_mil_w(gen_sdpa_fwd_dynamic(), sdpa_fwd_w,
        DIM*SDPA_FWD_SP*2, sdpa_out_ch*SEQ*2);
    if (!fk->sdpaFwd) return false;

    printf("  Compiling woFwd...\n");
    fk->woFwd = compile_kern_mil_w(gen_wo_fwd_dynamic(), @{},
        Q_DIM*WO_FWD_SP*2, DIM*SEQ*2);
    if (!fk->woFwd) return false;

    int ffn_fused_och = DIM + 3*HIDDEN;
    printf("  Compiling ffnFused...\n");
    fk->ffnFused = compile_kern_mil_w(gen_ffn_fused_dynamic(), @{},
        DIM*FFN_FUSED_SP*2, ffn_fused_och*SEQ*2);
    if (!fk->ffnFused) return false;

    return true;
}

// ===== Main =====
int main(int argc, char *argv[]) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        ane_init();
        mach_timebase_info(&g_tb);

        // CLI args
        const char *ckpt_path = CKPT_PATH;
        const char *tok_path = "../../assets/models/tokenizer.bin";
        const char *prompt = NULL;
        float temperature = 0.8f;
        float topp = 0.9f;
        int max_tokens = 200;
        unsigned long seed = 0;
        bool seed_set = false;

        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--ckpt") == 0 && i+1<argc) ckpt_path = argv[++i];
            else if (strcmp(argv[i], "--tokenizer") == 0 && i+1<argc) tok_path = argv[++i];
            else if (strcmp(argv[i], "--prompt") == 0 && i+1<argc) prompt = argv[++i];
            else if (strcmp(argv[i], "--temp") == 0 && i+1<argc) temperature = atof(argv[++i]);
            else if (strcmp(argv[i], "--topp") == 0 && i+1<argc) topp = atof(argv[++i]);
            else if (strcmp(argv[i], "--max_tokens") == 0 && i+1<argc) max_tokens = atoi(argv[++i]);
            else if (strcmp(argv[i], "--seed") == 0 && i+1<argc) { seed = atol(argv[++i]); seed_set = true; }
            else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
                printf("Usage: generate [options]\n");
                printf("  --ckpt PATH       Checkpoint file (default: %s)\n", CKPT_PATH);
                printf("  --tokenizer PATH  Tokenizer file (default: ../../assets/models/tokenizer.bin)\n");
                printf("  --prompt TEXT     Starting prompt (default: BOS token)\n");
                printf("  --temp FLOAT      Temperature (0=argmax, default: 0.8)\n");
                printf("  --topp FLOAT      Top-p nucleus sampling (default: 0.9)\n");
                printf("  --max_tokens INT  Max tokens to generate (default: 200)\n");
                printf("  --seed INT        Random seed\n");
                return 0;
            }
        }

        if (seed_set) srand48(seed); else srand48(time(NULL));

        // Load tokenizer
        printf("Loading tokenizer from %s...\n", tok_path);
        Tokenizer tok;
        if (!tokenizer_load(&tok, tok_path, VOCAB)) {
            printf("ERROR: Cannot load tokenizer from %s\n", tok_path);
            return 1;
        }
        printf("  Loaded %d tokens (max_len=%d)\n", tok.vocab_size, tok.max_token_length);

        // Allocate weights
        LayerWeights lw[NLAYERS];
        for (int L = 0; L < NLAYERS; L++) lw[L] = layer_weights_alloc();
        float *rms_final = (float*)malloc(DIM*4);
        float *embed = (float*)malloc((size_t)VOCAB*DIM*4);

        // Load checkpoint
        printf("Loading checkpoint from %s...\n", ckpt_path);
        CkptHdr hdr;
        if (!load_gen_checkpoint(ckpt_path, &hdr, lw, rms_final, embed)) {
            printf("ERROR: Cannot load checkpoint from %s\n", ckpt_path);
            return 1;
        }
        printf("  Model: %s, step=%d, loss=%.4f\n", MODEL_NAME, hdr.step, hdr.loss);
        printf("  dim=%d q_dim=%d kv_dim=%d heads=%d/%d layers=%d seq=%d\n",
               DIM, Q_DIM, KV_DIM, HEADS, KV_HEADS, NLAYERS, SEQ);

        // Transpose weights for forward kernels
        printf("Transposing weights...\n");
        float *Wqt_buf[NLAYERS], *Wkt_buf[NLAYERS], *Wvt_buf[NLAYERS], *Wot_buf[NLAYERS];
        float *W1t_buf[NLAYERS], *W3t_buf[NLAYERS];
        for (int L = 0; L < NLAYERS; L++) {
            Wqt_buf[L] = (float*)malloc(WQ_SZ*4);
            Wkt_buf[L] = (float*)malloc(WK_SZ*4);
            Wvt_buf[L] = (float*)malloc(WV_SZ*4);
            Wot_buf[L] = (float*)malloc(WO_SZ*4);
            W1t_buf[L] = (float*)malloc(W1_SZ*4);
            W3t_buf[L] = (float*)malloc(W3_SZ*4);
            transpose_weight(Wqt_buf[L], lw[L].Wq, Q_DIM, DIM);
            transpose_weight(Wkt_buf[L], lw[L].Wk, KV_DIM, DIM);
            transpose_weight(Wvt_buf[L], lw[L].Wv, KV_DIM, DIM);
            transpose_weight(Wot_buf[L], lw[L].Wo, DIM, Q_DIM);
            transpose_weight(W1t_buf[L], lw[L].W1, HIDDEN, DIM);
            transpose_weight(W3t_buf[L], lw[L].W3, HIDDEN, DIM);
        }

        // Compile forward kernels
        printf("Compiling 3 forward kernels...\n");
        uint64_t tc = mach_absolute_time();
        FwdKernels fk;
        if (!compile_fwd_kernels(&fk)) {
            printf("ERROR: Kernel compilation failed!\n");
            return 1;
        }
        printf("  Compiled in %.0fms\n", tb_ms(mach_absolute_time() - tc));

        // Allocate per-layer IOSurfaces + requests
        FwdLayerSurfaces fls[NLAYERS];
        FwdLayerRequests flr[NLAYERS];
        for (int L = 0; L < NLAYERS; L++) {
            fls[L].sdpaFwd_in  = make_surface(DIM*SDPA_FWD_SP*2);
            fls[L].woFwd_in    = make_surface(Q_DIM*WO_FWD_SP*2);
            fls[L].ffnFused_in = make_surface(DIM*FFN_FUSED_SP*2);
            flr[L].sdpaFwd  = make_request(fk.sdpaFwd,  fls[L].sdpaFwd_in);
            flr[L].woFwd    = make_request(fk.woFwd,    fls[L].woFwd_in);
            flr[L].ffnFused = make_request(fk.ffnFused,  fls[L].ffnFused_in);
        }

        // Stage weights
        printf("Staging weights...\n");
        for (int L = 0; L < NLAYERS; L++) {
            stage_sdpa_fwd_weights(fls[L].sdpaFwd_in, Wqt_buf[L], Wkt_buf[L], Wvt_buf[L]);
            stage_wo_fwd_weights(fls[L].woFwd_in, Wot_buf[L]);
            stage_ffn_fused_weights(fls[L].ffnFused_in, W1t_buf[L], W3t_buf[L], lw[L].W2);
        }

        // Build prompt tokens
        int *context = (int*)calloc(SEQ, sizeof(int));
        int ctx_len = 0;
        context[ctx_len++] = 1; // BOS

        if (prompt) {
            int prompt_tokens[SEQ];
            int n_prompt = tokenizer_encode(&tok, prompt, prompt_tokens, SEQ - 2);
            for (int i = 0; i < n_prompt && ctx_len < SEQ; i++)
                context[ctx_len++] = prompt_tokens[i];
        }

        // Print prompt tokens
        printf("\n--- Generation (temp=%.2f, topp=%.2f, max=%d) ---\n", temperature, topp, max_tokens);
        for (int i = 1; i < ctx_len; i++) // skip BOS
            printf("%s", tokenizer_decode(&tok, context[i]));
        fflush(stdout);

        // Work buffers for forward pass
        float *x_cur = (float*)malloc(SEQ*DIM*4);
        float *xnorm_buf = (float*)malloc(SEQ*DIM*4);
        float *x_final = (float*)malloc(SEQ*DIM*4);
        float *attn_out = (float*)malloc(SEQ*Q_DIM*4);
        float *o_out = (float*)malloc(SEQ*DIM*4);
        float *x2 = (float*)malloc(SEQ*DIM*4);
        float *x2norm = (float*)malloc(SEQ*DIM*4);
        float *logits = (float*)malloc(VOCAB*sizeof(float)); // single position logits
        float *logits_col = (float*)malloc(VOCAB*sizeof(float));
        float res_alpha = 1.0f;

        uint64_t t_gen_start = mach_absolute_time();
        int tokens_generated = 0;

        for (int gen = 0; gen < max_tokens; gen++) {
            // Build input: embed all context tokens into x_cur [DIM, SEQ]
            // Zero-pad positions beyond ctx_len
            memset(x_cur, 0, SEQ*DIM*4);
            uint16_t input_u16[SEQ];
            for (int t = 0; t < ctx_len; t++) input_u16[t] = (uint16_t)context[t];
            for (int t = ctx_len; t < SEQ; t++) input_u16[t] = 0;
            embed_lookup(x_cur, embed, input_u16, DIM, SEQ);

            // Forward pass through all layers
            for (int L = 0; L < NLAYERS; L++) {
                // RMSNorm (attention)
                rmsnorm(xnorm_buf, x_cur, lw[L].rms_att, DIM, SEQ);

                // SDPA forward (ANE)
                write_sdpa_fwd_acts(fls[L].sdpaFwd_in, xnorm_buf);
                ane_eval_req(fk.sdpaFwd, flr[L].sdpaFwd);

                // Read attn_out from SDPA output (first Q_DIM channels)
                IOSurfaceLock(fk.sdpaFwd->ioOut, kIOSurfaceLockReadOnly, NULL);
                _Float16 *fwd_out = (_Float16*)IOSurfaceGetBaseAddress(fk.sdpaFwd->ioOut);
                cvt_f16_f32(attn_out, fwd_out, Q_DIM*SEQ);
                IOSurfaceUnlock(fk.sdpaFwd->ioOut, kIOSurfaceLockReadOnly, NULL);

                // Wo forward (ANE)
                write_wo_fwd_acts(fls[L].woFwd_in, attn_out);
                ane_eval_req(fk.woFwd, flr[L].woFwd);
                io_read_dyn(fk.woFwd->ioOut, o_out, DIM, SEQ);

                // Residual connection
                vDSP_vsma(o_out, 1, &res_alpha, x_cur, 1, x2, 1, (vDSP_Length)(SEQ*DIM));

                // RMSNorm (FFN)
                rmsnorm(x2norm, x2, lw[L].rms_ffn, DIM, SEQ);

                // Fused FFN (ANE)
                write_ffn_fused_acts(fls[L].ffnFused_in, x2norm, x2);
                ane_eval_req(fk.ffnFused, flr[L].ffnFused);

                // Read x_next from fused output (first DIM channels only)
                IOSurfaceLock(fk.ffnFused->ioOut, kIOSurfaceLockReadOnly, NULL);
                _Float16 *ffn_out = (_Float16*)IOSurfaceGetBaseAddress(fk.ffnFused->ioOut);
                cvt_f16_f32(x_cur, ffn_out, DIM*SEQ);
                IOSurfaceUnlock(fk.ffnFused->ioOut, kIOSurfaceLockReadOnly, NULL);
            }

            // Final RMSNorm
            rmsnorm(x_final, x_cur, rms_final, DIM, SEQ);

            // Classifier: logits = embed^T @ x_final at last position
            // Full vocab: embed is [VOCAB, DIM], x_final is [DIM, SEQ]
            // We only need logits at position (ctx_len - 1)
            int last_pos = ctx_len - 1;
            // Extract the column at last_pos from x_final[DIM, SEQ]
            float *x_last = (float*)malloc(DIM * sizeof(float));
            for (int d = 0; d < DIM; d++)
                x_last[d] = x_final[d * SEQ + last_pos];

            // logits[v] = embed[v, :] . x_last[:] for all v in [0, VOCAB)
            cblas_sgemv(CblasRowMajor, CblasNoTrans, VOCAB, DIM,
                        1.0f, embed, DIM, x_last, 1, 0.0f, logits, 1);
            free(x_last);

            // Sample next token
            int next_token;
            if (temperature == 0.0f || temperature < 1e-6f) {
                next_token = sample_argmax(logits, VOCAB);
            } else if (topp > 0.0f && topp < 1.0f) {
                next_token = sample_topp(logits, VOCAB, temperature, topp);
            } else {
                next_token = sample_temperature(logits, VOCAB, temperature);
            }

            // Check EOS
            if (next_token == 2) break; // EOS

            // Append to context
            if (ctx_len >= SEQ) {
                // Context full — shift left by 1, keeping BOS
                memmove(context + 1, context + 2, (SEQ - 2) * sizeof(int));
                ctx_len = SEQ - 1;
            }
            context[ctx_len++] = next_token;
            tokens_generated++;

            // Print decoded token
            const char *piece = tokenizer_decode(&tok, next_token);
            printf("%s", piece);
            fflush(stdout);
        }

        double gen_ms = tb_ms(mach_absolute_time() - t_gen_start);
        printf("\n\n--- Stats ---\n");
        printf("Tokens generated: %d\n", tokens_generated);
        printf("Total time: %.1fms (%.1f ms/token, %.1f tokens/sec)\n",
               gen_ms, gen_ms / tokens_generated,
               tokens_generated / (gen_ms / 1000.0));

        // Cleanup
        free(x_cur); free(xnorm_buf); free(x_final);
        free(attn_out); free(o_out); free(x2); free(x2norm);
        free(logits); free(logits_col); free(context);
        for (int L = 0; L < NLAYERS; L++) {
            CFRelease(fls[L].sdpaFwd_in);
            CFRelease(fls[L].woFwd_in);
            CFRelease(fls[L].ffnFused_in);
            CFRelease(flr[L].sdpaFwd);
            CFRelease(flr[L].woFwd);
            CFRelease(flr[L].ffnFused);
            layer_weights_free(&lw[L]);
            free(Wqt_buf[L]); free(Wkt_buf[L]); free(Wvt_buf[L]); free(Wot_buf[L]);
            free(W1t_buf[L]); free(W3t_buf[L]);
        }
        free(rms_final); free(embed);
        free_kern(fk.sdpaFwd); free_kern(fk.woFwd); free_kern(fk.ffnFused);
        tokenizer_free(&tok);

        return 0;
    }
}
