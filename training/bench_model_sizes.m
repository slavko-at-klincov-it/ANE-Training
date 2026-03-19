// bench_model_sizes.m — Benchmark training step time across model sizes
// Tests compile time, forward eval time, backward eval time, CPU gradient time
// for different model dimensions WITHOUT needing pretrained checkpoints.
//
// Build: xcrun clang -O2 -DACCELERATE_NEW_LAPACK -fobjc-arc -o bench_model_sizes bench_model_sizes.m \
//        -framework Foundation -framework IOSurface -framework Accelerate -ldl
#import <Foundation/Foundation.h>
#import <Accelerate/Accelerate.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>

static mach_timebase_info_data_t g_tb;
static double tb_ms(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }

// ===== Model Configs =====
typedef struct {
    const char *name;
    int dim, hidden, heads, kv_heads, hd, seq, nlayers, vocab;
    double approx_params_m; // millions
} ModelConfig;

static const ModelConfig MODELS[] = {
    // Tiny — smoke test
    {"Tiny-1M",      64,   256,   4,  4,  16, 64,   2,   256,    1.0},
    // Small — quick experiments
    {"Small-15M",   256,   768,   4,  4,  64, 128,  6,  4096,   15.0},
    // Medium-Small
    {"Medium-42M",  512,  1536,   8,  8,  64, 256,  8, 16000,   42.0},
    // Stories110M — our baseline
    {"Stories-110M", 768,  2048,  12, 12,  64, 256, 12, 32000,  110.0},
    // Medium-Large
    {"Medium-250M", 1024,  2816,  16,  8, 64, 256, 16, 32000,  250.0},
    // Qwen3-0.6B scale
    {"Large-600M",  1024,  3072,  16,  8, 128, 256, 28, 32000,  600.0},
    // Large (pushing limits)
    {"XL-1B",       2048,  5504,  32, 16, 64, 256, 22, 32000, 1000.0},
    {NULL, 0, 0, 0, 0, 0, 0, 0, 0, 0}
};

// ===== Helpers =====
static IOSurfaceRef make_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0});
}

static Class g_D, g_I, g_AR, g_AIO;
static SEL g_selMIL, g_selModel, g_selCompile, g_selLoad, g_selEval, g_selUnload;
static SEL g_selHex, g_selSurf, g_selReq;
static int g_compiles = 0;

static int ane_setup(void) {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    g_D   = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_I   = NSClassFromString(@"_ANEInMemoryModel");
    g_AR  = NSClassFromString(@"_ANERequest");
    g_AIO = NSClassFromString(@"_ANEIOSurfaceObject");
    if (!g_D || !g_I || !g_AR || !g_AIO) return -1;
    g_selMIL     = @selector(modelWithMILText:weights:optionsPlist:);
    g_selModel   = @selector(inMemoryModelWithDescriptor:);
    g_selCompile = @selector(compileWithQoS:options:error:);
    g_selLoad    = @selector(loadWithQoS:options:error:);
    g_selEval    = @selector(evaluateWithQoS:options:request:error:);
    g_selUnload  = @selector(unloadWithQoS:error:);
    g_selHex     = @selector(hexStringIdentifier);
    g_selSurf    = @selector(objectWithIOSurface:);
    g_selReq     = @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:);
    return 0;
}

// Compile + load a single conv1x1 kernel (simulates one layer forward)
static id compile_conv(int ch_in, int ch_out, int sp) {
    _Float16 *w = (_Float16*)calloc(ch_in * ch_out, sizeof(_Float16));
    for (int i = 0; i < (ch_in < ch_out ? ch_in : ch_out); i++) w[i*ch_in+i] = (_Float16)0.01f;
    int ws = ch_in * ch_out * 2, tot = 128 + ws;
    uint8_t *blob = (uint8_t*)calloc(tot, 1);
    blob[0]=1; blob[4]=2; blob[64]=0xEF; blob[65]=0xBE; blob[66]=0xAD; blob[67]=0xDE; blob[68]=1;
    *(uint32_t*)(blob+72)=ws; *(uint32_t*)(blob+80)=128;
    memcpy(blob+128, w, ws);
    NSData *wdata = [NSData dataWithBytesNoCopy:blob length:tot freeWhenDone:YES];
    free(w);

    NSString *mil = [NSString stringWithFormat:
        @"program(1.3)\n"
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n"
        "{\n"
        "    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
        "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];\n"
        "        tensor<fp16, [1,%d,1,%d]> x16 = cast(dtype=to16,x=x)[name=string(\"cin\")];\n"
        "        tensor<fp16, [%d,%d,1,1]> W = const()[name=string(\"W\"), "
        "val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(64)))];\n"
        "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
        "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
        "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
        "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
        "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
        "        tensor<fp16, [1,%d,1,%d]> y16 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W,x=x16)"
        "[name=string(\"conv\")];\n"
        "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];\n"
        "        tensor<fp32, [1,%d,1,%d]> y = cast(dtype=to32,x=y16)[name=string(\"cout\")];\n"
        "    } -> (y);\n"
        "}\n", ch_in, sp, ch_in, sp, ch_out, ch_in, ch_out, ch_in, ch_out, sp, ch_out, sp];

    NSData *md = [mil dataUsingEncoding:NSUTF8StringEncoding];
    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D, g_selMIL,
        md, @{@"@model_path/weights/weight.bin": @{@"offset":@0, @"data":wdata}}, nil);
    id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(g_I, g_selModel, desc);
    id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, g_selHex);
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    NSFileManager *fm = [NSFileManager defaultManager];
    [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
        withIntermediateDirectories:YES attributes:nil error:nil];
    [md writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    [wdata writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

    NSError *e = nil;
    uint64_t t0 = mach_absolute_time();
    BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, g_selCompile, 9, @{}, &e);
    if (!ok) return nil;
    ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, g_selLoad, 9, @{}, &e);
    g_compiles++;
    return mdl;
}

// Time a single eval
static double time_eval(id mdl, int ch_in, int ch_out, int sp, int iters) {
    size_t inBytes = ch_in * sp * 4;
    size_t outBytes = ch_out * sp * 4;
    IOSurfaceRef ioIn = make_surface(inBytes);
    IOSurfaceRef ioOut = make_surface(outBytes);
    id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, g_selSurf, ioIn);
    id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, g_selSurf, ioOut);
    id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR, g_selReq,
        @[wI], @[@0], @[wO], @[@0], nil, nil, @0);

    IOSurfaceLock(ioIn, 0, NULL);
    float *inp = (float*)IOSurfaceGetBaseAddress(ioIn);
    for (int i = 0; i < ch_in * sp; i++) inp[i] = 0.01f;
    IOSurfaceUnlock(ioIn, 0, NULL);

    // Warmup
    NSError *e = nil;
    for (int i = 0; i < 3; i++)
        ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(mdl, g_selEval, 9,
            @{@"kANEFDisableIOFencesUseSharedEventsKey": @YES}, req, &e);

    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < iters; i++)
        ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(mdl, g_selEval, 9,
            @{@"kANEFDisableIOFencesUseSharedEventsKey": @YES}, req, &e);
    double ms = tb_ms(mach_absolute_time() - t0) / iters;

    CFRelease(ioIn); CFRelease(ioOut);
    return ms;
}

// Time CPU cblas matmul (simulates dW gradient computation)
static double time_cpu_matmul(int M, int K, int N, int iters) {
    float *A = (float*)calloc(M * K, sizeof(float));
    float *B = (float*)calloc(K * N, sizeof(float));
    float *C = (float*)calloc(M * N, sizeof(float));
    for (int i = 0; i < M*K; i++) A[i] = 0.01f;
    for (int i = 0; i < K*N; i++) B[i] = 0.01f;

    // Warmup
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);

    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < iters; i++)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
    double ms = tb_ms(mach_absolute_time() - t0) / iters;

    free(A); free(B); free(C);
    return ms;
}

int main() {
    @autoreleasepool {
        setbuf(stdout, NULL);
        mach_timebase_info(&g_tb);

        printf("\n  ██████ MODEL SIZE BENCHMARK ██████\n\n");

        if (ane_setup() != 0) { printf("  ERROR: ANE init failed\n"); return 1; }

        printf("  %-14s  %6s  %5s %5s %4s %3s %5s  %8s %8s %8s %8s  %8s\n",
            "Model", "Params", "Dim", "Hid", "Hds", "Lyr", "Vocab",
            "ANE Fwd", "ANE Bwd", "CPU dW", "Step",  "TFLOPS");
        printf("  %-14s  %6s  %5s %5s %4s %3s %5s  %8s %8s %8s %8s  %8s\n",
            "--------------", "------", "-----", "-----", "----", "---", "-----",
            "--------", "--------", "--------", "--------", "--------");

        for (int m = 0; MODELS[m].name; m++) {
            const ModelConfig *mc = &MODELS[m];

            // Check compile budget
            // Each model needs: nlayers * 2 (fwd attn + fwd ffn) + 3 (bwd kernels) ≈ nlayers*2 + 3
            int kernels_needed = 3; // We test 3 representative kernels
            if (g_compiles + kernels_needed >= 110) {
                printf("  %-14s  Skipped (compile budget: %d/119)\n", mc->name, g_compiles);
                continue;
            }

            // Compile representative kernels for this model size
            // Forward attention: dim→dim conv
            id fwdAttn = compile_conv(mc->dim, mc->dim, mc->seq);
            if (!fwdAttn) {
                printf("  %-14s  COMPILE FAILED (dim=%d)\n", mc->name, mc->dim);
                continue;
            }

            // Forward FFN: dim→hidden conv
            id fwdFFN = compile_conv(mc->dim, mc->hidden, mc->seq);
            if (!fwdFFN) {
                printf("  %-14s  COMPILE FAILED (hidden=%d)\n", mc->name, mc->hidden);
                // Unload attn
                NSError *e = nil;
                ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(fwdAttn, g_selUnload, 9, &e);
                continue;
            }

            // Backward: hidden→dim conv (transpose shape)
            id bwd = compile_conv(mc->hidden, mc->dim, mc->seq);
            if (!bwd) {
                printf("  %-14s  COMPILE FAILED (bwd hidden=%d)\n", mc->name, mc->hidden);
                NSError *e = nil;
                ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(fwdAttn, g_selUnload, 9, &e);
                ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(fwdFFN, g_selUnload, 9, &e);
                continue;
            }

            int eval_iters = (mc->dim <= 512) ? 50 : 20;

            // ANE Forward: attention + FFN per layer
            double ane_attn_ms = time_eval(fwdAttn, mc->dim, mc->dim, mc->seq, eval_iters);
            double ane_ffn_ms = time_eval(fwdFFN, mc->dim, mc->hidden, mc->seq, eval_iters);
            double ane_fwd_total = (ane_attn_ms + ane_ffn_ms) * mc->nlayers;

            // ANE Backward: ~same as forward (similar shapes)
            double ane_bwd_ms = time_eval(bwd, mc->hidden, mc->dim, mc->seq, eval_iters);
            double ane_bwd_total = (ane_attn_ms + ane_bwd_ms) * mc->nlayers; // attn bwd ≈ fwd

            // CPU dW gradients: 5 matmuls per layer (Wq,Wk,Wv,Wo,W1,W2,W3 → simplified to 3 representative)
            int cblas_iters = (mc->dim <= 512) ? 20 : 5;
            double cpu_dw_attn = time_cpu_matmul(mc->dim, mc->seq, mc->dim, cblas_iters); // dWq ~= dWk ~= dWv ~= dWo
            double cpu_dw_ffn  = time_cpu_matmul(mc->hidden, mc->seq, mc->dim, cblas_iters); // dW1 ~= dW3
            double cpu_dw_ffn2 = time_cpu_matmul(mc->dim, mc->seq, mc->hidden, cblas_iters); // dW2
            double cpu_dw_total = (4 * cpu_dw_attn + 2 * cpu_dw_ffn + cpu_dw_ffn2) * mc->nlayers;

            // Total step time (simplified — no IOSurface overhead, no residual adds)
            double total_step = ane_fwd_total + ane_bwd_total + cpu_dw_total;

            // Compute TFLOPS: rough estimate of total FLOPs per step
            // Per layer: ~12 * dim^2 * seq (fwd+bwd matmuls) + 4 * dim * hidden * seq (FFN fwd+bwd)
            double flops_per_layer = (12.0 * mc->dim * mc->dim * mc->seq +
                                      4.0 * mc->dim * mc->hidden * mc->seq) * 2; // *2 for fwd+bwd
            double total_gflops = flops_per_layer * mc->nlayers / 1e9;
            double tflops = total_gflops / (total_step / 1000.0) / 1000.0;

            printf("  %-14s  %5.0fM  %5d %5d %4d %3d %5d  %6.1fms %6.1fms %6.1fms %6.1fms  %6.2f TF\n",
                mc->name, mc->approx_params_m,
                mc->dim, mc->hidden, mc->heads, mc->nlayers, mc->vocab,
                ane_fwd_total, ane_bwd_total, cpu_dw_total, total_step, tflops);

            // Unload
            NSError *e = nil;
            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(fwdAttn, g_selUnload, 9, &e);
            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(fwdFFN, g_selUnload, 9, &e);
            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(bwd, g_selUnload, 9, &e);
        }

        printf("\n  Compiles used: %d / 119\n", g_compiles);
        printf("  Note: Times are per training step (1 forward + 1 backward pass).\n");
        printf("  Excludes IOSurface I/O overhead (~20%%), residual adds, embedding.\n");
        printf("  CPU dW uses cblas_sgemm (Accelerate/AMX).\n\n");
    }
    return 0;
}
