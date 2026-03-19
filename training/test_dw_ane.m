// test_dw_ane.m — Proof of concept: dW gradient computation on ANE
//
// Tests whether ANE can compute dW = grad @ act^T faster than cblas_sgemm.
// Uses Dynamic Spatial Packing: compile ONE kernel per shape, pack grad+act
// into a single IOSurface input, ANE computes the matmul.
//
// The shapes per layer (Stories110M):
//   dWq/Wk/Wv/Wo: grad=[768,256], act=[768,256]  -> dW=[768,768]
//   dW1/W3:        grad=[2048,256], act=[768,256]  -> dW=[2048,768]
//   dW2:           grad=[768,256], act=[2048,256]  -> dW=[768,2048]
//
// Build: make test_dw_ane
// Run:   ./test_dw_ane

#import <Foundation/Foundation.h>
#import <Accelerate/Accelerate.h>
#import <mach/mach_time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../libane/ane.h"

// Stories110M dimensions
#define DIM 768
#define HIDDEN 2048
#define SEQ 256
#define NLAYERS 12

// Number of benchmark iterations
#define WARMUP 3
#define BENCH_ITERS 20

static mach_timebase_info_data_t g_tb;
static double tb_ms(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }

// Generate MIL program for dW computation:
//   Input:  [1, out_dim + in_dim, 1, SEQ] fp32
//     - channels [0..out_dim): gradient
//     - channels [out_dim..out_dim+in_dim): activation
//   Output: [1, out_dim, 1, in_dim] fp32  (the dW matrix)
//
// Operation: dW = grad @ act^T
//   grad is [out_dim, SEQ], act is [in_dim, SEQ]
//   grad @ act^T = [out_dim, in_dim]
//
// MIL plan:
//   1. Cast input to fp16
//   2. Slice grad: [1, out_dim, 1, SEQ]
//   3. Slice act:  [1, in_dim, 1, SEQ]
//   4. Reshape grad to [1, 1, out_dim, SEQ]
//   5. Reshape act to [1, 1, in_dim, SEQ]
//   6. Transpose act to [1, 1, SEQ, in_dim]
//   7. matmul(grad_reshaped, act_transposed) = [1, 1, out_dim, in_dim]
//   8. Reshape to [1, out_dim, 1, in_dim]
//   9. Cast back to fp32
static char *gen_dw_mil(int out_dim, int in_dim, int seq) {
    int total_ch = out_dim + in_dim;
    size_t bufsize = 8192;
    char *buf = (char *)malloc(bufsize);
    if (!buf) return NULL;

    snprintf(buf, bufsize,
        "program(1.3)\n"
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n{\n"
        "  func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"

        // Cast to fp16
        "    string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];\n"
        "    tensor<fp16, [1, %d, 1, %d]> xh = cast(dtype=to16, x=x)[name=string(\"cin\")];\n"

        // Slice gradient: channels [0..out_dim)
        "    tensor<int32, [4]> bg = const()[name=string(\"bg\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
        "    tensor<int32, [4]> sg = const()[name=string(\"sg\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n"
        "    tensor<fp16, [1,%d,1,%d]> grad = slice_by_size(x=xh,begin=bg,size=sg)[name=string(\"sgrad\")];\n"

        // Slice activation: channels [out_dim..out_dim+in_dim)
        "    tensor<int32, [4]> ba = const()[name=string(\"ba\"), val=tensor<int32, [4]>([0,%d,0,0])];\n"
        "    tensor<int32, [4]> sa = const()[name=string(\"sa\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n"
        "    tensor<fp16, [1,%d,1,%d]> act = slice_by_size(x=xh,begin=ba,size=sa)[name=string(\"sact\")];\n"

        // Reshape grad: [1, out_dim, 1, SEQ] -> [1, 1, out_dim, SEQ]
        "    tensor<int32, [4]> grs = const()[name=string(\"grs\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n"
        "    tensor<fp16, [1,1,%d,%d]> gr = reshape(shape=grs,x=grad)[name=string(\"rg\")];\n"

        // Reshape act: [1, in_dim, 1, SEQ] -> [1, 1, in_dim, SEQ]
        "    tensor<int32, [4]> ars = const()[name=string(\"ars\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n"
        "    tensor<fp16, [1,1,%d,%d]> ar = reshape(shape=ars,x=act)[name=string(\"ra\")];\n"

        // Transpose act: [1, 1, in_dim, SEQ] -> [1, 1, SEQ, in_dim]
        "    tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"
        "    tensor<fp16, [1,1,%d,%d]> at = transpose(perm=pm,x=ar)[name=string(\"ta\")];\n"

        // matmul: [1,1,out_dim,SEQ] @ [1,1,SEQ,in_dim] = [1,1,out_dim,in_dim]
        "    bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"
        "    tensor<fp16, [1,1,%d,%d]> mm = matmul(transpose_x=bF,transpose_y=bF,x=gr,y=at)[name=string(\"mm\")];\n"

        // Reshape: [1,1,out_dim,in_dim] -> [1,out_dim,1,in_dim]
        "    tensor<int32, [4]> os = const()[name=string(\"os\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n"
        "    tensor<fp16, [1,%d,1,%d]> yr = reshape(shape=os,x=mm)[name=string(\"yr\")];\n"

        // Cast back to fp32
        "    string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];\n"
        "    tensor<fp32, [1,%d,1,%d]> y = cast(dtype=to32,x=yr)[name=string(\"cout\")];\n"
        "  } -> (y);\n}\n",

        // Arguments in order:
        total_ch, seq,         // input shape
        total_ch, seq,         // cast shape
        out_dim, seq,          // sg (grad slice size)
        out_dim, seq,          // grad shape
        out_dim,               // ba (act begin channel)
        in_dim, seq,           // sa (act slice size)
        in_dim, seq,           // act shape
        out_dim, seq,          // grs (grad reshape)
        out_dim, seq,          // gr shape
        in_dim, seq,           // ars (act reshape)
        in_dim, seq,           // ar shape
        seq, in_dim,           // at shape (transposed)
        out_dim, in_dim,       // mm shape (matmul output)
        out_dim, in_dim,       // os (output reshape)
        out_dim, in_dim,       // yr shape
        out_dim, in_dim        // y shape (output)
    );
    return buf;
}

// Pack gradient and activation into a single IOSurface for the dW kernel
static void pack_dw_input(ANEKernel *k, const float *grad, const float *act,
                           int out_dim, int in_dim, int seq) {
    ane_lock_input(k, 0);
    float *ptr = (float *)ane_input_ptr(k, 0);
    int total_ch = out_dim + in_dim;
    memset(ptr, 0, (size_t)total_ch * seq * sizeof(float));

    // grad [out_dim, seq] -> channels [0..out_dim)
    for (int c = 0; c < out_dim; c++)
        memcpy(ptr + c * seq, grad + c * seq, seq * sizeof(float));

    // act [in_dim, seq] -> channels [out_dim..out_dim+in_dim)
    for (int c = 0; c < in_dim; c++)
        memcpy(ptr + (out_dim + c) * seq, act + c * seq, seq * sizeof(float));

    ane_unlock_input(k, 0);
}

// Read dW output from IOSurface
static void read_dw_output(ANEKernel *k, float *dW, int out_dim, int in_dim) {
    ane_lock_output(k, 0);
    float *ptr = (float *)ane_output_ptr(k, 0);
    // Output: [1, out_dim, 1, in_dim] -> dW[out_dim][in_dim]
    for (int i = 0; i < out_dim; i++)
        memcpy(dW + i * in_dim, ptr + i * in_dim, in_dim * sizeof(float));
    ane_unlock_output(k, 0);
}

// Compute dW on CPU using cblas_sgemm for reference
// dW = grad @ act^T
// grad: [out_dim, seq], act: [in_dim, seq]
// dW: [out_dim, in_dim]
static void cpu_dw(const float *grad, const float *act, float *dW,
                   int out_dim, int in_dim, int seq) {
    // C = alpha * A * B^T + beta * C
    // A = grad [out_dim x seq], B = act [in_dim x seq]
    // C = dW [out_dim x in_dim]
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                out_dim, in_dim, seq,
                1.0f, grad, seq, act, seq,
                0.0f, dW, in_dim);
}

// Compare two matrices and report max/mean absolute error
static void compare_results(const float *ref, const float *test, int n,
                             const char *label, float *max_err, float *mean_err) {
    float mx = 0, sum = 0;
    for (int i = 0; i < n; i++) {
        float err = fabsf(ref[i] - test[i]);
        if (err > mx) mx = err;
        sum += err;
    }
    *max_err = mx;
    *mean_err = sum / n;

    // Also compute relative error
    float ref_norm = 0;
    for (int i = 0; i < n; i++) ref_norm += ref[i] * ref[i];
    ref_norm = sqrtf(ref_norm);

    float diff_norm = 0;
    for (int i = 0; i < n; i++) diff_norm += (ref[i] - test[i]) * (ref[i] - test[i]);
    diff_norm = sqrtf(diff_norm);

    float rel_err = (ref_norm > 0) ? diff_norm / ref_norm : 0;
    printf("  [%s] max_err=%.6f  mean_err=%.6f  rel_err=%.6f\n",
           label, mx, *mean_err, rel_err);
}

typedef struct {
    const char *name;
    int out_dim;
    int in_dim;
    int count_per_layer;  // how many per layer
} DWShape;

int main(void) {
    @autoreleasepool {
        mach_timebase_info(&g_tb);
        srand(42);

        printf("=== ANE dW Gradient Computation Test ===\n\n");

        // Initialize ANE
        if (ane_init() != 0) {
            printf("FAIL: Cannot initialize ANE\n");
            return 1;
        }
        ANEDeviceInfo info = ane_device_info();
        printf("Hardware: %s, %d cores\n", info.arch, info.num_cores);
        printf("Model dims: DIM=%d HIDDEN=%d SEQ=%d NLAYERS=%d\n\n", DIM, HIDDEN, SEQ, NLAYERS);

        // Three distinct dW shapes
        DWShape shapes[] = {
            { "dWq/Wk/Wv/Wo", DIM,    DIM,    4 },   // 4 per layer
            { "dW1/W3",        HIDDEN, DIM,    2 },   // 2 per layer
            { "dW2",           DIM,    HIDDEN, 1 },   // 1 per layer
        };
        int nshapes = sizeof(shapes) / sizeof(shapes[0]);

        // Total dW computations per backward: 4+2+1 = 7 per layer, 12 layers = 84
        int total_ops = 0;
        for (int s = 0; s < nshapes; s++) total_ops += shapes[s].count_per_layer * NLAYERS;
        printf("Total dW matmuls per backward pass: %d\n\n", total_ops);

        // ===== Phase 1: Accuracy test for each shape =====
        printf("--- Phase 1: Accuracy Test ---\n\n");

        ANEKernel *kernels[3] = {NULL, NULL, NULL};
        int all_pass = 1;

        for (int s = 0; s < nshapes; s++) {
            int od = shapes[s].out_dim;
            int id = shapes[s].in_dim;
            printf("Shape: %s  grad=[%d,%d] act=[%d,%d] -> dW=[%d,%d]\n",
                   shapes[s].name, od, SEQ, id, SEQ, od, id);

            // Generate and compile MIL
            char *mil = gen_dw_mil(od, id, SEQ);
            if (!mil) { printf("  FAIL: MIL generation\n"); all_pass = 0; continue; }

            int total_ch = od + id;
            size_t in_bytes = (size_t)total_ch * SEQ * sizeof(float);
            size_t out_bytes = (size_t)od * id * sizeof(float);

            printf("  Input:  %d channels x %d seq = %.1f KB\n", total_ch, SEQ, in_bytes/1024.0);
            printf("  Output: %d x %d = %.1f KB\n", od, id, out_bytes/1024.0);

            uint64_t t0 = mach_absolute_time();
            ANEKernel *k = ane_compile(mil, strlen(mil), NULL, 0,
                                        1, &in_bytes, 1, &out_bytes,
                                        ANE_QOS_BACKGROUND);
            uint64_t t1 = mach_absolute_time();
            printf("  Compile: %.1f ms", tb_ms(t1 - t0));

            if (!k) {
                printf(" FAILED!\n");
                free(mil);
                all_pass = 0;
                continue;
            }
            printf("  (SRAM spill: %s)\n", ane_sram_spill(k) ? "YES" : "no");
            kernels[s] = k;
            free(mil);

            // Create test data
            float *grad = (float *)malloc(od * SEQ * sizeof(float));
            float *act  = (float *)malloc(id * SEQ * sizeof(float));
            float *dW_cpu = (float *)malloc(od * id * sizeof(float));
            float *dW_ane = (float *)malloc(od * id * sizeof(float));

            // Random data with small values (avoid fp16 overflow)
            for (int i = 0; i < od * SEQ; i++) grad[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
            for (int i = 0; i < id * SEQ; i++) act[i]  = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;

            // CPU reference
            cpu_dw(grad, act, dW_cpu, od, id, SEQ);

            // ANE computation
            pack_dw_input(k, grad, act, od, id, SEQ);
            ane_eval(k, ANE_QOS_BACKGROUND);
            read_dw_output(k, dW_ane, od, id);

            // Compare
            float max_err, mean_err;
            compare_results(dW_cpu, dW_ane, od * id, shapes[s].name, &max_err, &mean_err);

            // FP16 tolerance: values are ~0.05 magnitude, SEQ=256 accumulations
            // Expected precision: ~SEQ * max_val^2 * eps_fp16 ~ 256 * 0.005 * 0.001 ~ 0.001
            float tolerance = 0.01f;
            if (max_err > tolerance) {
                printf("  FAIL: max error %.6f exceeds tolerance %.6f\n", max_err, tolerance);
                all_pass = 0;
            } else {
                printf("  PASS\n");
            }
            printf("\n");

            free(grad); free(act); free(dW_cpu); free(dW_ane);
        }

        // ===== Phase 2: Performance benchmark =====
        printf("--- Phase 2: Performance Benchmark ---\n");
        printf("  %d warmup + %d timed iterations per shape\n\n", WARMUP, BENCH_ITERS);

        double total_cpu_ms = 0, total_ane_ms = 0;
        double total_cpu_flops = 0;

        printf("%-16s  %6s  %6s  %8s  %8s  %6s  %8s\n",
               "Shape", "MxN", "K", "CPU(ms)", "ANE(ms)", "Ratio", "GFLOPS");
        printf("%-16s  %6s  %6s  %8s  %8s  %6s  %8s\n",
               "----------------", "------", "------", "--------", "--------", "------", "--------");

        for (int s = 0; s < nshapes; s++) {
            if (!kernels[s]) {
                printf("%-16s  SKIPPED (compile failed)\n", shapes[s].name);
                continue;
            }
            int od = shapes[s].out_dim;
            int id = shapes[s].in_dim;
            int count = shapes[s].count_per_layer;

            // Allocate test data
            float *grad = (float *)malloc(od * SEQ * sizeof(float));
            float *act  = (float *)malloc(id * SEQ * sizeof(float));
            float *dW   = (float *)malloc(od * id * sizeof(float));
            for (int i = 0; i < od * SEQ; i++) grad[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
            for (int i = 0; i < id * SEQ; i++) act[i]  = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;

            // FLOPs for one matmul: 2 * out_dim * in_dim * seq
            double flops_one = 2.0 * od * id * SEQ;

            // --- CPU benchmark ---
            // Warmup
            for (int i = 0; i < WARMUP; i++) cpu_dw(grad, act, dW, od, id, SEQ);

            uint64_t t0 = mach_absolute_time();
            for (int i = 0; i < BENCH_ITERS; i++) cpu_dw(grad, act, dW, od, id, SEQ);
            uint64_t t1 = mach_absolute_time();
            double cpu_ms = tb_ms(t1 - t0) / BENCH_ITERS;
            double cpu_gflops = flops_one / (cpu_ms * 1e6);

            // --- ANE benchmark ---
            // Warmup
            for (int i = 0; i < WARMUP; i++) {
                pack_dw_input(kernels[s], grad, act, od, id, SEQ);
                ane_eval(kernels[s], ANE_QOS_BACKGROUND);
                read_dw_output(kernels[s], dW, od, id);
            }

            // Timed: include pack + eval + read (full pipeline)
            t0 = mach_absolute_time();
            for (int i = 0; i < BENCH_ITERS; i++) {
                pack_dw_input(kernels[s], grad, act, od, id, SEQ);
                ane_eval(kernels[s], ANE_QOS_BACKGROUND);
                read_dw_output(kernels[s], dW, od, id);
            }
            t1 = mach_absolute_time();
            double ane_ms = tb_ms(t1 - t0) / BENCH_ITERS;
            double ane_gflops = flops_one / (ane_ms * 1e6);

            // Also benchmark ANE eval only (no pack/read overhead)
            t0 = mach_absolute_time();
            for (int i = 0; i < BENCH_ITERS; i++) {
                ane_eval(kernels[s], ANE_QOS_BACKGROUND);
            }
            t1 = mach_absolute_time();
            double ane_eval_ms = tb_ms(t1 - t0) / BENCH_ITERS;
            double ane_eval_gflops = flops_one / (ane_eval_ms * 1e6);

            double ratio = cpu_ms / ane_ms;

            printf("%-16s  %4dx%-4d  %4d  %7.3f  %7.3f  %5.2fx  CPU:%.0f ANE:%.0f(eval:%.0f)\n",
                   shapes[s].name, od, id, SEQ, cpu_ms, ane_ms, ratio,
                   cpu_gflops, ane_gflops, ane_eval_gflops);

            // Aggregate for total backward pass estimate
            double ops_per_bwd = count * NLAYERS;
            total_cpu_ms += cpu_ms * ops_per_bwd;
            total_ane_ms += ane_ms * ops_per_bwd;
            total_cpu_flops += flops_one * ops_per_bwd;

            free(grad); free(act); free(dW);
        }

        // ===== Phase 3: Full backward pass estimate =====
        printf("\n--- Phase 3: Full Backward Pass dW Estimate ---\n\n");
        printf("Per backward pass (%d layers x 7 dW matmuls):\n", NLAYERS);
        printf("  CPU total:  %.2f ms\n", total_cpu_ms);
        printf("  ANE total:  %.2f ms\n", total_ane_ms);
        printf("  Speedup:    %.2fx\n", total_cpu_ms / total_ane_ms);
        printf("  CPU TFLOPS: %.3f\n", total_cpu_flops / (total_cpu_ms * 1e9));
        printf("  ANE TFLOPS: %.3f\n", total_cpu_flops / (total_ane_ms * 1e9));
        printf("  Total FLOPs per backward dW: %.2f GFLOPS\n", total_cpu_flops / 1e9);

        // Context: current pipeline spends ~28ms on CPU dW
        printf("\n  Context: current pipeline CPU dW time = ~28 ms\n");
        if (total_ane_ms < 28.0) {
            printf("  -> ANE dW (%.1f ms) would SAVE %.1f ms per step (%.0f%% of dW time)\n",
                   total_ane_ms, 28.0 - total_ane_ms, (28.0 - total_ane_ms) / 28.0 * 100);
        } else {
            printf("  -> ANE dW (%.1f ms) is SLOWER than CPU (28 ms)\n", total_ane_ms);
        }

        printf("\n  NOTE: ANE is single-threaded, so dW would serialize with\n");
        printf("  backward ANE kernels. The win comes from freeing CPU for\n");
        printf("  other work, or if ANE dW is truly faster end-to-end.\n");

        // ===== Phase 4: Compile budget check =====
        printf("\n--- Phase 4: Compile Budget ---\n");
        printf("  Kernels compiled: %d (3 shapes)\n", ane_compile_count());
        printf("  Budget remaining: %d / %d\n",
               ANE_COMPILE_BUDGET - ane_compile_count(), ANE_COMPILE_BUDGET);
        printf("  In production: compile 3 dW kernels at startup, reuse forever\n");

        // Cleanup
        for (int s = 0; s < nshapes; s++) {
            if (kernels[s]) ane_free(kernels[s]);
        }

        printf("\n=== %s ===\n", all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
        return all_pass ? 0 : 1;
    }
}
