// bench_inference.m — CPU vs GPU vs ANE Inference Benchmark
// Runs identical matmul workloads on all three backends, measures TFLOPS.
//
// Build & run: cd examples && make bench_inference && ./bench_inference
//
// Backends:
//   CPU: Accelerate framework (cblas_sgemm — dispatches to AMX on Apple Silicon)
//   GPU: Metal compute shader (MPSMatrixMultiplication)
//   ANE: libane (1x1 conv, which IS matmul on the Neural Engine)

#import <Foundation/Foundation.h>
#import <Accelerate/Accelerate.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <mach/mach_time.h>
#import <stdio.h>
#import <stdlib.h>
#import <string.h>
#include "../libane/ane.h"

// ===== Timing =====

static mach_timebase_info_data_t g_tb;

static inline double tb_ms(uint64_t elapsed) {
    return (double)elapsed * g_tb.numer / g_tb.denom / 1e6;
}

// ===== Workload configs =====
// Each config: M (batch/seq), K (input dim), N (output dim)
// Represents typical transformer inference shapes

typedef struct {
    const char *name;
    int M;   // sequence length / batch
    int K;   // input channels
    int N;   // output channels
} WorkloadConfig;

static const WorkloadConfig WORKLOADS[] = {
    {"Small  (256x256,  seq=64)",   64,  256,  256},
    {"Medium (512x512,  seq=64)",   64,  512,  512},
    {"Large  (768x768,  seq=64)",   64,  768,  768},
    {"XL     (1024x1024,seq=64)",   64, 1024, 1024},
    {"FFN    (768x3072, seq=64)",   64,  768, 3072},
    {"Proj   (3072x768, seq=64)",   64, 3072,  768},
    {"Huge   (2048x2048,seq=64)",   64, 2048, 2048},
    {NULL, 0, 0, 0}
};

// ===== CPU Benchmark (Accelerate / AMX) =====

static double bench_cpu(int M, int K, int N, int iters) {
    float *A = (float *)calloc((size_t)M * K, sizeof(float));
    float *B = (float *)calloc((size_t)K * N, sizeof(float));
    float *C = (float *)calloc((size_t)M * N, sizeof(float));

    // Fill with small values to avoid overflow
    for (int i = 0; i < M * K; i++) A[i] = 0.01f * (float)(i % 100);
    for (int i = 0; i < K * N; i++) B[i] = 0.01f * (float)(i % 100);

    // Warmup
    for (int i = 0; i < 3; i++) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
    }

    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < iters; i++) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
    }
    double ms = tb_ms(mach_absolute_time() - t0) / iters;

    free(A); free(B); free(C);
    return ms;
}

// ===== GPU Benchmark (Metal / MPS) =====

static id<MTLDevice> g_mtl_device = nil;
static id<MTLCommandQueue> g_mtl_queue = nil;

static bool gpu_init(void) {
    g_mtl_device = MTLCreateSystemDefaultDevice();
    if (!g_mtl_device) {
        fprintf(stderr, "[GPU] No Metal device found\n");
        return false;
    }
    g_mtl_queue = [g_mtl_device newCommandQueue];
    return g_mtl_queue != nil;
}

static double bench_gpu(int M, int K, int N, int iters) {
    if (!g_mtl_device) return -1;

    // Create MPS matrices
    MPSMatrixDescriptor *descA = [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                                                       columns:K
                                                                      rowBytes:K * sizeof(float)
                                                                      dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *descB = [MPSMatrixDescriptor matrixDescriptorWithRows:K
                                                                       columns:N
                                                                      rowBytes:N * sizeof(float)
                                                                      dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *descC = [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                                                       columns:N
                                                                      rowBytes:N * sizeof(float)
                                                                      dataType:MPSDataTypeFloat32];

    id<MTLBuffer> bufA = [g_mtl_device newBufferWithLength:(size_t)M * K * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufB = [g_mtl_device newBufferWithLength:(size_t)K * N * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufC = [g_mtl_device newBufferWithLength:(size_t)M * N * sizeof(float)
                                                   options:MTLResourceStorageModeShared];

    // Fill with data
    float *pA = (float *)[bufA contents];
    float *pB = (float *)[bufB contents];
    for (int i = 0; i < M * K; i++) pA[i] = 0.01f * (float)(i % 100);
    for (int i = 0; i < K * N; i++) pB[i] = 0.01f * (float)(i % 100);

    MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:bufA descriptor:descA];
    MPSMatrix *matB = [[MPSMatrix alloc] initWithBuffer:bufB descriptor:descB];
    MPSMatrix *matC = [[MPSMatrix alloc] initWithBuffer:bufC descriptor:descC];

    MPSMatrixMultiplication *mmul = [[MPSMatrixMultiplication alloc]
        initWithDevice:g_mtl_device
         transposeLeft:NO
        transposeRight:NO
            resultRows:M
         resultColumns:N
       interiorColumns:K
                 alpha:1.0
                  beta:0.0];

    // Warmup
    for (int i = 0; i < 3; i++) {
        id<MTLCommandBuffer> cmd = [g_mtl_queue commandBuffer];
        [mmul encodeToCommandBuffer:cmd leftMatrix:matA rightMatrix:matB resultMatrix:matC];
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    // Benchmark
    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < iters; i++) {
        id<MTLCommandBuffer> cmd = [g_mtl_queue commandBuffer];
        [mmul encodeToCommandBuffer:cmd leftMatrix:matA rightMatrix:matB resultMatrix:matC];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
    double ms = tb_ms(mach_absolute_time() - t0) / iters;

    return ms;
}

// Also benchmark batched GPU (submit all at once, wait once)
static double bench_gpu_batched(int M, int K, int N, int iters) {
    if (!g_mtl_device) return -1;

    MPSMatrixDescriptor *descA = [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                                                       columns:K
                                                                      rowBytes:K * sizeof(float)
                                                                      dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *descB = [MPSMatrixDescriptor matrixDescriptorWithRows:K
                                                                       columns:N
                                                                      rowBytes:N * sizeof(float)
                                                                      dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *descC = [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                                                       columns:N
                                                                      rowBytes:N * sizeof(float)
                                                                      dataType:MPSDataTypeFloat32];

    id<MTLBuffer> bufA = [g_mtl_device newBufferWithLength:(size_t)M * K * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufB = [g_mtl_device newBufferWithLength:(size_t)K * N * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufC = [g_mtl_device newBufferWithLength:(size_t)M * N * sizeof(float)
                                                   options:MTLResourceStorageModeShared];

    float *pA = (float *)[bufA contents];
    float *pB = (float *)[bufB contents];
    for (int i = 0; i < M * K; i++) pA[i] = 0.01f * (float)(i % 100);
    for (int i = 0; i < K * N; i++) pB[i] = 0.01f * (float)(i % 100);

    MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:bufA descriptor:descA];
    MPSMatrix *matB = [[MPSMatrix alloc] initWithBuffer:bufB descriptor:descB];
    MPSMatrix *matC = [[MPSMatrix alloc] initWithBuffer:bufC descriptor:descC];

    MPSMatrixMultiplication *mmul = [[MPSMatrixMultiplication alloc]
        initWithDevice:g_mtl_device
         transposeLeft:NO
        transposeRight:NO
            resultRows:M
         resultColumns:N
       interiorColumns:K
                 alpha:1.0
                  beta:0.0];

    // Warmup
    for (int i = 0; i < 3; i++) {
        id<MTLCommandBuffer> cmd = [g_mtl_queue commandBuffer];
        [mmul encodeToCommandBuffer:cmd leftMatrix:matA rightMatrix:matB resultMatrix:matC];
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    // Batched: encode all, commit all, wait for last
    uint64_t t0 = mach_absolute_time();
    id<MTLCommandBuffer> lastCmd = nil;
    for (int i = 0; i < iters; i++) {
        id<MTLCommandBuffer> cmd = [g_mtl_queue commandBuffer];
        [mmul encodeToCommandBuffer:cmd leftMatrix:matA rightMatrix:matB resultMatrix:matC];
        [cmd commit];
        lastCmd = cmd;
    }
    [lastCmd waitUntilCompleted];
    double ms = tb_ms(mach_absolute_time() - t0) / iters;

    return ms;
}

// ===== ANE Benchmark =====

static double bench_ane(int M, int K, int N, int iters) {
    // ANE uses 1x1 conv: input [1, K, 1, M], output [1, N, 1, M]
    // This is exactly a matmul: Y[n,m] = sum_k W[n,k] * X[k,m]
    // But K and N must be >= 128 for ANE (error 0x1d otherwise)
    if (K < 128 || N < 128) return -1;

    // Check compile budget
    if (ane_compile_count() >= ANE_COMPILE_SAFE_LIMIT) return -2;

    float *W = (float *)calloc((size_t)N * K, sizeof(float));
    for (int i = 0; i < N * K; i++) W[i] = 0.01f * (float)(i % 100);

    ANEWeight w = ane_weight_fp16("@model_path/weights/weight.bin", W, N, K);
    char *mil = ane_mil_linear(K, N, M, "@model_path/weights/weight.bin");
    free(W);

    size_t in_bytes  = (size_t)K * M * 4;
    size_t out_bytes = (size_t)N * M * 4;

    ANEKernel *k = ane_compile(mil, strlen(mil), &w, 1,
                                1, &in_bytes, 1, &out_bytes,
                                ANE_QOS_BACKGROUND);
    if (!k) {
        free(mil);
        ane_weight_free(&w);
        return -1;
    }

    // Write input
    float *inp = (float *)calloc((size_t)K * M, sizeof(float));
    for (int i = 0; i < K * M; i++) inp[i] = 0.5f;
    ane_write(k, 0, inp, in_bytes);
    free(inp);

    // Warmup
    for (int i = 0; i < 5; i++) ane_eval(k, ANE_QOS_BACKGROUND);

    // Benchmark
    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < iters; i++) ane_eval(k, ANE_QOS_BACKGROUND);
    double ms = tb_ms(mach_absolute_time() - t0) / iters;

    bool spill = ane_sram_spill(k);
    if (spill) printf("    [ANE: SRAM spill detected — throughput reduced ~30%%]\n");

    ane_free(k);
    free(mil);
    ane_weight_free(&w);
    return ms;
}

// ===== Display =====

static void print_bar(double val, double max_val, int width) {
    int fill = (max_val > 0) ? (int)(val / max_val * width + 0.5) : 0;
    if (fill > width) fill = width;
    if (fill < 1 && val > 0) fill = 1;
    for (int i = 0; i < fill; i++) printf("\xe2\x96\x88");
    for (int i = fill; i < width; i++) printf("\xe2\x96\x91");
}

// ===== Main =====

int main(int argc, char **argv) {
    @autoreleasepool {
        mach_timebase_info(&g_tb);

        printf("╔══════════════════════════════════════════════════════════════╗\n");
        printf("║         CPU vs GPU vs ANE — Inference Benchmark            ║\n");
        printf("╚══════════════════════════════════════════════════════════════╝\n\n");

        // --- Init backends ---
        printf("Initializing backends...\n");

        // CPU
        printf("  CPU: Accelerate (cblas_sgemm / AMX) — FP32\n");

        // GPU
        bool has_gpu = gpu_init();
        if (has_gpu) {
            printf("  GPU: %s (MPS MatrixMultiplication) — FP32\n",
                   [[g_mtl_device name] UTF8String]);
        } else {
            printf("  GPU: not available\n");
        }

        // ANE
        int ane_rc = ane_init();
        bool has_ane = (ane_rc == 0);
        if (has_ane) {
            ANEDeviceInfo info = ane_device_info();
            printf("  ANE: %s (%d cores) — FP16 (1x1 conv = matmul)\n",
                   info.arch ? info.arch : "unknown", info.num_cores);
        } else {
            printf("  ANE: not available (init returned %d)\n", ane_rc);
        }

        printf("\nWorkload: Y = X @ W  (matmul / 1x1 conv)\n");
        printf("Metric: GFLOPS = 2*M*N*K / time  (multiply-accumulate)\n");
        printf("Note: ANE runs FP16, CPU/GPU run FP32. ANE needs dims >= 128.\n\n");

        // Iteration count
        int iters = 50;
        if (argc > 1) iters = atoi(argv[1]);
        printf("Iterations per measurement: %d (pass number as arg to change)\n\n", iters);

        // --- Results table ---
        printf("┌────────────────────────────┬──────────────┬──────────────┬──────────────┬──────────────┐\n");
        printf("│ Workload                   │  CPU (GFLOPS)│  GPU (GFLOPS)│ GPU-B(GFLOPS)│  ANE (GFLOPS)│\n");
        printf("├────────────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤\n");

        double max_gflops = 0;

        typedef struct {
            const char *name;
            double cpu_gf, gpu_gf, gpub_gf, ane_gf;
            double cpu_ms, gpu_ms, gpub_ms, ane_ms;
        } Result;
        Result results[16];
        int n_results = 0;

        for (int w = 0; WORKLOADS[w].name; w++) {
            int M = WORKLOADS[w].M;
            int K = WORKLOADS[w].K;
            int N = WORKLOADS[w].N;
            double flops = 2.0 * M * K * N;

            printf("│ %-26s │", WORKLOADS[w].name);
            fflush(stdout);

            // CPU
            double cpu_ms = bench_cpu(M, K, N, iters);
            double cpu_gf = flops / (cpu_ms * 1e6);
            printf(" %8.1f     │", cpu_gf);
            fflush(stdout);

            // GPU (serial)
            double gpu_ms = -1, gpu_gf = 0;
            if (has_gpu) {
                gpu_ms = bench_gpu(M, K, N, iters);
                gpu_gf = (gpu_ms > 0) ? flops / (gpu_ms * 1e6) : 0;
            }
            printf(" %8.1f     │", gpu_gf);
            fflush(stdout);

            // GPU (batched)
            double gpub_ms = -1, gpub_gf = 0;
            if (has_gpu) {
                gpub_ms = bench_gpu_batched(M, K, N, iters);
                gpub_gf = (gpub_ms > 0) ? flops / (gpub_ms * 1e6) : 0;
            }
            printf(" %8.1f     │", gpub_gf);
            fflush(stdout);

            // ANE
            double ane_ms = -1, ane_gf = 0;
            if (has_ane) {
                ane_ms = bench_ane(M, K, N, iters);
                if (ane_ms == -2) {
                    printf("  budget hit  │\n");
                } else if (ane_ms < 0) {
                    printf("  dim<128     │\n");
                } else {
                    ane_gf = flops / (ane_ms * 1e6);
                    printf(" %8.1f     │\n", ane_gf);
                }
            } else {
                printf("      N/A     │\n");
            }

            // Track max
            if (cpu_gf > max_gflops) max_gflops = cpu_gf;
            if (gpu_gf > max_gflops) max_gflops = gpu_gf;
            if (gpub_gf > max_gflops) max_gflops = gpub_gf;
            if (ane_gf > max_gflops) max_gflops = ane_gf;

            results[n_results] = (Result){
                WORKLOADS[w].name,
                cpu_gf, gpu_gf, gpub_gf, ane_gf,
                cpu_ms, gpu_ms, gpub_ms, ane_ms
            };
            n_results++;
        }

        printf("└────────────────────────────┴──────────────┴──────────────┴──────────────┴──────────────┘\n");
        printf("\n  GPU-B = GPU Batched (all commands submitted, single wait)\n");

        // --- Latency table ---
        printf("\n┌────────────────────────────┬──────────┬──────────┬──────────┬──────────┐\n");
        printf("│ Workload                   │ CPU (ms) │ GPU (ms) │GPU-B(ms) │ ANE (ms) │\n");
        printf("├────────────────────────────┼──────────┼──────────┼──────────┼──────────┤\n");

        for (int i = 0; i < n_results; i++) {
            printf("│ %-26s │ %8.3f │ %8.3f │ %8.3f │ %8.3f │\n",
                   results[i].name,
                   results[i].cpu_ms > 0 ? results[i].cpu_ms : 0,
                   results[i].gpu_ms > 0 ? results[i].gpu_ms : 0,
                   results[i].gpub_ms > 0 ? results[i].gpub_ms : 0,
                   results[i].ane_ms > 0 ? results[i].ane_ms : 0);
        }
        printf("└────────────────────────────┴──────────┴──────────┴──────────┴──────────┘\n");

        // --- Visual comparison (bar chart per workload) ---
        printf("\n=== Throughput Comparison (GFLOPS) ===\n\n");
        int bar_width = 40;

        for (int i = 0; i < n_results; i++) {
            printf("  %s\n", results[i].name);

            printf("    CPU  %7.1f  ", results[i].cpu_gf);
            print_bar(results[i].cpu_gf, max_gflops, bar_width);
            printf("\n");

            if (has_gpu) {
                printf("    GPU  %7.1f  ", results[i].gpu_gf);
                print_bar(results[i].gpu_gf, max_gflops, bar_width);
                printf("\n");

                printf("    GPUB %7.1f  ", results[i].gpub_gf);
                print_bar(results[i].gpub_gf, max_gflops, bar_width);
                printf("\n");
            }

            if (has_ane && results[i].ane_gf > 0) {
                printf("    ANE  %7.1f  ", results[i].ane_gf);
                print_bar(results[i].ane_gf, max_gflops, bar_width);
                printf("\n");
            }
            printf("\n");
        }

        // --- Summary ---
        printf("=== Summary ===\n\n");
        double cpu_total = 0, gpu_total = 0, gpub_total = 0, ane_total = 0;
        int cpu_n = 0, gpu_n = 0, gpub_n = 0, ane_n = 0;
        for (int i = 0; i < n_results; i++) {
            if (results[i].cpu_gf > 0) { cpu_total += results[i].cpu_gf; cpu_n++; }
            if (results[i].gpu_gf > 0) { gpu_total += results[i].gpu_gf; gpu_n++; }
            if (results[i].gpub_gf > 0) { gpub_total += results[i].gpub_gf; gpub_n++; }
            if (results[i].ane_gf > 0) { ane_total += results[i].ane_gf; ane_n++; }
        }

        printf("  Average GFLOPS across workloads:\n");
        if (cpu_n)  printf("    CPU:          %7.1f GFLOPS\n", cpu_total / cpu_n);
        if (gpu_n)  printf("    GPU (serial): %7.1f GFLOPS\n", gpu_total / gpu_n);
        if (gpub_n) printf("    GPU (batch):  %7.1f GFLOPS\n", gpub_total / gpub_n);
        if (ane_n)  printf("    ANE:          %7.1f GFLOPS\n", ane_total / ane_n);
        printf("\n");

        if (has_ane) {
            ANEDeviceInfo info = ane_device_info();
            printf("  Chip: %s | Thermal: %s | ANE compiles used: %d/%d\n",
                   info.arch ? info.arch : "?",
                   ane_thermal_state_str(ane_thermal_state()),
                   ane_compile_count(), ANE_COMPILE_BUDGET);
        }

        printf("\n  Notes:\n");
        printf("  - CPU uses FP32 (Accelerate/AMX). GPU uses FP32 (MPS).\n");
        printf("  - ANE uses FP16 internally (1x1 conv = matmul). 2x less memory bandwidth.\n");
        printf("  - GPU serial: encode+commit+wait per iteration (realistic single-inference).\n");
        printf("  - GPU batch: encode+commit all, wait once (best-case pipelined throughput).\n");
        printf("  - ANE dims must be >= 128 (hardware constraint, error 0x1d otherwise).\n");

        return 0;
    }
}
