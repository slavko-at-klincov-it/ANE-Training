// bench_gpu.c — Compare GPU (Metal) vs CPU (Accelerate) for training-relevant ops
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <Accelerate/Accelerate.h>
#include "ane.h"
#include "ane_gpu.h"

static double time_ms(struct timespec *t0, struct timespec *t1) {
    return (t1->tv_sec - t0->tv_sec) * 1e3 + (t1->tv_nsec - t0->tv_nsec) / 1e6;
}

static void bench_matmul(int M, int K, int N, int iters) {
    float *A = malloc(M * K * sizeof(float));
    float *B = malloc(K * N * sizeof(float));
    float *C_cpu = malloc(M * N * sizeof(float));
    float *C_gpu = malloc(M * N * sizeof(float));

    for (int i = 0; i < M * K; i++) A[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < K * N; i++) B[i] = (float)(rand() % 100) / 100.0f;

    double gflop = 2.0 * M * K * N / 1e9;

    // Warmup
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0f, A, K, B, N, 0.0f, C_cpu, N);
    ane_gpu_matmul(A, B, C_gpu, M, K, N);

    // CPU benchmark
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < iters; i++)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K, 1.0f, A, K, B, N, 0.0f, C_cpu, N);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double cpu_ms = time_ms(&t0, &t1) / iters;
    double cpu_tflops = gflop / cpu_ms;

    // GPU benchmark
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < iters; i++)
        ane_gpu_matmul(A, B, C_gpu, M, K, N);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double gpu_ms = time_ms(&t0, &t1) / iters;
    double gpu_tflops = gflop / gpu_ms;

    double speedup = cpu_ms / gpu_ms;
    printf("  %4dx%-4d @ %-4dx%-4d  %6.2f GFLOP  CPU %7.2fms (%5.2f TF)  GPU %7.2fms (%5.2f TF)  %5.2fx %s\n",
        M, K, K, N, gflop, cpu_ms, cpu_tflops, gpu_ms, gpu_tflops,
        speedup, speedup > 1.1 ? "GPU wins" : (speedup < 0.9 ? "CPU wins" : "~tied"));

    free(A); free(B); free(C_cpu); free(C_gpu);
}

static void bench_rmsnorm(int dim, int seq, int iters) {
    float *x = malloc(dim * seq * sizeof(float));
    float *w = malloc(dim * sizeof(float));
    float *out_cpu = malloc(dim * seq * sizeof(float));
    float *out_gpu = malloc(dim * seq * sizeof(float));

    for (int i = 0; i < dim * seq; i++) x[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < dim; i++) w[i] = 1.0f;

    // CPU RMSNorm
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int it = 0; it < iters; it++) {
        for (int s = 0; s < seq; s++) {
            float ss = 0;
            for (int d = 0; d < dim; d++) ss += x[s * dim + d] * x[s * dim + d];
            ss = 1.0f / sqrtf(ss / dim + 1e-5f);
            for (int d = 0; d < dim; d++) out_cpu[s * dim + d] = x[s * dim + d] * ss * w[d];
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double cpu_ms = time_ms(&t0, &t1) / iters;

    // GPU RMSNorm
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int it = 0; it < iters; it++)
        ane_gpu_rmsnorm(x, w, out_gpu, dim, seq);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double gpu_ms = time_ms(&t0, &t1) / iters;

    double speedup = cpu_ms / gpu_ms;
    printf("  dim=%-4d seq=%-4d  CPU %7.3fms  GPU %7.3fms  %5.2fx %s\n",
        dim, seq, cpu_ms, gpu_ms, speedup,
        speedup > 1.1 ? "GPU wins" : (speedup < 0.9 ? "CPU wins" : "~tied"));

    free(x); free(w); free(out_cpu); free(out_gpu);
}

static void bench_softmax(int vocab, int seq, int iters) {
    float *x = malloc(vocab * seq * sizeof(float));
    float *out_cpu = malloc(vocab * seq * sizeof(float));
    float *out_gpu = malloc(vocab * seq * sizeof(float));

    for (int i = 0; i < vocab * seq; i++) x[i] = (float)(rand() % 100) / 100.0f - 0.5f;

    // CPU Softmax
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int it = 0; it < iters; it++) {
        for (int s = 0; s < seq; s++) {
            float mx = -1e30f;
            for (int v = 0; v < vocab; v++) if (x[s * vocab + v] > mx) mx = x[s * vocab + v];
            float sum = 0;
            for (int v = 0; v < vocab; v++) {
                out_cpu[s * vocab + v] = expf(x[s * vocab + v] - mx);
                sum += out_cpu[s * vocab + v];
            }
            for (int v = 0; v < vocab; v++) out_cpu[s * vocab + v] /= sum;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double cpu_ms = time_ms(&t0, &t1) / iters;

    // GPU Softmax
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int it = 0; it < iters; it++)
        ane_gpu_softmax(x, out_gpu, vocab, seq);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double gpu_ms = time_ms(&t0, &t1) / iters;

    double speedup = cpu_ms / gpu_ms;
    printf("  vocab=%-5d seq=%-4d  CPU %7.3fms  GPU %7.3fms  %5.2fx %s\n",
        vocab, seq, cpu_ms, gpu_ms, speedup,
        speedup > 1.1 ? "GPU wins" : (speedup < 0.9 ? "CPU wins" : "~tied"));

    free(x); free(out_cpu); free(out_gpu);
}

int main() {
    printf("\n  \xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88 GPU vs CPU BENCHMARK \xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\n\n");

    if (ane_gpu_init() != 0) {
        printf("  ERROR: Metal GPU not available\n");
        return 1;
    }
    ANEGPUInfo gi = ane_gpu_info();
    printf("  GPU: %s (max buffer: %llu MB)\n\n", gi.name, gi.max_buffer_length >> 20);

    // ── Matmul ──
    printf("  ── Matrix Multiply (CPU cblas_sgemm vs GPU Metal) ──\n\n");
    // Training-relevant sizes: dW gradients
    bench_matmul(768, 256, 768, 50);      // Small (dW attention)
    bench_matmul(768, 256, 2048, 50);     // Medium (dW FFN W1)
    bench_matmul(2048, 256, 768, 50);     // Medium (dW FFN W2)
    bench_matmul(768, 256, 32000, 10);    // Large (dW embedding/classifier)
    bench_matmul(2048, 256, 2048, 20);    // Large (dW FFN square)
    bench_matmul(4096, 256, 4096, 10);    // Very large

    // ── RMSNorm ──
    printf("\n  ── RMSNorm (CPU loop vs GPU Metal) ──\n\n");
    bench_rmsnorm(768, 256, 100);         // Stories110M
    bench_rmsnorm(1024, 256, 100);        // Qwen3-0.6B
    bench_rmsnorm(2048, 256, 50);         // Larger model
    bench_rmsnorm(4096, 256, 50);         // Large model

    // ── Softmax ──
    printf("\n  ── Softmax (CPU loop vs GPU Metal) ──\n\n");
    bench_softmax(32000, 256, 20);        // Stories110M vocab
    bench_softmax(32000, 64, 50);         // Shorter seq
    bench_softmax(151936, 64, 5);         // Qwen3 vocab (large!)
    bench_softmax(151936, 256, 2);        // Qwen3 full

    printf("\n");
    ane_gpu_shutdown();
    return 0;
}
