// test_gpu.c — Test Metal GPU acceleration (standalone, no ANE dependency)
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "ane_gpu.h"

#define PASS "\033[32mPASS\033[0m"
#define FAIL "\033[31mFAIL\033[0m"

static int g_tests = 0, g_passed = 0;

static void check(const char *name, int ok) {
    g_tests++;
    if (ok) { g_passed++; printf("  [%s] %s\n", PASS, name); }
    else    { printf("  [%s] %s\n", FAIL, name); }
}

// ===== Reference CPU implementations for verification =====

static void cpu_matmul(const float *A, const float *B, float *C, int M, int K, int N) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0;
            for (int k = 0; k < K; k++) s += A[i*K+k] * B[k*N+j];
            C[i*N+j] = s;
        }
}

static void cpu_rmsnorm(const float *x, const float *w, float *out, int dim, int seq) {
    for (int s = 0; s < seq; s++) {
        float ss = 0;
        for (int i = 0; i < dim; i++) { float v = x[s*dim+i]; ss += v*v; }
        float rms = 1.0f / sqrtf(ss / dim + 1e-6f);
        for (int i = 0; i < dim; i++) out[s*dim+i] = x[s*dim+i] * w[i] * rms;
    }
}

static void cpu_softmax(const float *x, float *out, int vocab, int seq) {
    for (int s = 0; s < seq; s++) {
        float mx = x[s*vocab];
        for (int i = 1; i < vocab; i++) if (x[s*vocab+i] > mx) mx = x[s*vocab+i];
        float sum = 0;
        for (int i = 0; i < vocab; i++) { out[s*vocab+i] = expf(x[s*vocab+i]-mx); sum += out[s*vocab+i]; }
        for (int i = 0; i < vocab; i++) out[s*vocab+i] /= sum;
    }
}

static float max_diff(const float *a, const float *b, int n) {
    float md = 0;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > md) md = d;
    }
    return md;
}

static void fill_random(float *buf, int n) {
    for (int i = 0; i < n; i++) buf[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
}

int main(void) {
    printf("=== ane_gpu test suite ===\n\n");

    // --- Init ---
    printf("1. Initialization\n");
    int rc = ane_gpu_init();
    check("ane_gpu_init() returns 0", rc == 0);
    check("ane_has_gpu() returns true", ane_has_gpu());

    if (rc != 0) {
        printf("\nMetal not available, skipping remaining tests.\n");
        return 1;
    }

    // Double init is safe
    check("double init is safe", ane_gpu_init() == 0);

    // --- Device info ---
    printf("\n2. Device Info\n");
    ANEGPUInfo info = ane_gpu_info();
    printf("  Device: %s\n", info.name ? info.name : "(null)");
    printf("  Max buffer: %llu MB\n", info.max_buffer_length >> 20);
    printf("  Shared events: %s\n", info.supports_shared_events ? "yes" : "no");
    check("device name not null", info.name != NULL);
    check("max buffer > 0", info.max_buffer_length > 0);

    // --- Matmul ---
    printf("\n3. Matrix Multiply (C = A @ B)\n");
    {
        // Small test: 4x3 @ 3x5 = 4x5
        int M = 4, K = 3, N = 5;
        float A[] = {1,2,3, 4,5,6, 7,8,9, 10,11,12};
        float B[] = {1,0,0,0,1, 0,1,0,1,0, 0,0,1,0,0};
        float C_gpu[20], C_cpu[20];

        ane_gpu_matmul(A, B, C_gpu, M, K, N);
        cpu_matmul(A, B, C_cpu, M, K, N);
        float diff = max_diff(C_gpu, C_cpu, M*N);
        printf("  4x3 @ 3x5 max diff: %.6f\n", diff);
        check("small matmul matches CPU (tol 1e-4)", diff < 1e-4f);
    }
    {
        // Larger test: 64x128 @ 128x64
        int M = 64, K = 128, N = 64;
        float *A = malloc(M*K*sizeof(float));
        float *B = malloc(K*N*sizeof(float));
        float *C_gpu = malloc(M*N*sizeof(float));
        float *C_cpu = malloc(M*N*sizeof(float));
        srand(42);
        fill_random(A, M*K);
        fill_random(B, K*N);

        ane_gpu_matmul(A, B, C_gpu, M, K, N);
        cpu_matmul(A, B, C_cpu, M, K, N);
        float diff = max_diff(C_gpu, C_cpu, M*N);
        printf("  64x128 @ 128x64 max diff: %.6f\n", diff);
        check("large matmul matches CPU (tol 1e-2)", diff < 1e-2f);
        free(A); free(B); free(C_gpu); free(C_cpu);
    }

    // --- RMSNorm ---
    printf("\n4. RMSNorm\n");
    {
        int dim = 64, seq = 8;
        float *x = malloc(seq*dim*sizeof(float));
        float *w = malloc(dim*sizeof(float));
        float *out_gpu = malloc(seq*dim*sizeof(float));
        float *out_cpu = malloc(seq*dim*sizeof(float));
        srand(123);
        fill_random(x, seq*dim);
        fill_random(w, dim);

        ane_gpu_rmsnorm(x, w, out_gpu, dim, seq);
        cpu_rmsnorm(x, w, out_cpu, dim, seq);
        float diff = max_diff(out_gpu, out_cpu, seq*dim);
        printf("  dim=64 seq=8 max diff: %.6f\n", diff);
        check("rmsnorm matches CPU (tol 1e-4)", diff < 1e-4f);
        free(x); free(w); free(out_gpu); free(out_cpu);
    }

    // --- Softmax ---
    printf("\n5. Softmax\n");
    {
        int vocab = 128, seq = 4;
        float *x = malloc(seq*vocab*sizeof(float));
        float *out_gpu = malloc(seq*vocab*sizeof(float));
        float *out_cpu = malloc(seq*vocab*sizeof(float));
        srand(456);
        fill_random(x, seq*vocab);

        ane_gpu_softmax(x, out_gpu, vocab, seq);
        cpu_softmax(x, out_cpu, vocab, seq);
        float diff = max_diff(out_gpu, out_cpu, seq*vocab);
        printf("  vocab=128 seq=4 max diff: %.6f\n", diff);
        check("softmax matches CPU (tol 1e-5)", diff < 1e-5f);

        // Verify softmax sums to 1.0
        float sum = 0;
        for (int i = 0; i < vocab; i++) sum += out_gpu[i];
        printf("  row 0 sum: %.6f\n", sum);
        check("softmax row sums to ~1.0 (tol 1e-5)", fabsf(sum - 1.0f) < 1e-5f);
        free(x); free(out_gpu); free(out_cpu);
    }

    // --- Sync ---
    printf("\n6. Sync API\n");
    if (info.supports_shared_events) {
        ANEGPUSync *sync = ane_gpu_sync_create();
        check("sync create returns non-NULL", sync != NULL);
        if (sync) {
            ane_gpu_sync_signal(sync, 1);
            ane_gpu_sync_wait(sync, 1);
            check("signal+wait round-trip works", 1);
            ane_gpu_sync_free(sync);
        }
    } else {
        printf("  (shared events not supported, skipping)\n");
    }

    // --- Shutdown ---
    printf("\n7. Shutdown\n");
    ane_gpu_shutdown();
    check("shutdown succeeds", !ane_has_gpu());
    check("double shutdown is safe", (ane_gpu_shutdown(), !ane_has_gpu()));

    // Re-init after shutdown
    rc = ane_gpu_init();
    check("re-init after shutdown works", rc == 0);
    ane_gpu_shutdown();

    // --- Summary ---
    printf("\n=== Results: %d/%d passed ===\n", g_passed, g_tests);
    return (g_passed == g_tests) ? 0 : 1;
}
