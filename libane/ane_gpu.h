// ane_gpu.h — Optional Metal GPU acceleration for ANE Training
// Loads Metal.framework at runtime via dlopen (NOT linked at compile time).
// If Metal is unavailable (e.g., VM), ane_gpu_init() returns -1 gracefully.
// All Metal objects accessed via objc_msgSend (same pattern as ane.m).
//
// Usage:
//   ane_gpu_init();
//   ANEGPUInfo info = ane_gpu_info();
//   ane_gpu_matmul(A, B, C, M, K, N);
//   ane_gpu_shutdown();

#ifndef ANE_GPU_H
#define ANE_GPU_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ===== Initialization =====

// Initialize Metal GPU (loads framework at runtime via dlopen).
// Returns 0 on success, -1 if Metal not available.
int ane_gpu_init(void);

// Check if GPU is available and initialized.
bool ane_has_gpu(void);

// ===== Device Info =====

typedef struct {
    const char *name;               // e.g. "Apple M3 Pro"
    uint64_t max_buffer_length;
    bool supports_shared_events;
} ANEGPUInfo;

ANEGPUInfo ane_gpu_info(void);

// ===== ANE <-> GPU Synchronization =====
// Uses MTLSharedEvent / IOSurfaceSharedEvent for zero-copy sync.

typedef struct ANEGPUSync ANEGPUSync;

// Create a shared sync event between ANE and GPU.
// Returns opaque handle, or NULL on failure.
ANEGPUSync *ane_gpu_sync_create(void);
void ane_gpu_sync_free(ANEGPUSync *sync);

// Signal from ANE side (call after ane_eval).
void ane_gpu_sync_signal(ANEGPUSync *sync, uint64_t value);

// Wait on GPU side (call before GPU compute).
void ane_gpu_sync_wait(ANEGPUSync *sync, uint64_t value);

// ===== GPU Compute Operations =====
// Metal compute shaders, synchronous from caller's perspective.

// Matrix multiply on GPU: C = A @ B
// A: [M, K], B: [K, N], C: [M, N], all float32
void ane_gpu_matmul(const float *A, const float *B, float *C,
                    int M, int K, int N);

// RMSNorm on GPU: out[i] = x[i] * weight[i] / rms(x)
// x: [seq, dim], weight: [dim], out: [seq, dim]
void ane_gpu_rmsnorm(const float *x, const float *weight, float *out,
                     int dim, int seq);

// Softmax on GPU: out = softmax(x) along last dimension
// x: [seq, vocab], out: [seq, vocab]
void ane_gpu_softmax(const float *x, float *out, int vocab, int seq);

// ===== Cleanup =====

void ane_gpu_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif // ANE_GPU_H
