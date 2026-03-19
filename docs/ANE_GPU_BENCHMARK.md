# GPU vs CPU Benchmark — When to Use Metal

> Tested on M3 Pro (h15g), macOS 26.3.1, Build 25D2128.
> GPU: Apple M3 Pro, 10229 MB max buffer.

---

## Key Finding: GPU Is Not Useful at Current Model Sizes

Apple's CPU (via Accelerate/AMX) is **3-8x faster** than Metal GPU for all training-relevant
operations at Stories110M / Qwen3-0.6B scale. The GPU API is built and ready but provides
no benefit until model sizes grow significantly.

---

## Matmul: CPU cblas_sgemm vs GPU Metal

| Shape (M×K @ K×N) | GFLOP | CPU | GPU | Winner |
|:---|:---|:---|:---|:---|
| 768×256 @ 256×768 | 0.30 | 0.25ms (1.19 TF) | 1.96ms (0.15 TF) | **CPU 8x** |
| 768×256 @ 256×2048 | 0.81 | 0.72ms (1.12 TF) | 2.87ms (0.28 TF) | **CPU 4x** |
| 2048×256 @ 256×768 | 0.81 | 0.49ms (1.66 TF) | 2.75ms (0.29 TF) | **CPU 5.5x** |
| 768×256 @ 256×32000 | 12.58 | 10.53ms (1.20 TF) | 49.17ms (0.26 TF) | **CPU 4.7x** |
| 2048×256 @ 256×2048 | 2.15 | 1.72ms (1.25 TF) | 7.01ms (0.31 TF) | **CPU 4x** |
| 4096×256 @ 256×4096 | 8.59 | 7.70ms (1.12 TF) | 25.71ms (0.33 TF) | **CPU 3.3x** |

CPU achieves **1.12-1.66 TFLOPS** via AMX. GPU peaks at **0.33 TFLOPS**.

## RMSNorm: CPU vs GPU

| Dimensions | CPU | GPU | Winner |
|:---|:---|:---|:---|
| dim=768 seq=256 | <0.001ms | 0.638ms | **CPU >100x** |
| dim=1024 seq=256 | <0.001ms | 0.685ms | **CPU >100x** |
| dim=2048 seq=256 | <0.001ms | 1.235ms | **CPU >100x** |
| dim=4096 seq=256 | <0.001ms | 2.218ms | **CPU >100x** |

RMSNorm is too small for GPU — the Metal dispatch overhead alone (~0.6ms) exceeds total CPU time.

## Softmax: CPU vs GPU

| Dimensions | CPU | GPU | Winner |
|:---|:---|:---|:---|
| vocab=32000 seq=256 | 15.6ms | 41.3ms | **CPU 2.6x** |
| vocab=32000 seq=64 | 4.0ms | 11.5ms | **CPU 2.9x** |
| vocab=151936 seq=64 | 19.6ms | 55.9ms | **CPU 2.9x** |
| vocab=151936 seq=256 | 84.6ms | 183.3ms | **CPU 2.2x** |

Even for large vocab (151k, Qwen3), CPU wins by 2-3x.

---

## Why GPU Loses

1. **Apple AMX is extremely fast** — `cblas_sgemm` uses the Apple Matrix coprocessor,
   achieving 1.2-1.7 TFLOPS for FP32 matmul on CPU. This is purpose-built silicon.

2. **Metal command buffer overhead** — Each GPU operation has ~1-2ms dispatch overhead
   (buffer creation, command encoding, `waitUntilCompleted`). Dominates for small ops.

3. **RMSNorm is microsecond-scale** — CPU finishes before GPU even starts dispatching.

4. **Synchronous GPU calls** — Our implementation uses `waitUntilCompleted` per call.
   A persistent compute pipeline with command buffer batching might help, but adds complexity.

---

## When GPU Would Win

GPU would become beneficial when:
- **Batch size >> 256** — amortizes the ~2ms dispatch overhead
- **Model dim >> 4096** — operations become large enough for GPU parallelism
- **Persistent pipelines** — reuse command buffers instead of recreating each call
- **Operations CPU/AMX can't handle** — complex attention patterns, custom kernels

Estimated crossover point: **~1B+ parameter models** with batch size ≥ 512.

---

## Optimal Configuration for Current Models

```
Stories110M / Qwen3-0.6B:

ANE:  Forward + Backward (Conv/Matmul)     12.13 TFLOPS (fp16)
CPU:  dW Gradients + Adam + RMSNorm         1.2-1.7 TFLOPS (fp32 via AMX)
GPU:  ❌ Not beneficial at this scale

Total pipeline: ANE + CPU is optimal.
GPU API (ane_gpu.h) is built and ready for larger models.
```

---

*Last updated: 2026-03-19 | M3 Pro (h15g), macOS 26.3.1 (25D2128)*
*Source: `libane/bench_gpu.c`*
