# Model Size Benchmark — Training Performance Across Scales

> Measured on M3 Pro (h15g), 16 ANE cores, 18 GB RAM.
> Both synthetic per-step timing AND real training runs.

---

## Real Training Results (Stories-110M)

Actual training run with `train_large_ane`, random init, 100 steps:

| Metric | Value |
|:---|:---|
| **ms/step (sustained)** | **93.0 ms** |
| **Total TFLOPS** | **1.87** (ANE + CPU combined) |
| **ANE TFLOPS** | 1.13 |
| **Compilation** | 86 kernels, 5.3s (one-time per batch) |
| **Compile overhead** | 36% at 100 steps, <5% at 1000 steps |

### Why 1.87 TFLOPS and Not 12.79?

| Factor | Explanation |
|:---|:---|
| **12.79 TFLOPS** | Theoretical ANE peak (128x stacked conv, single dispatch) |
| **Per-layer dispatch** | Each of 12 layers needs separate ANE dispatch (~0.1ms each) |
| **IOSurface I/O** | FP32↔FP16 conversion + lock/unlock per kernel (~3-5ms total) |
| **CPU operations** | Residual adds, embedding, cross-entropy, Adam (~20ms) |
| **CPU dW gradients** | cblas_sgemm for 7 weight matrices × 12 layers (~28ms) |
| **1.87 TFLOPS** | Real training throughput including ALL overheads |

The 12.79 peak is what the ANE silicon can do. The 1.87 is what a full training step achieves.

---

## Synthetic Per-Step Timing (ANE Eval + CPU dW Only)

Measures only ANE eval latency and CPU gradient computation, excluding IOSurface I/O,
embedding, softmax, residual adds, and Adam. ~2.4x faster than real training.

| Model | Params | Dim | Hidden | Layers | Step Time | ANE | CPU dW | TFLOPS | Bottleneck |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| **Tiny-1M** | 1M | 64 | 256 | 2 | 1.4ms | 1.0ms (71%) | 0.0ms | 0.02 | ANE Dispatch |
| **Small-15M** | 15M | 256 | 768 | 6 | 4.7ms | 3.8ms (81%) | 0.9ms | 0.51 | ANE Dispatch |
| **Medium-42M** | 42M | 512 | 1536 | 8 | 14.0ms | 5.2ms (37%) | 8.8ms | 1.84 | CPU dW |
| **Stories-110M** | 110M | 768 | 2048 | 12 | 39.2ms | 10.9ms (28%) | 28.3ms | 2.10 | CPU dW |
| **Medium-250M** | 250M | 1024 | 2816 | 16 | 99.5ms | 22.7ms (23%) | 76.7ms | 1.99 | CPU dW |
| **Large-600M** | 600M | 1024 | 3072 | 28 | 188.1ms | 38.2ms (20%) | 149.9ms | 1.92 | CPU dW |
| **XL-1B** | 1B | 2048 | 5504 | 22 | 698.3ms | 128.8ms (18%) | 569.4ms | 1.54 | CPU dW |

---

## Memory Limits (18 GB RAM, M3 Pro)

Training requires: weights (FP32) + gradients (FP32) + Adam states (2× FP32) + activations.

| Model | Weights | + Gradients | + Adam | + Activations | Total | Fits 18GB? |
|:---|:---|:---|:---|:---|:---|:---|
| Tiny-1M | 4 MB | 8 MB | 16 MB | ~2 MB | ~30 MB | Yes |
| Small-15M | 60 MB | 120 MB | 240 MB | ~20 MB | ~440 MB | Yes |
| Medium-42M | 168 MB | 336 MB | 672 MB | ~50 MB | ~1.2 GB | Yes |
| Stories-110M | 440 MB | 880 MB | 1.76 GB | ~120 MB | ~3.2 GB | Yes |
| Medium-250M | 1.0 GB | 2.0 GB | 4.0 GB | ~250 MB | ~7.3 GB | Yes |
| Large-600M | 2.4 GB | 4.8 GB | 9.6 GB | ~500 MB | ~17.3 GB | Tight |
| XL-1B | 4.0 GB | 8.0 GB | 16.0 GB | ~1 GB | ~29 GB | **No** |
| 3B | 12 GB | 24 GB | 48 GB | ~3 GB | ~87 GB | **No** |
| 12B | 48 GB | 96 GB | 192 GB | ~10 GB | ~346 GB | **No** |

**Maximum trainable on M3 Pro (18GB): ~600M-800M parameters in FP32.**

To train larger:
- **FP16 weights + gradients**: 2x less memory → up to ~1.5B
- **LoRA/QLoRA**: Only train adapter layers → up to ~7B (inference weights in INT4)
- **More RAM**: M4 Pro 48GB → up to ~3B FP32, ~6B FP16

---

## Scaling Behavior

### Where Each Optimization Helps

| Optimization | Helps Most At | Expected Gain |
|:---|:---|:---|
| **ANE kernel tuning** (shape optimization) | <42M | +30-50% (amortize dispatch) |
| **DisableIOFences** | All sizes | +1-2% |
| **Pipeline Parallelism** (Forward‖Backward) | 110M+ | +30-40% (overlap ANE+CPU) |
| **GPU for dW gradients** | 1B+ | +50-70% (when CPU is 80%+ bottleneck) |
| **INT8 W8A8** (M4+ only) | All sizes | +88% ANE throughput |
| **FP16 training** | 600M+ | 2x memory savings → larger models fit |

### Estimated Training Times (Real, M3 Pro)

| Model | ms/step | Steps for 1M tokens | Time |
|:---|:---|:---|:---|
| Tiny-1M | ~3ms | 15,625 (seq=64) | ~47s |
| Small-15M | ~12ms | 7,812 (seq=128) | ~1.6 min |
| Medium-42M | ~35ms | 3,906 (seq=256) | ~2.3 min |
| **Stories-110M** | **93ms** | **3,906** | **~6.0 min** |
| Medium-250M | ~240ms | 3,906 | ~15.6 min |
| Large-600M | ~450ms | 3,906 | ~29.3 min |

Real times include IOSurface overhead (~2.4x synthetic benchmark).
First batch adds ~5s compilation overhead.

---

*Last updated: 2026-03-19 | M3 Pro (h15g), macOS 26.3.1 (25D2128)*
*Source: `training/bench_model_sizes.m`, `training/train_large_ane`*
