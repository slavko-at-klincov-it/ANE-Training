# Model Size Benchmark — Training Performance Across Scales

> Measured on M3 Pro (h15g, 18 GB) and M4 (h16g, 16 GB).
> Both synthetic per-step timing AND real training runs.

---

## Real Training Results (Stories-110M)

### Sequential Training (`train_large_ane`)

Actual training run, random init, 100 steps:

| Metric | Value |
|:---|:---|
| **ms/step (sustained)** | **93.0 ms** |
| **Total TFLOPS** | **1.87** (ANE + CPU combined) |
| **ANE TFLOPS** | 1.13 |
| **Compilation** | 86 kernels, 5.3s (one-time per batch) |
| **Compile overhead** | 36% at 100 steps, <5% at 1000 steps |

### Pipeline Parallel Training (`train_pipeline`)

Overlaps CPU backward of step N with ANE forward of step N+1:

| Metric | M3 Pro | M4 |
|:---|:---|:---|
| **ms/step (sustained)** | **80.9 ms** | **71.8 ms** |
| **Total TFLOPS** | **2.15** | **2.43** |
| **ANE TFLOPS** | — | 1.47 |
| **Improvement vs sequential** | 13% faster | 23% faster |

### Why 2.15 TFLOPS and Not 12.79?

Think of it like a truck:
- **ANE Peak (12.79 TFLOPS)** = Empty truck on an open highway, pedal to the metal
- **Sustained Single-Kernel (5.01 TFLOPS)** = Loaded truck on a highway -- steady speed, no stops
- **Real Training (2.15 TFLOPS)** = Fully loaded truck in city traffic -- stops at every intersection (per-layer dispatch), loads/unloads cargo (IOSurface I/O), waits for paperwork (CPU gradients)

| Factor | Explanation |
|:---|:---|
| **12.79 TFLOPS** | ANE silicon peak (128x stacked conv, single dispatch, benchmark only) |
| **Per-layer dispatch** | Each of 12 layers needs separate ANE dispatch (~0.1ms each) |
| **IOSurface I/O** | FP32/FP16 conversion + lock/unlock per kernel (~3-5ms total) |
| **CPU operations** | Residual adds, embedding, cross-entropy, Adam (~20ms) |
| **CPU dW gradients** | cblas_sgemm for 7 weight matrices x 12 layers (~28ms) |
| **2.15 TFLOPS** | Real training throughput including ALL overheads (pipeline) |
| **1.87 TFLOPS** | Real training throughput including ALL overheads (sequential) |

The 12.79 peak is what the ANE silicon can do in isolation. The 2.15 is what a full training step actually achieves.

---

## Synthetic Per-Step Timing (ANE Eval + CPU dW Only) -- NOT Real Training

Measures only ANE eval latency and CPU gradient computation, excluding IOSurface I/O,
embedding, softmax, residual adds, and Adam. ~2.4x faster than real training.
**These TFLOPS numbers are synthetic -- real training achieves 2.15 TFLOPS for Stories-110M.**

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
| ~~GPU for dW gradients~~ | ~~1B+~~ | Tested: GPU 3-8x slower than CPU/AMX, rejected |
| **INT8 W8A8** (M4+ only) | All sizes | +88% ANE throughput |
| **FP16 training** | 600M+ | 2x memory savings → larger models fit |

### Estimated Training Times (Real)

| Model | M3 Pro seq. | M3 Pro pipeline | M4 seq. | M4 pipeline | Steps for 1M tokens |
|:---|:---|:---|:---|:---|:---|
| Tiny-1M | ~3ms | ~3ms | — | — | 15,625 (seq=64) |
| Small-15M | ~12ms | ~11ms | — | — | 7,812 (seq=128) |
| Medium-42M | ~35ms | ~30ms | — | — | 3,906 (seq=256) |
| **Stories-110M** | **93ms** | **80.9ms** | **93.1ms** | **71.8ms** | **3,906** |
| Medium-250M | ~240ms | ~210ms | — | — | 3,906 |
| Large-600M | ~450ms | ~390ms | — | — | 3,906 |

Real times include IOSurface overhead (~2.4x synthetic benchmark).
First batch adds ~5s compilation overhead.
Stories-110M measured: M3 Pro sequential = 1.87 TFLOPS, pipeline = 2.15 TFLOPS. M4 pipeline = 2.43 TFLOPS.

---

### Model Sweep (M4, 200 Steps Each)

| Model | Params | ms/step | Loss@0 | Loss@100 | Loss@200 | Compile |
|:------|-------:|--------:|-------:|---------:|---------:|--------:|
| tiny_ane | 13.3M | 26.0 | 9.16 | 5.89 | 6.18 | 345ms |
| small_ane | 27.6M | 41.0 | 9.11 | 5.89 | 6.15 | 347ms |
| medium_ane | 43.7M | 49.7 | 9.10 | 5.90 | 6.05 | 365ms |
| wide_ane | 52.4M | 47.8 | 9.16 | 5.87 | 5.92 | 414ms |

> See [ANE_M4_BENCHMARK.md](ANE_M4_BENCHMARK.md) for complete M4 training results including long runs and stability analysis.

---

*Last updated: 2026-03-28 | M3 Pro (h15g) + M4 (h16g)*
*Source: `training/bench_model_sizes.m`, `training/train_large_ane`, `benchmark/results_m4/`*
