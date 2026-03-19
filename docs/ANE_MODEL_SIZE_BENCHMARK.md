# Model Size Benchmark — Training Performance Across Scales

> Measured on M3 Pro (h15g), 16 ANE cores, 18 GB RAM.
> Per-step timing for forward + backward pass (no checkpointing overhead).

---

## Results

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

## Key Findings

### 1. CPU dW Gradients Are the Bottleneck Above 42M Params

Below 42M parameters, the ANE dispatch overhead (~0.1ms per kernel) dominates.
Above 42M, the CPU gradient computation (cblas_sgemm via Accelerate/AMX) becomes
the limiting factor, consuming 63-82% of total step time.

```
Tiny/Small:    ANE dispatch dominates (ops too small)
42M-110M:      Sweet spot — ANE and CPU balanced
250M+:         CPU dW dominates (ANE waits for CPU)
1B:            82% CPU time — pipeline parallelism critical
```

### 2. Peak Efficiency at 42-110M Parameters

The ANE achieves its best utilization (1.84-2.10 effective TFLOPS) at 42-110M parameter
models. This is where the ANE compute is large enough to amortize dispatch but the CPU
can still keep up with gradient computation.

### 3. Scaling Behavior

| Scale | ANE % of Step | CPU % of Step | Strategy |
|:---|:---|:---|:---|
| <15M | 71-81% | <20% | ANE-bound — use larger spatial dims |
| 42-110M | 28-37% | 63-72% | Balanced — current optimum |
| 250M+ | 18-23% | 77-82% | CPU-bound — pipeline parallelism needed |
| 1B+ | <18% | >82% | CPU-bound — GPU for dW or pipeline critical |

### 4. Estimated Training Times (M3 Pro)

| Model | ms/step | Steps/min | Time for 1 epoch (1M tokens) |
|:---|:---|:---|:---|
| Tiny-1M | 1.4 | 42,857 | ~0.1 min |
| Small-15M | 4.7 | 12,766 | ~0.3 min |
| Medium-42M | 14.0 | 4,286 | ~0.9 min |
| Stories-110M | 39.2 | 1,531 | ~2.6 min |
| Medium-250M | 99.5 | 603 | ~6.5 min |
| Large-600M | 188.1 | 319 | ~12.3 min |
| XL-1B | 698.3 | 86 | ~45.6 min |

Based on SEQ=256, so 1M tokens ≈ 3906 steps.

### 5. Where Each Optimization Helps

| Optimization | Helps Most At | Expected Gain |
|:---|:---|:---|
| **ANE kernel tuning** (shape optimization) | <42M | +30-50% (amortize dispatch) |
| **DisableIOFences** | All sizes | +1-2% |
| **Pipeline Parallelism** (Forward‖Backward) | 110M+ | +30-40% (overlap ANE+CPU) |
| **GPU for dW gradients** | 600M+ | +50-70% (when CPU is 80%+ bottleneck) |
| **INT8 W8A8** (M4+ only) | All sizes | +88% ANE throughput |

---

## Methodology

- Compiled 3 representative kernels per model size (attention conv, FFN conv, backward conv)
- Forward: ANE eval of dim→dim + dim→hidden convolutions × nlayers
- Backward: ANE eval of hidden→dim + dim→dim convolutions × nlayers
- CPU dW: cblas_sgemm for 7 gradient matrices (Wq,Wk,Wv,Wo,W1,W2,W3) × nlayers
- Excludes: IOSurface I/O overhead (~20%), residual adds, embedding, softmax, Adam update
- Real training steps would be ~20% slower due to excluded overheads

---

*Last updated: 2026-03-19 | M3 Pro (h15g), macOS 26.3.1 (25D2128)*
*Source: `training/bench_model_sizes.m`*
