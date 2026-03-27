# ANE Use Cases — Honest Assessment

> Based on [rigorous benchmarks](ANE_VS_GPU_BENCHMARK.md) comparing ANE vs GPU on the same M3 Pro chip with powermetrics power measurement.

---

## What We Measured

| Metric | ANE | GPU (MPS) | Winner |
|:---|---:|---:|:---|
| Inference speed | 41.3 tok/s | 47.8 tok/s | GPU 1.2x |
| Inference power | **4.6 W** | 10.9 W | **ANE 2.4x less** |
| **Inference energy/token** | **112.5 mJ** | **229.0 mJ** | **ANE 2.0x** |
| Training speed (Adam step) | 1,875 ms | 850 ms | GPU 2.2x |
| Training power | **8.1 W** | 16.2 W | **ANE 2.0x less** |
| Training energy/step | 15,219 mJ | 13,811 mJ | GPU 1.1x |

---

## Throughput Calculations

All numbers measured on **M3 Pro (h15g)** with **Tiny-ANE 13M** parameters.

### Raw Numbers

| Metric | Value | Source |
|:---|---:|:---|
| Steps/second | **12.6** | Measured (80.9 ms/step, pipeline parallel) |
| Tokens/step | **256** | Sequence length (ctx_len) x batch_size |
| Tokens/second | **3,226** | 12.6 steps/sec x 256 tokens |
| Tokens/hour | **11.6M** | 3,226 x 3,600 |
| Tokens/8h (overnight) | **92.6M** | 11.6M x 8 |

---

## Use Case 1: Energy-Efficient Inference (ANE's Strength)

This is where the ANE genuinely excels — measured and proven.

| Metric | ANE | GPU |
|:---|---:|---:|
| Tokens/sec (Stories-110M) | 41.3 | 47.8 |
| System power | **4.6 W** | 10.9 W |
| ANE chip power | **2.2 W** | 0 W |
| GPU chip power | 0 W | **8.3 W** |
| **Tokens per Joule** | **8.9** | **4.4** |

**2x more tokens per Joule.** The ANE delivers 87% of GPU inference speed at 42% of the power. This is why Apple routes all CoreML models to the ANE — it's the energy-efficient choice.

### When to use ANE for inference:
- Battery-powered operation (2x longer ML battery life)
- GPU busy with rendering, video, gaming
- Always-on ML features (background processing)
- Thermal-constrained environments (ANE stays Nominal, no fan)

---

## Use Case 2: GPU-Free Training

Training on the ANE works — we proved it. But GPU is faster.

### Honest Performance Comparison

| | ANE | GPU | Reality |
|:---|:---|:---|:---|
| Speed | 1,875 ms/step | 850 ms/step | **GPU 2.2x faster** |
| System power | 8.1 W | 16.2 W | **ANE 2x less power** |
| Energy/step | 15.2 J | 13.8 J | **GPU 1.1x more efficient** |
| GPU utilization | **0%** | 100% | **ANE keeps GPU free** |
| CPU utilization | **~60%** | moderate | **ANE loads CPU heavily** |
| Thermal | Nominal | May throttle | **ANE advantage** |

### Hardware Breakdown (Why GPU Wins at Training)

The ANE hardware is fast — only 24% of each training step. The bottleneck is CPU:

| Component | Time | % of Step |
|:----------|---:|---:|
| ANE matmuls | 49 ms | 24% |
| **CPU ops (classifier, gradients, Adam)** | **120 ms** | **59%** |
| IOSurface I/O | 19 ms | 9% |
| Other | 15 ms | 8% |

The GPU runs the entire pipeline on-device. The ANE pipeline must offload optimizer, normalization, and gradient computation to CPU — which dominates each step.

### When ANE training makes sense:
- GPU is busy and can't be interrupted
- You want to train overnight at low power (8.1W vs 16.2W)
- Thermal matters (fanless Mac, hot environment)
- You need GPU free for display/rendering during training

### When GPU training is better:
- Speed matters (2.2x faster)
- CPU is already loaded (ANE training needs ~60% CPU)
- You want maximum energy efficiency per step

---

## Use Case 3: Overnight Fine-Tuning

Start before bed, MacBook charges, ANE+CPU train.

### What 92M Tokens Overnight Buys You

| Scenario | Training Data | Epochs (8h) | Expected Result |
|:---|---:|---:|:---|
| Personal codebase (5M tokens) | Your code | 18 | Strong code completion for your style |
| Company docs (20M tokens) | Internal wiki | 4 | Domain-specific Q&A |
| Research papers (10M tokens) | Your field | 9 | Specialized summarization |

### Power & Thermal (Measured)

| Metric | ANE Training | GPU Training |
|:---|---:|---:|
| System power | **8.1 W** | 16.2 W |
| ANE chip | 0.9 W | 0 W |
| GPU chip | 0.1 W | 14.0 W |
| CPU | 7.1 W | 2.3 W |
| Thermal | **Nominal** | May throttle |
| Fan | Silent | Possibly audible |

> **Note:** The 8.1W system draw includes heavy CPU load for gradient computation. The ANE chip itself draws only ~1W, but the pipeline needs CPU for optimizer, normalization, and gradient ops.

### Energy Cost Overnight (8h)

| | ANE | GPU |
|:---|---:|---:|
| Energy (8h) | **233 Wh** | 467 Wh |
| MacBook battery (70Wh) | Uses ~3.3x battery | Uses ~6.7x battery |
| **Must be plugged in** | **Yes** | **Yes** |

Both need wall power for 8h training. ANE uses half the electricity.

---

## Use Case 4: 100% Private AI

| Concern | Cloud AI | ANE Training |
|:---|:---|:---|
| Data leaves device | Yes | **Never** |
| Requires internet | Yes | **No** |
| API costs | $0.01-0.10/1K tokens | **$0** |
| Account required | Yes | **No** |
| Works offline | No | **Yes** |

The ANE is a hardware accelerator **inside your Mac's SoC**. There is no network interface, no telemetry. Data goes from RAM to ANE SRAM and back. It physically cannot leave the device.

This is equally true for GPU training on your Mac — privacy is a Mac advantage, not ANE-specific.

---

## Use Case 5: Background Training While GPU Works

```
GPU:  Rendering video in DaVinci Resolve      [100% busy]
ANE:  Training your personal AI model          [100% busy]
CPU:  Shared — gradient ops + your apps        [~60% ANE + your work]
                                                    ^
                                          GPU is free.
                                          ANE is separate silicon.
                                          CPU is shared (be aware).
```

> **Important:** ANE training loads the CPU to ~60% for gradient computation, BLAS operations, and optimizer. This WILL impact CPU-heavy tasks (compilation, data processing). It does NOT impact GPU-bound tasks (rendering, gaming, video export).

---

## Comparison: ANE vs MLX vs Cloud

| | ANE Training | ANE Inference | MLX/Metal (GPU) | Cloud (A100/H100) |
|:---|:---|:---|:---|:---|
| **Speed** | ~2 TFLOPS | 722 GFLOPS avg (FP16) | 1449 CPU / 1828 GPU (FP32) | 300+ TFLOPS |
| **Energy efficiency** | 8.1W system | **4.6W system** | 10.9-16.2W | N/A |
| **GFLOPS/Watt** | ~265 | **~2400** | ~290 (CPU) / ~230 (GPU) | N/A |
| **Cost** | $0 | $0 | $0 (but blocks GPU) | $2-8/hour |
| **Privacy** | On-device | On-device | On-device | Data leaves |
| **GPU impact** | None | None | 100% GPU busy | N/A |
| **CPU impact** | **~60%** | Low | Low | N/A |
| **Best for** | Background training | **Energy-efficient inference** | Fast training | Large-scale |

### When to Use What

- **ANE Inference**: Energy-efficient inference, GPU busy, battery operation
- **ANE Training**: GPU is busy, overnight low-power training, thermal-constrained
- **MLX/GPU**: You need speed and the GPU is free
- **Cloud**: Pre-training, or the model is too large for your Mac

---

## What ANE Training is NOT For

| Task | Why Not | Better Alternative |
|:---|:---|:---|
| **Speed-critical training** | GPU is 2.2x faster | MLX/Metal |
| **Pre-training LLMs (>1B)** | 0.6 TFLOPS throughput. A 7B model: months. | Cloud GPU cluster |
| **CPU-heavy multitasking** | ANE training uses ~60% CPU | Wait for GPU to be free |
| **Image generation** | Diffusion models need GPU throughput | Metal/MLX |
| **Models >1B params** | 32MB SRAM, compile budget limits | GPU with unified memory |

### The Right Mental Model

The ANE is a **low-power inference accelerator** that we've taught to train. Think of it as:
- **For inference**: A more efficient alternative to GPU (2x energy savings, proven)
- **For training**: A backup option when GPU is unavailable (slower but uses less power)

---

## Current Status

| Component | Status |
|:---|:---|
| ANE direct access (libane) | Done |
| Dynamic Spatial Packing | Done |
| Training pipeline (forward/backward) | Done |
| ANE vs GPU benchmark (speed + power) | Done |
| Inference energy efficiency (2x proven) | Done |
| Continuous learning daemon | Phase D (planned) |
| LoRA adapter hot-swap | Phase D (planned) |

> See [ROADMAP.md](../ROADMAP.md) for the full plan.
