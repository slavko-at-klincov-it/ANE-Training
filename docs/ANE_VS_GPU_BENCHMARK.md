# ANE vs GPU (MPS) Benchmark

Direct comparison of **training** and **inference** on **Apple Neural Engine** vs **Metal GPU (MPS)** — same chip, same models, same weights.

**Hardware:** Apple M3 Pro (11-core CPU, 14-core GPU, 16-core ANE, 18GB RAM)
**Framework:** ANE = libane (custom C/ObjC pipeline), GPU = PyTorch 2.9.1 MPS
**Date:** 2026-03-24

---

## TL;DR

| Task | Model | ANE | GPU | Speed | Energy |
|:-----|:------|---:|---:|:---|:---|
| **Inference** | Stories-110M | 41.3 tok/s | 47.8 tok/s | GPU 1.1x faster | **ANE 2.0x efficient** |
| Training | Stories-110M | 0.64 TFLOPS | 1.50 TFLOPS | GPU 2.3x faster | ~equal |

**Key insight:** ANE inference is nearly as fast as GPU (within 10-15%) but uses **2x less energy per token** — at 4.6W vs 10.9W system power. For training, GPU is faster but ANE uses half the power, resulting in roughly equal energy per optimizer step.

---

## Part 1: Inference Benchmark

### Results

| Model | ANE ms/token | GPU ms/token | ANE tok/s | GPU tok/s | ANE TFLOPS | GPU TFLOPS |
|:------|---:|---:|---:|---:|---:|---:|
| **Tiny-ANE-15M** | 4.5 | 3.9 | 221 | 255 | 0.58 | 0.67 |
| **Stories-110M** | 24.0 | 21.0 | 41.7 | 47.6 | 1.81 | 2.07 |

> Both use **full-sequence recompute** (no KV cache) for a fair comparison. Same trained checkpoint loaded on both.

### What this means

- ANE inference is **within 10-20% of GPU speed** — remarkably close for a fixed-function accelerator vs a general-purpose GPU
- The larger the model, the **smaller the gap** (1.2x → 1.1x)
- ANE achieves this while keeping **GPU at 0% utilization** — the GPU is free for display, other apps, or parallel inference
- On battery, ANE inference likely uses significantly **less power** (dedicated low-power accelerator vs GPU)

### Why this matters

Apple routes all CoreML inference to the ANE for a reason — it's nearly as fast as GPU but far more power-efficient. Our benchmark confirms this with actual measurements on a real transformer model.

---

## Part 2: Training Benchmark

### Results

### Summary Table

| Model | ANE (ms/step) | GPU (ms/step) | ANE TFLOPS | GPU TFLOPS | Speed Winner |
|:------|---:|---:|---:|---:|:---|
| **Tiny-ANE-15M** | 955 | 190 | 0.082 | 0.414 | GPU 5.0x faster |
| **Stories-110M** | 2,025 | 870 | 0.644 | 1.501 | GPU 2.3x faster |

> **"Step"** = one Adam optimizer update = 10 micro-steps (gradient accumulation).
> Both systems use identical hyperparameters: lr=3e-4, AdamW(beta2=0.95), cosine schedule, grad_clip=1.0.

### Tiny-ANE-15M (6 layers, dim=256)

| Metric | ANE | GPU (MPS) |
|:-------|---:|---:|
| Adam step time | 955 ms | 190 ms |
| Micro-step | 95.5 ms | 19.0 ms |
| Throughput | 0.082 TFLOPS | 0.414 TFLOPS |
| Final loss (100 updates) | 10.393 | 10.353 |

**ANE hardware breakdown per micro-step:**

| Component | Time | % of Step |
|:----------|---:|---:|
| ANE hardware (matmuls) | 13.1 ms | 14% |
| CPU ops (classifier, RMSNorm, SiLU, Adam) | 77.5 ms | **81%** |
| IOSurface I/O | 2.3 ms | 2% |
| Other overhead | 2.6 ms | 3% |

### Stories-110M (12 layers, dim=768)

| Metric | ANE | GPU (MPS) |
|:-------|---:|---:|
| Adam step time | 2,025 ms | 870 ms |
| Micro-step | 202.5 ms | 87.0 ms |
| Throughput | 0.644 TFLOPS | 1.501 TFLOPS |
| Final loss (100 updates) | 10.451 | 10.404 |

**ANE hardware breakdown per micro-step:**

| Component | Time | % of Step |
|:----------|---:|---:|
| ANE hardware (matmuls) | 49.0 ms | **24%** |
| CPU ops (classifier, RMSNorm, SiLU, Adam) | 120.2 ms | **59%** |
| IOSurface I/O | 18.5 ms | 9% |
| Other overhead | 14.8 ms | 8% |

## Key Findings

### 1. GPU is faster for training on the same chip

The M3 Pro GPU outperforms the ANE pipeline **5x for small models** and **2.3x for larger models**. The gap narrows with model size because the ANE's share of compute increases.

### 2. The ANE hardware itself is not the bottleneck

The ANE silicon is fast — only 13-49ms per micro-step. The bottleneck is **CPU overhead** (81% for Tiny, 59% for Stories-110M), primarily:
- **Classifier/embedding gradients** (cblas_sgemm for 32K vocab, runs on CPU/AMX)
- **Weight gradients** (cblas_sgemm for dW computation)
- **RMSNorm, SiLU, Adam** (FP32 CPU ops, cannot run on FP16-only ANE)

### 3. The scaling trend favors ANE at larger models

| Model | Params | ANE % of step | CPU % of step | GPU speed advantage |
|:------|---:|---:|---:|:---|
| Tiny-ANE-15M | 5.1M | 14% | 81% | 5.0x |
| Stories-110M | 85M | 24% | 59% | 2.3x |
| *Projected 1B+* | *1B* | *~40%+* | *~40%* | *~1.5x or less* |

As models grow, ANE matmuls dominate and CPU overhead becomes a smaller fraction. At some model size, the ANE pipeline could match or exceed GPU — but practical limits (memory, compile budget) may prevent reaching that crossover on current hardware.

### 4. ANE keeps GPU free

During ANE training, the GPU is **completely idle** (0% utilization, confirmed by `hw_monitor.h`). This means:
- GPU is available for other tasks (rendering, inference, display)
- On battery, the GPU power domain can stay in low-power state
- ANE training does not compete with GPU workloads

### 5. Convergence is equivalent

Both pipelines converge at the same rate — the final loss is within noise. The ANE's FP16 matmuls (with loss scaling) produce gradients equivalent to the GPU's FP32 matmuls for this model size.

## Why the ANE pipeline pays CPU overhead

The ANE only supports **FP16 inference-style evaluation** — no native backward pass, no FP32, no optimizer. Our training pipeline works around this:

```
ANE handles:     Weight matmuls (QKV, Wo, FFN projections) — the big ops
CPU handles:     Everything else — RMSNorm, SiLU, cross-entropy, Adam, BLAS dW
IOSurface:       Shared memory between CPU and ANE (zero-copy but has setup cost)
```

The GPU (via PyTorch MPS) runs the **entire pipeline on-device** — forward, backward, optimizer — with no data transfer overhead.

## What would make ANE competitive

1. **Move classifier to ANE** — the 32K vocab matmul is the single biggest CPU cost (~65ms). If compiled as an ANE kernel, this would eliminate ~60% of CPU time.
2. **Larger models** — as dimension grows, ANE matmul fraction increases and CPU fraction decreases.
3. **Batch size > 1** — amortizes fixed per-step overhead over more data.
4. **FP32 on ANE** — Apple's M3+ ANE may support FP32 internally; if unlocked, could eliminate CPU RMSNorm/Adam.

## Part 3: Power & Energy Efficiency

> Measured with `sudo powermetrics` (500ms sampling) on M3 Pro, Stories-110M.

### Idle Baseline

| Rail | Power |
|:-----|---:|
| CPU | 272 mW |
| GPU | 28 mW |
| ANE | 0 mW |

### Power Rails During Workloads

| Power Rail | ANE Inference | GPU Inference | ANE Training | GPU Training |
|:-----------|---:|---:|---:|---:|
| **ANE** | **2,168 mW** | 0 mW | **910 mW** | 0 mW |
| **GPU** | 33 mW | **8,268 mW** | 98 mW | **13,959 mW** |
| **CPU** | 2,446 mW | 2,674 mW | **7,108 mW** | 2,286 mW |
| **System total** | **4.6 W** | **10.9 W** | **8.1 W** | **16.2 W** |

Key observations:
- **ANE inference draws 2.2W** from the ANE rail — the GPU stays at idle (33 mW)
- **GPU inference draws 8.3W** from the GPU rail — the ANE stays at 0
- During **ANE training**, CPU dominates (7.1W) because of gradient/optimizer ops on CPU
- **GPU training pulls 14W** from the GPU rail alone

### Energy Efficiency

#### Inference (the ANE's strength)

| Metric | ANE | GPU | Winner |
|:-------|---:|---:|:---|
| Speed | 41.3 tok/s | 47.8 tok/s | GPU 1.2x |
| System power | 4.6 W | 10.9 W | **ANE 2.4x less** |
| **Energy/token** | **112.5 mJ** | **229.0 mJ** | **ANE 2.0x better** |
| **Tokens/Joule** | **8.9** | **4.4** | **ANE 2.0x better** |

**The ANE generates tokens at 87% of GPU speed but uses only 42% of the power.** Net result: **2x more tokens per Joule**.

This is why Apple routes all CoreML inference to the ANE — it's nearly as fast but far more energy-efficient.

#### Training

| Metric | ANE | GPU | Winner |
|:-------|---:|---:|:---|
| Speed (Adam step) | 1,875 ms | 850 ms | GPU 2.2x |
| System power | 8.1 W | 16.2 W | ANE 2.0x less |
| **Energy/step** | **15,219 mJ** | **13,811 mJ** | **GPU 1.1x better** |

For training, GPU's speed advantage (2.2x) slightly outweighs ANE's power advantage (2.0x), making them **roughly equal in energy per optimizer step**. The GPU finishes faster but burns more Watts; the ANE takes longer but sips power.

### What This Means for On-Device ML

| Use Case | Best Choice | Why |
|:---------|:------------|:----|
| **Inference on battery** | **ANE** | 2x more tokens per charge |
| **Inference while multitasking** | **ANE** | GPU stays free for UI/gaming |
| **Training (speed priority)** | **GPU** | 2.2x faster per step |
| **Training (power/thermal)** | **ANE** | Half the power, no GPU throttling |
| **Always-on ML** | **ANE** | Low sustained power, no thermal issues |

---

## Reproducing

```bash
cd benchmark

# Inference benchmark (~30 seconds)
bash run_inference_benchmark.sh 100

# Training benchmark (~3 minutes)
bash run_benchmark.sh 100

# Power & energy efficiency (~5 minutes, requires root)
sudo bash run_power_benchmark.sh 100

# Single model runs
bash run_inference_benchmark.sh 100 tiny_ane
bash run_benchmark.sh 100 stories110m
```

## Files

```
benchmark/
  gpu_train.py                PyTorch MPS training benchmark
  gpu_inference.py            PyTorch MPS inference benchmark
  run_benchmark.sh            Training orchestration (GPU → ANE → compare)
  run_inference_benchmark.sh  Inference orchestration (GPU → ANE → compare)
  run_power_benchmark.sh      Power measurement (requires sudo)
  compare.py                  Training comparison tables
  compare_inference.py        Inference comparison tables
  compare_power.py            Power & energy efficiency tables
  results/                    Output directory (JSON + logs)
```
