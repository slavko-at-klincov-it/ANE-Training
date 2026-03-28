# M4 Mac Mini Benchmark Summary

**Date:** 2026-03-28
**Hardware:** Apple M4 Mac Mini (10-core CPU, 10-core GPU, 16-core ANE, 16GB RAM)
**ANE Architecture:** h16g
**Comparison:** Apple M3 Pro MacBook Pro (11-core CPU, 14-core GPU, 16-core ANE, 18GB RAM, h15g)

---

## 1. Peak ANE Performance

| Metric | M4 (h16g) | M3 Pro (h15g) | Change |
|--------|-----------|---------------|--------|
| Peak TFLOPS (stacked conv) | **13.86** | 12.79 | +8.4% |
| Sustained single-kernel | **5.47** | 5.01 | +9.2% |
| Dispatch overhead | 0.07 ms | — | — |
| Thermal under load | Nominal | Nominal | Same |

## 2. Training Benchmarks (Sequential, Solo Runs)

### Stories-110M (110M params, 12 layers, dim=768)

| Config | ms/step | TFLOPS | 5000-step Loss | Wall Time |
|--------|---------|--------|----------------|-----------|
| **ANE (lr=3e-4)** | 93.1 | 1.40 | 5.99 (x explodes to [-1850,598]) | 634s |
| **ANE (lr=1e-4, maxact=100)** | 93.8 | 1.39 | 4.55 (x stable at [-15,20]) | 641s |
| **ANE Pipeline** | 71.8 | 1.47 (ANE) / 2.43 (total) | — (1000 steps) | 119s |
| **GPU (MPS, PyTorch)** | 1202 | 1.09 | 1.66 | 6011s |

**Key Finding:** ANE is **12.9x faster** per step than GPU on M4 (93ms vs 1202ms), but GPU converges to lower loss (1.66 vs 4.55) due to FP32 vs FP16 precision.

### Tiny-ANE-15M (15M params, 6 layers, dim=256)

| Config | ms/step | 10K-step Loss | Wall Time |
|--------|---------|---------------|-----------|
| **ANE (lr=3e-4)** | 37.7 | 3.25 best, 3.71 final | 408s |

### ANE Training Speed: M4 vs M3 Pro

| Model | M4 ANE ms/step | M3 Pro ANE ms/step | Speedup |
|-------|----------------|--------------------|---------|
| Stories-110M | **93.1** | ~168 (sequential) | **1.8x** |
| Stories-110M Pipeline | **71.8** | 80.9 | **1.13x** |

### Model Sweep (200 steps each, ANE)

| Model | Params | ms/step | Loss@0 | Loss@100 | Loss@200 |
|-------|--------|---------|--------|----------|----------|
| tiny_ane | 13.3M | 26.0 | 9.16 | 5.89 | 6.18 |
| small_ane | 27.6M | 41.0 | 9.11 | 5.89 | 6.15 |
| medium_ane | 43.7M | 49.7 | 9.10 | 5.90 | 6.05 |
| wide_ane | 52.4M | 47.8 | 9.16 | 5.87 | 5.92 |

## 3. Inference Throughput (Solo, No Parallel Load)

### ANE Inference

| Model | ms/token | tokens/sec | TFLOPS |
|-------|----------|------------|--------|
| **Stories-110M** | 21.0 | **47.6** | 2.07 |
| Wide-ANE (52M) | 9.5 | **105.5** | — |
| Tiny-ANE (15M) | 3.9 | **255.0** | 0.65 |

Throughput is constant regardless of generation length (50-1000 tokens tested).

### GPU (MPS) Inference

| Model | ms/token | tokens/sec | TFLOPS |
|-------|----------|------------|--------|
| **Stories-110M** | 30.0 | **33.4** | 1.43 |

### ANE vs GPU Inference

| Model | ANE tok/s | GPU tok/s | ANE Speedup |
|-------|-----------|-----------|-------------|
| Stories-110M | 47.6 | 33.4 | **1.43x** |
| Tiny-ANE | 255.0 | 190.6* | **1.34x** |

*From 100-step benchmark

### M4 vs M3 Pro Inference

| Model | M4 ANE tok/s | M3 Pro ANE tok/s | Change |
|-------|--------------|------------------|--------|
| Stories-110M | **47.6** | 41.3 | +15% |

## 4. CPU vs GPU vs ANE Compute (Matmul Throughput)

| Workload | CPU (GFLOPS) | GPU Batch (GFLOPS) | ANE (GFLOPS) |
|----------|-------------|-------------------|-------------|
| Small (256x256) | **1157** | 211 | 107 |
| Medium (512x512) | **1710** | 915 | 453 |
| Large (768x768) | **1766** | 939 | 853 |
| XL (1024x1024) | 1662 | 1234 | **1314** |
| FFN (768x3072) | 958 | 1748 | **1785** |
| Huge (2048x2048) | 719 | **2363** | 2328 |

**Crossover point:** ANE surpasses CPU at ~1024x1024. ANE matches GPU-batched at FFN sizes and above.

## 5. Hardware Utilization

| Metric | Stories-110M ANE | Stories-110M GPU |
|--------|-----------------|-----------------|
| RAM usage | 2749 MB avg | — |
| Peak RAM | 2764 MB | — |
| Thermal | Nominal (always) | Nominal (always) |
| GPU util during ANE training | 0.8% | — |
| Compile time (one-time) | 433 ms | — |

## 6. Training Stability

| LR | maxact | Activations at step 500 | Loss at step 500 | Stable? |
|----|--------|------------------------|------------------|---------|
| 3e-4 | off | x[-1850, 598] | 5.99 | Diverges eventually |
| 1e-4 | 100 | x[-15, 20] | 4.55 | Stable |

## 7. Key Differences M4 vs M3 Pro

| Aspect | M4 | M3 Pro |
|--------|-----|--------|
| ANE Architecture | h16g | h15g |
| ANE Peak TFLOPS | 13.86 | 12.79 |
| GPU Cores | 10 | 14 |
| RAM | 16 GB | 18 GB |
| ANE Training (Stories-110M) | 93 ms/step | ~168 ms/step |
| ANE Inference (Stories-110M) | 47.6 tok/s | 41.3 tok/s |
| GPU Training (Stories-110M) | 1202 ms/step | 870 ms/step |
| GPU Inference (Stories-110M) | 33.4 tok/s | 47.8 tok/s |
| **ANE vs GPU Training** | **ANE 12.9x faster** | GPU 2.3x faster |
| **ANE vs GPU Inference** | **ANE 1.43x faster** | GPU 1.15x faster |

The M4's ANE (h16g) is substantially faster than M3 Pro's (h15g), while the M4's GPU is weaker (10 vs 14 cores). This flips the ANE vs GPU comparison dramatically in ANE's favor on M4.

---

*All benchmarks run sequentially (no parallel load) unless noted otherwise.*
*Power benchmarks pending (requires sudo).*
