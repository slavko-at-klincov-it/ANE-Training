# ANE Kernel Shape Sweep — Empirical Results

**Date:** 2026-03-20
**Chip:** M3 Pro (h15g), 16 cores
**Thermal:** Nominal throughout all tests
**Tools:** `examples/bench`, `training/sweep_training`, `training/sweep_stacked`

## Executive Summary

- **Peak single-kernel throughput: 7.61 TFLOPS** (768x2048 sp4096, FFN-up shape)
- **Peak stacked throughput: 11.65 TFLOPS** (384ch sp128 x256 depth)
- **Zero SRAM spills** across all 204 configurations tested (including 44MB activations)
- **Spatial dimension is the dominant throughput lever** — 10-100x impact on TFLOPS
- **768ch stacked convs are slow** (~5-7 TFLOPS) — 384-640ch sweet spot for stacking
- **Dispatch overhead ~0.3ms/kernel** — amortize by larger spatial or stacking

## Stories-110M Layer Throughput Map

These are the exact shapes used in the transformer (QKV projection 768->768, FFN up 768->2048, FFN down 2048->768):

| Spatial | QKV 768x768 | FFN-up 768x2048 | FFN-down 2048x768 |
|---------|-------------|------------------|---------------------|
| 1       | 0.02 TFLOPS | 0.10 TFLOPS      | 0.09 TFLOPS         |
| 64      | 0.29 TFLOPS | 0.59 TFLOPS      | 0.47 TFLOPS         |
| 128     | 0.46 TFLOPS | 1.36 TFLOPS      | 1.61 TFLOPS         |
| 256     | 0.87 TFLOPS | 3.52 TFLOPS      | 2.07 TFLOPS         |
| 512     | 1.92 TFLOPS | 4.27 TFLOPS      | 3.00 TFLOPS         |
| 1024    | 3.80 TFLOPS | 5.15 TFLOPS      | 4.34 TFLOPS         |
| 2048    | 5.04 TFLOPS | 7.07 TFLOPS      | 5.75 TFLOPS         |
| 4096    | 6.28 TFLOPS | 7.61 TFLOPS      | 5.92 TFLOPS         |

**Key observations:**
- FFN-up (768->2048) consistently fastest — larger output channels amortize dispatch better
- FFN-down (2048->768) is ~20-25% slower than FFN-up despite same FLOP count — wider input with narrower output has worse memory access pattern
- QKV (768->768) saturates around 6.3 TFLOPS — smaller total FLOPs per eval limits throughput
- Going from sp256 to sp2048 doubles throughput for all layer types
- sp1 is catastrophically slow (dispatch overhead dominates)

## Stacked Conv Results (Top 10)

Stacking multiple convs in one compiled MIL program amortizes dispatch overhead:

| Rank | Ch  | Sp  | Depth | GFLOP | Latency   | TFLOPS |
|------|-----|-----|-------|-------|-----------|--------|
| 1    | 384 | 128 | 256   | 9.66  | 0.830 ms  | 11.65  |
| 2    | 512 | 64  | 256   | 8.59  | 0.777 ms  | 11.05  |
| 3    | 512 | 128 | 128   | 8.59  | 0.782 ms  | 10.99  |
| 4    | 640 | 128 | 64    | 6.71  | 0.645 ms  | 10.40  |
| 5    | 640 | 64  | 128   | 6.71  | 0.646 ms  | 10.39  |
| 6    | 640 | 64  | 256   | 13.42 | 1.304 ms  | 10.29  |
| 7    | 640 | 128 | 128   | 13.42 | 1.407 ms  | 9.54   |
| 8    | 384 | 64  | 256   | 4.83  | 0.517 ms  | 9.35   |
| 9    | 384 | 128 | 128   | 4.83  | 0.532 ms  | 9.09   |
| 10   | 512 | 128 | 256   | 17.18 | 1.909 ms  | 9.00   |

**Critical finding — 768ch stacked convs are slower than expected:**

| Ch   | Sp  | Depth | TFLOPS | Why slower?                          |
|------|-----|-------|--------|--------------------------------------|
| 768  | 32  | 128   | 1.45   | Weight matrix too large for registers |
| 768  | 64  | 64    | 3.59   | Only 4.83 GFLOP to amortize          |
| 768  | 128 | 64    | 7.23   | Decent but 384ch is 60% faster       |
| 768  | 128 | 128   | 5.10   | Latency jumps to 3.8ms               |
| 768  | 128 | 256   | 4.87   | Latency 7.9ms — heavy weight traffic |
| 1024 | 128 | 256   | 4.54   | Even worse — 15ms latency            |

The 768ch and 1024ch stacked kernels suffer a sharp performance cliff. The weight matrix (ch*ch*2 bytes per layer) is 1.1MB for 768ch and 2.0MB for 1024ch vs only 0.3MB for 384ch. At depth 256, that's 288MB of weights for 768ch vs 75MB for 384ch. The ANE's weight fetch pipeline saturates.

## SRAM Analysis

**Zero SRAM spills detected** across all 204 configurations tested, including:
- 768x2048 sp4096 with 44MB of activations (I/O)
- 2048x2048 sp512 with 8MB of activations
- 4096x4096 sp64 with 32MB weights

The libane `ane_sram_spill()` check reported no spills for any configuration. The SRAM warning printed for the sp4096 configs (44MB I/O) is a software heuristic warning, not an actual hardware spill. The ANE compiler appears to handle streaming/tiling internally — activations don't need to fit in SRAM all at once.

**Implication:** The 32MB SRAM limit likely applies to intermediate activations within a single operation, not to total I/O. For 1x1 convolutions, the computation is essentially a matrix multiply that can be tiled, so SRAM only needs to hold partial accumulations.

## Dispatch Overhead Analysis

Minimum latencies observed across all configs:

| Config           | Latency   | Notes                     |
|------------------|-----------|---------------------------|
| 768x768 sp1      | 0.060 ms  | Near-zero compute         |
| 768x2048 sp1     | 0.031 ms  | Faster — already set up   |
| 2048x768 sp1     | 0.033 ms  | Similar                   |
| 64x64 sp256      | 0.305 ms  | Small compute, full setup |
| 128x128 sp2048   | 0.216 ms  | Best small-channel        |

**Dispatch floor: ~0.03-0.06ms** for cached/warmed kernels, ~0.2-0.3ms amortized including setup.

For training with 12 layers (Stories-110M), each forward pass needs ~36 kernel dispatches (3 per layer: QKV, FFN-up, FFN-down). At 0.3ms overhead, that's ~10.8ms of pure dispatch overhead. This is why stacking (fewer dispatches, more work per dispatch) or large spatial (more FLOPs per dispatch) are essential.

## Optimal Shapes for Training

### For Stories-110M (dim=768, hidden=2048)

**Best single-kernel configuration:** Use sp256-sp512 as the practical training spatial dimension.

| Layer     | Shape      | sp256 TFLOPS | sp256 Latency | Recommendation |
|-----------|------------|-------------|---------------|----------------|
| QKV proj  | 768x768    | 0.87        | 0.347 ms      | Batch to sp512+ |
| FFN up    | 768x2048   | 3.52        | 0.229 ms      | Good at sp256  |
| FFN down  | 2048x768   | 2.07        | 0.390 ms      | Batch to sp512+ |
| Output    | 768x4096   | 5.94        | 0.271 ms      | Excellent       |

**Practical recommendations:**
1. **Increase sequence batch** (spatial dim) to at least 256, ideally 512-1024
2. **FFN-up is the fastest layer** — forward pass will be FFN-down bottlenecked
3. **768x4096 sp256 hits 5.94 TFLOPS** — vocab projection is surprisingly efficient
4. **Asymmetry matters:** 768->2048 is 70% faster than 2048->768 at sp256

### For Architecture Design (if flexible)

If designing a new model for ANE training:
- **Use dim=384 or dim=512** instead of 768 — stacked convs are 2x faster
- **Prefer wider models (more hidden dim) over deeper** — 384ch x256 depth = 11.65 TFLOPS
- **Spatial dim >= 128** is essential for efficiency
- **Sweet spot: 384-640 channels, 64-128 spatial, 128-256 depth stacked**

## Single Kernel Throughput Scaling

Throughput vs. GFLOP per evaluation (sorted):

| GFLOP | Config            | TFLOPS | Efficiency |
|-------|-------------------|--------|------------|
| 0.001 | QKV sp1           | 0.02   | 0.3%       |
| 0.034 | 64x64 sp4096      | 0.10   | 1.3%       |
| 0.075 | QKV sp64          | 0.29   | 3.8%       |
| 0.134 | 512x512 sp256     | 0.58   | 7.6%       |
| 0.302 | QKV sp256         | 0.87   | 11.4%      |
| 0.604 | QKV sp512         | 1.92   | 25.2%      |
| 0.805 | FFN-up sp256      | 3.52   | 46.2%      |
| 1.074 | 2048x2048 sp128   | 3.83   | 50.3%      |
| 1.611 | FFN-up sp512      | 4.27   | 56.1%      |
| 2.416 | QKV sp2048        | 5.04   | 66.2%      |
| 4.295 | 2048x2048 sp512   | 4.85   | 63.7%      |
| 6.442 | FFN-up sp2048     | 7.07   | 92.9%      |
| 12.885| FFN-up sp4096     | 7.61   | 100.0%     |

**Empirical rule:** You need ~1 GFLOP per kernel eval to exceed 50% efficiency. Below 0.3 GFLOP, dispatch overhead dominates and throughput collapses.

## Key Takeaways

1. **Spatial is everything.** Going from sp64 to sp2048 improves FFN-up from 0.59 to 7.07 TFLOPS (12x).
2. **768ch is a bad stacking dimension.** Use 384-512ch for stacked benchmarks. For actual training, use single-kernel with large spatial instead.
3. **No SRAM spills observed** — the 32MB limit doesn't apply to 1x1 conv I/O in practice.
4. **FFN-up > FFN-down** at same FLOP count — output-wider shapes are faster.
5. **Dispatch overhead is ~0.3ms** — plan at least 1 GFLOP per dispatch for >50% efficiency.
6. **Thermal: always Nominal** — ANE never throttled across 200+ configs and sustained benchmarks.
