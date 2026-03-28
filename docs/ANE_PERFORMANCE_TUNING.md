# ANE Performance Tuning Guide

Comprehensive results from performance sweeps on Apple Neural Engine.
Sweep data from M3 Pro (h15g). M4 (h16g) peak results added for comparison.
All measurements use 1x1 convolution kernels compiled and executed via the private ANE framework.

---

## Peak vs Real Training Performance

> **All numbers in this document are ANE silicon peak benchmarks, not training throughput.**
>
> | Metric | M3 Pro (h15g) | M4 (h16g) | What it measures |
> |:---|---:|---:|:---|
> | ANE Silicon Peak | 12.79 | **13.86** | Stacked conv, single dispatch, all overhead amortized |
> | Best Single Kernel | 11.64 | — | 512x4096 sp4096, one large kernel |
> | Sustained Single-Kernel | 5.01 | **5.47** | Continuous eval of one kernel for 5 seconds |
> | **Real Training (pipeline)** | **2.15** | **2.43** | **Stories-110M, full training loop** |
> | **Real Training (sequential)** | **1.87** | **1.40** | **Stories-110M, full training loop** |
>
> The sweep results below help find the optimal kernel shapes for maximum ANE throughput.
> However, real training throughput is ~6x lower than peak because each training step involves
> per-layer dispatch overhead, IOSurface I/O (FP32/FP16 conversion), and CPU gradient computation.
> See [ANE_MODEL_SIZE_BENCHMARK.md](ANE_MODEL_SIZE_BENCHMARK.md) for real training numbers.

---

## Table of Contents

1. [Stacked Sweep (90 Combinations)](#1-stacked-sweep-90-combinations)
2. [Single Kernel Sweep (~250 Combinations)](#2-single-kernel-sweep-250-combinations)
3. [Eval Method Sweep](#3-eval-method-sweep)
4. [Practical Rules](#4-practical-rules)
5. [Before/After Comparison](#5-beforeafter-comparison)

---

## 1. Stacked Sweep (90 Combinations)

Stacked sweeps chain multiple identical convolution layers back-to-back in a single
compiled program, amortizing dispatch overhead and keeping the ANE pipeline saturated.

**Parameters swept:**
- Channels: 128, 256, 384, 512, 640, 768, 896, 1024
- Spatial sizes: 64, 128, 256
- Stack depths: 32, 64, 128, 256, 512

### Peak Result

**12.79 TFLOPS** at 512 channels, spatial 128, depth 128.

### Top 10 Configurations

| Rank | Channels | Spatial | Depth | TFLOPS |
|------|----------|---------|-------|--------|
| 1    | 512      | 128     | 128   | 12.79  |
| 2    | 512      | 128     | 256   | 12.65  |
| 3    | 384      | 128     | 256   | 12.51  |
| 4    | 640      | 128     | 128   | 12.44  |
| 5    | 384      | 128     | 128   | 12.38  |
| 6    | 640      | 128     | 256   | 12.31  |
| 7    | 512      | 256     | 128   | 12.18  |
| 8    | 512      | 64      | 256   | 12.05  |
| 9    | 384      | 256     | 128   | 11.92  |
| 10   | 640      | 256     | 128   | 11.87  |

### Key Findings

- **Channel sweet spot: 384-640.** All top configurations fall within this range.
  The ANE MAC array appears optimally utilized at these widths.

- **768+ channels are catastrophically slow.** Throughput drops 3-7x compared to
  the 384-640 range. At 1024 channels, performance can fall below 2 TFLOPS. This
  strongly suggests the ANE must split wide channel operations across multiple tiles,
  introducing pipeline stalls and SRAM thrashing.

- **depth=512 crashes the compiler service.** The `ANECompilerService` process is
  killed (likely by the system watchdog) when attempting to compile programs with
  512 stacked layers. The practical maximum is depth=256, with depth=128 being
  the optimal trade-off between throughput and compilation time.

- **Diminishing returns past depth=128 for large spatial.** At spatial 256, going
  from depth=128 to depth=256 gains less than 1%. The dispatch overhead is already
  fully amortized at 128 layers for large inputs.

---

## 2. Single Kernel Sweep (~250 Combinations)

Single-kernel sweeps test one convolution per dispatch to isolate raw compute
throughput from stacking effects.

**Parameters swept:**
- ch_in: 64, 128, 256, 512, 1024, 2048
- ch_out: 64, 128, 256, 512, 1024, 2048, 4096
- Spatial sizes: 256, 512, 1024, 2048, 4096

### Peak Result

**11.64 TFLOPS** at ch_in=512, ch_out=4096, spatial 4096.

### Top 10 Configurations

| Rank | ch_in | ch_out | Spatial | TFLOPS |
|------|-------|--------|---------|--------|
| 1    | 512   | 4096   | 4096    | 11.64  |
| 2    | 512   | 2048   | 4096    | 11.41  |
| 3    | 256   | 4096   | 4096    | 11.28  |
| 4    | 512   | 4096   | 2048    | 11.15  |
| 5    | 512   | 1024   | 4096    | 10.97  |
| 6    | 1024  | 2048   | 4096    | 10.82  |
| 7    | 256   | 2048   | 4096    | 10.71  |
| 8    | 512   | 2048   | 2048    | 10.63  |
| 9    | 512   | 4096   | 1024    | 10.49  |
| 10   | 1024  | 4096   | 4096    | 10.35  |

### Key Findings

- **Spatial size is the dominant factor.** Going from spatial 256 to spatial 4096
  yields approximately 3x throughput improvement. Large spatial dimensions keep
  the ANE's data path fully utilized and reduce the relative cost of kernel launch.

- **ch_in=512 is the sweet spot.** Mirrors the stacked sweep finding. The ANE's
  internal MAC array width appears to be optimized for 512-channel inputs.

- **Asymmetric shapes (large ch_out) outperform square.** A 512x4096 configuration
  beats 2048x2048 despite identical FLOP counts. The ANE handles fan-out (many
  output channels) more efficiently than large square weight matrices, likely because
  output channels can be computed independently across tiles.

- **Zero SRAM spills across all configurations.** Even at the largest tested sizes
  (2048x4096 at spatial 4096), the compiler reports zero SRAM spill bytes. The ANE
  compiler's tiling strategy is highly effective for 1x1 convolutions, keeping all
  intermediate data on-chip.

---

## 3. Eval Method Sweep

This sweep tested different ways of dispatching work to the ANE: QoS levels,
IO fence settings, evaluation methods, thread priorities, and dispatch patterns.

**Baseline configuration:** 768x768, spatial 256, depth 64.

### Results Summary

| Configuration                          | TFLOPS | vs. Default |
|----------------------------------------|--------|-------------|
| QoS=9 + DisableIOFences               | 3.307  | best        |
| QoS=9 (standard)                      | 3.298  | -0.3%       |
| evaluateRealTimeWithModel              | 3.297  | -0.3%       |
| Default (QoS=0)                        | 3.112  | baseline    |
| Thread QOS_CLASS_BACKGROUND            | 1.503  | **-2.2x**   |

### Key Findings

- **QoS=9 + DisableIOFences is optimal.** This combination yields the highest
  throughput. QoS=9 gives the ANE maximum scheduling priority. DisableIOFences
  removes synchronization barriers between operations, allowing the ANE to pipeline
  more aggressively.

- **`evaluateRealTimeWithModel` is identical to standard evaluation.** Despite the
  name suggesting a different code path, there is no measurable difference. The API
  appears to be a thin wrapper around the same dispatch mechanism.

- **Thread QOS_CLASS_BACKGROUND makes the ANE 2.2x SLOWER.** This is a critical
  pitfall. Setting the calling thread to background priority causes the system to
  also deprioritize the ANE hardware work, even though the computation runs on a
  dedicated accelerator.

- **Tight dispatch loop is best.** Any artificial delay between dispatches (sleep,
  yielding) hurts throughput. The ANE has internal queuing and should be kept
  continuously fed.

### ANE QoS vs. Thread QoS: A Critical Distinction

These are two completely independent priority systems that happen to share a name:

| Property            | ANE QoS (programQoS)               | Thread QoS (QOS_CLASS)                  |
|---------------------|-------------------------------------|-----------------------------------------|
| What it controls    | ANE hardware scheduling priority    | CPU thread scheduling priority          |
| Set via             | `ANECCompilationOptions` key        | `pthread_set_qos_class_self()`          |
| Range               | 0-9 (9 = highest)                   | Enum: BACKGROUND → USER_INTERACTIVE     |
| Effect on ANE       | Direct: higher = faster dispatch    | Indirect: low thread QoS deprioritizes ANE work |
| Optimal setting     | **9** (maximum)                     | **USER_INITIATED or higher**            |

**The mistake to avoid:** Setting ANE QoS=9 (good) but running on a
QOS_CLASS_BACKGROUND thread (bad). The thread QoS overrides the ANE QoS from
the system scheduler's perspective, negating the benefit. Always ensure the
calling thread is at least USER_INITIATED priority.

---

## 4. Practical Rules

### Channel Sizing
- **Use 384-640 channels.** 512 is the single best choice.
- **Never exceed 768.** Performance falls off a cliff due to tile splitting.
- If your model needs more than 640 channels, restructure as two parallel
  smaller convolutions rather than one wide one.

### Spatial Dimensions
- **As large as possible.** Spatial 4096 is 3x faster than spatial 256
  for the same channel configuration.
- Large spatial dimensions amortize dispatch overhead and keep the data
  pipeline saturated.

### Stacking / Depth
- **128 layers is optimal.** This fully amortizes dispatch overhead without
  hitting compiler limits.
- depth=256 works but offers negligible gains over 128.
- **Never use depth=512.** The compiler service will crash.

### Dispatch Configuration
- **Always set QoS=9** in compilation options.
- **Always set DisableIOFences=true** for maximum pipeline throughput.
- **Never set thread to QOS_CLASS_BACKGROUND.** Use USER_INITIATED or higher.
- **Dispatch in tight loops.** Do not sleep or yield between submissions.

### Weight Matrix Shape
- Prefer asymmetric shapes (small ch_in, large ch_out) over square.
- 512×4096 outperforms 2048×2048 at identical FLOP counts.

---

## 5. Before/After Comparison

Peak benchmark improvements achieved through systematic tuning (these are ANE silicon peak numbers, not training throughput):

| Metric               | Before   | After    | Improvement |
|----------------------|----------|----------|-------------|
| Peak stacked (benchmark) | 9.90 TFLOPS | 12.79 TFLOPS | **+29%**    |
| Peak single kernel (benchmark) | 4.73 TFLOPS | 11.64 TFLOPS | **+146%**   |
| Real training (sequential) | — | 1.87 TFLOPS | 93 ms/step |
| Real training (pipeline) | — | 2.15 TFLOPS | 80.9 ms/step |

The stacked improvement (+29%) comes from finding the optimal channel/depth
configuration. The single-kernel improvement (+146%) is more dramatic because
the initial measurements used suboptimal spatial sizes and symmetric channel
configurations.

### What Changed

| Factor                | Before                  | After                       |
|-----------------------|-------------------------|-----------------------------|
| Channel width         | 1024 (too wide)         | 512 (sweet spot)            |
| Spatial size          | 256                     | 4096                        |
| Stack depth           | 64                      | 128                         |
| ANE QoS              | Default (0)             | 9                           |
| IO Fences            | Enabled                 | Disabled                    |
| Thread QoS           | Uncontrolled            | USER_INITIATED minimum      |

---

## Hardware Context

Sweep measurements taken on:
- **Apple M3 Pro** (h15g ANE identity)
- **macOS** with private ANE framework access
- **1×1 convolution** kernels (compute-bound, no spatial reduction)
- Throughput calculated as `2 × ch_in × ch_out × spatial² × depth / time`

M4 peak numbers from `examples/bench.c` auto-benchmark. Full M4 results: [ANE_M4_BENCHMARK.md](ANE_M4_BENCHMARK.md)
