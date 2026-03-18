<div align="center">

<pre>
 ___    _   __ ______
/   |  / | / // ____/
/ /| | /  |/ // __/
/ ___ |/ /|  // /___
/_/  |_/_/ |_//_____/  Training
</pre>

### Reverse-Engineering Apple's Neural Engine

**1 Compile · Auto Peak Detection · 35 Private Classes · Zero Recompilation**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/macOS_15+-111111.svg?logo=apple&logoColor=white)](https://www.apple.com/macos/)
[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-M1--M5-FF3B30.svg)](https://support.apple.com/en-us/116943)
[![Compile](https://img.shields.io/badge/Compile-1x_for_∞_Steps-34C759.svg)](#dynamic-spatial-packing--the-breakthrough)
[![TFLOPS](https://img.shields.io/badge/Peak-Auto_Detect-007AFF.svg)](#benchmark-results-m3-pro)
[![Classes](https://img.shields.io/badge/ANE_Classes-35-FF9500.svg)](RESEARCH_ANE_COMPLETE.md)
[![Deutsch](https://img.shields.io/badge/Deutsch-README__DE.md-lightgrey.svg)](README_DE.md)

</div>

<br>

> The first standalone C API (`libane`) for Apple's private Neural Engine. Enables **training** directly on the ANE — Apple officially restricts it to inference via CoreML. Complete hardware research, runnable demos, benchmark suite.

---

<table>
<tr>
<td width="50%">

**Discoveries**

| | |
|:--|:--|
| `35` | Private API classes discovered (previously known: only 4) |
| `42%` | Faster with QoS Background instead of Default |
| `auto` | Peak TFLOPS detected on startup (~1s) |
| `1` | Compile is enough — unlimited training steps |
| `3x` | Conv 1x1 faster than matmul on ANE |

</td>
<td width="50%">

**libane — 73KB C API**

| | |
|:--|:--|
| Dynamic | Weights via IOSurface, zero recompile |
| Zero-Copy | I/O directly into ANE memory |
| Version-Detection | Survives Apple API changes |
| 6 QoS Levels | Background (9) to Realtime (0) |
| FP16 Safe | Overflow protection built in |

</td>
</tr>
</table>

> [!NOTE]
> **Compilation Pipeline:** `MIL → MLIR → LLIR → HWX` — see [Glossary](#glossary) for all technical terms.

---

## Prerequisites

| What | Minimum | Tested with |
|:---|:---|:---|
| Mac | Apple Silicon (M1–M5) | M3 Pro |
| macOS | 15+ | 26.3.1 (Build 25D2128) |
| Xcode CLI Tools | Required | `xcode-select --install` |

```bash
uname -m           # → "arm64"
xcode-select -p    # → /Library/Developer/CommandLineTools
```

> [!WARNING]
> **Intel Mac?** This project works **only** on Apple Silicon. The Neural Engine exists only in M-series chips.

---

## Installation

<table>
<tr>
<td width="50%">

**Option A — One-Liner**

```bash
curl -sSL https://raw.githubusercontent.com/\
slavko-at-klincov-it/ANE-Training/\
main/install.sh | bash
```

Checks prerequisites, clones, builds, benchmarks.

</td>
<td width="50%">

**Option B — Manual**

```bash
git clone https://github.com/slavko-at-klincov-it/ANE-Training.git
cd ANE-Training
./ane
```

Interactive menu — builds everything automatically.

</td>
</tr>
</table>

> [!TIP]
> `./ane` detects your hardware, **measures ANE peak TFLOPS automatically** (~1s), builds all binaries on first run, and guides you through everything. You don't need to know any Makefiles or paths.

---

## Quick Start

```bash
./ane                # Interactive menu (auto-detects ANE peak TFLOPS)
./ane train          # Training demo (Y=2X, 1 compile, 60 steps)
./ane bench          # Full benchmark (sweep + sustained peak + chip comparison)
./ane generate       # Shakespeare text generation on ANE
./ane explore        # Explore 35 ANE classes interactively
./ane info           # Hardware detection
./ane test           # libane test suite
```

<details open>
<summary><b>Training Demo</b> — <code>make demo</code></summary>

&nbsp;

Trains a linear layer directly on the ANE with **Dynamic Spatial Packing** — compiles once, trains 60 steps without recompilation. Forward on Neural Engine, backward + SGD on CPU.

```
Hardware: h15g, 16 ANE cores
Compiled once (dynamic weights, no recompilation needed)

Goal: Train W so that Y = W @ X approximates Y = 2*X

step   loss       W[0,0]   W[1,1]   ms/step
0        1.3493    0.148    0.241   0.4
5        0.5334    0.847    0.868   0.1
10       0.2304    1.260    1.254   0.1
30       0.0164    1.841    1.825   0.1
59       0.0010    1.975    1.967   0.3

Diagonal average: 1.959 (converged!)
Compile count: 1 / 119 budget
```

**1 compilation instead of 60.** Weights are updated via IOSurface write, not via recompile. 0.1ms/step.

</details>

<details>
<summary><b>Auto-Benchmark</b> — <code>./ane bench</code></summary>

&nbsp;

Detects your chip, sweeps kernel shapes, finds peak TFLOPS, compares to Apple spec:

```
Chip:   h15g (M3 Pro), 16 cores
Apple:  18 TOPS (marketing, INT8)

---- Single Conv Sweep ----
768x2048 sp256     3.0 MB   0.81  0.221 ms    3.64
2048x2048 sp128    8.0 MB   1.07  0.346 ms    3.10

---- Sustained Peak (5 seconds) ----
Kernel:    768x2048 sp256 (best from sweep)
Evals:     18400 in 5.0s
Sustained: 2.95 TFLOPS (fp16)

---- Chip Comparison (measured fp16 TFLOPS) ----
>> h15g (M3 Pro)        5.46 TFLOPS  ███████████░░░░░░░░░░░░░░░░░░░
   h16g (M4)           11.00 TFLOPS  ███████████████████████░░░░░░░

---- Summary ----
Measured peak:       5.46 TFLOPS (fp16 matmul)
Apple marketing:     18 TOPS (INT8, theoretical)
Efficiency:          30.3% of Apple spec
```

On startup, `./ane` runs a quick peak measurement (~1s) and shows:
```
  + Hardware: h15g (M3 Pro), 16 cores
  + ANE Peak: 3.6 TFLOPS (18 TOPS Apple spec, 20%)
  + API: v1(35)
```

</details>

<details>
<summary><b>Text Generation</b> — <code>make generate</code></summary>

&nbsp;

Bigram model on Shakespeare, typewriter output. Compiles once, trains + generates without recompile:

```
Compiled once (dynamic weights, no recompilation needed)

Training bigram model on Shakespeare...
step   loss      perplexity
0       4.1589   64.00
29      3.1245   22.76

Generating text (200 chars, temperature=0.8)...
To be or not to be, that is the question...

Compiles used: 1 / 119
```

</details>

<details>
<summary><b>ANE Explorer</b> — <code>make explore</code></summary>

&nbsp;

All 35 ANE classes categorized, interactive inspection:

```
Found 35 ANE classes

┌─ Core (Model compilation, loading, evaluation)
│  █ _ANEInMemoryModel
│  █ _ANEInMemoryModelDescriptor
│  █ _ANERequest
└─

Interactive Mode: Enter a class name to inspect
> _ANEInMemoryModel
Instance Methods (23):
  - compileWithQoS:options:error:
  - loadWithQoS:options:error:
  ...
```

</details>

---

## Dynamic Spatial Packing — The Breakthrough

> [!IMPORTANT]
> **The Problem:** The ANE bakes weights into the HWX binary at compilation time. Every weight update requires a recompilation (~520ms). Hard limit: **~119 compilations per process**, then silent failures.
>
> **The Solution:** Instead of baking weights into the binary, pack them as **input channels** alongside the activations. MIL code slices them apart → matmul. **Compile once, train unlimited.**

<table>
<tr>
<td width="50%">

**Before — Standard**
```
Weights ──→ BLOBFILE ──→ Compile
Input   ──→ IOSurface ──→ Eval

Per Step:  1 Compile + 1 Eval
Budget:    119 Steps max
Latency:   0.3–0.5 ms/step
```

</td>
<td width="50%">

**After — Dynamic Spatial Packing**
```
Weights ──→ IOSurface ──→ Write
Input   ──→ IOSurface ──→ Eval

Per Step:  1 Write + 1 Eval
Budget:    ∞ Steps
Latency:   0.1 ms/step
```

</td>
</tr>
</table>

<details>
<summary><b>How does this work in detail?</b></summary>

&nbsp;

**Input Layout:** `[1, in_ch + in_ch*out_ch, 1, seq]`
- Channels `[0..in_ch)`: Activations (training data)
- Channels `[in_ch..in_ch+in_ch*out_ch)`: Weight matrix (flattened), spatial position 0 only

**MIL Program (generated by `ane_mil_linear_dynamic()`):**
```
1. Cast FP32 → FP16
2. slice_by_size → Activations [1, in_ch, 1, seq]
3. slice_by_size → Weights [1, in_ch*out_ch, 1, 1]
4. reshape → Weights [1, 1, out_ch, in_ch]
5. reshape + transpose → Activations [1, 1, seq, in_ch]
6. matmul(activations, weights^T) → [1, 1, seq, out_ch]
7. transpose + reshape → [1, out_ch, 1, seq]
8. Cast FP16 → FP32
```

**New libane API:**
```c
// Generate MIL with weights-as-input
char *mil = ane_mil_linear_dynamic(in_ch, out_ch, seq);

// Compile ONCE — no weights needed
ANEKernel *k = ane_compile(mil, strlen(mil), NULL, 0,
                           1, &in_bytes, 1, &out_bytes, ANE_QOS_BACKGROUND);

// Training loop: only update IOSurface, never recompile
for (int step = 0; step < 10000; step++) {
    ane_write_dynamic_weights(k, 0, W, in_ch, out_ch, seq);
    ane_lock_input(k, 0);
    float *ptr = (float *)ane_input_ptr(k, 0);
    // Write activations into channels 0..in_ch
    ane_unlock_input(k, 0);
    ane_eval(k, ANE_QOS_BACKGROUND);
    // ... backward + SGD ...
}
```

</details>

<details>
<summary><b>RE Results: What did NOT work</b></summary>

&nbsp;

Before Dynamic Spatial Packing, we tested 5 other approaches:

| Approach | Result | Reason |
|:---|:---|:---|
| Disk patch + unload/reload | Weights not updated | ANE bakes weights into HWX at compile time |
| In-memory VM patch | No effect | ANE uses SRAM copy, ignores RAM patches |
| `_ANEWeight` class | Request created, weights not applied | Unknown binding |
| `weightsBuffer` in `_ANERequest` | Accepted, no effect | ANE ignores runtime weights |
| `ane_reload_weights()` | MIL gets deleted | Unload deletes tmpDir |

**Only Dynamic Spatial Packing works** — weights as input data, not as compile-time constants.

</details>

---

## Write Your Own Code with libane

<details>
<summary><b>Static Weights (Inference)</b> — <code>my_test.c</code></summary>

&nbsp;

```c
#include <stdio.h>
#include <string.h>
#include "ane.h"

int main() {
    ane_init();
    ANEDeviceInfo info = ane_device_info();
    printf("ANE: %s, %d cores\n", info.arch, info.num_cores);

    // Prepare weights (2x2 identity matrix)
    float weights[] = {1.0f, 0.0f, 0.0f, 1.0f};
    ANEWeight w = ane_weight_fp16("@model_path/weights/w.bin", weights, 2, 2);

    // Generate MIL + compile
    char *mil = ane_mil_linear(2, 2, 1, "@model_path/weights/w.bin");
    size_t in_sz = 2 * sizeof(float), out_sz = 2 * sizeof(float);
    ANEKernel *k = ane_compile(mil, strlen(mil), &w, 1,
                               1, &in_sz, 1, &out_sz, ANE_QOS_BACKGROUND);

    // Execute
    float input[] = {3.0f, 7.0f}, output[2];
    ane_write(k, 0, input, in_sz);
    ane_eval(k, ANE_QOS_BACKGROUND);
    ane_read(k, 0, output, out_sz);

    printf("Input:  [%.1f, %.1f]\n", input[0], input[1]);
    printf("Output: [%.1f, %.1f]\n", output[0], output[1]);

    ane_free(k); free(mil); ane_weight_free(&w);
}
```

```bash
xcrun clang -O2 -fobjc-arc -I libane -o my_test my_test.c libane/ane.m \
    -framework Foundation -framework IOSurface -ldl
```

</details>

<details>
<summary><b>Dynamic Weights (Training)</b> — Compile-Once Pattern</summary>

&nbsp;

```c
#include <string.h>
#include "ane.h"

int in_ch = 8, out_ch = 8, seq = 64;

// 1. Generate MIL with dynamic weights
char *mil = ane_mil_linear_dynamic(in_ch, out_ch, seq);

// 2. Compile ONCE — no weights needed
size_t in_sz = (in_ch + in_ch * out_ch) * seq * sizeof(float);
size_t out_sz = out_ch * seq * sizeof(float);
ANEKernel *k = ane_compile(mil, strlen(mil), NULL, 0,
                           1, &in_sz, 1, &out_sz, ANE_QOS_BACKGROUND);

// 3. Training loop — weights via IOSurface write, never recompile
float W[8 * 8];
for (int step = 0; step < 10000; step++) {
    // Pack weights + activations
    ane_write_dynamic_weights(k, 0, W, in_ch, out_ch, seq);
    ane_lock_input(k, 0);
    float *ptr = (float *)ane_input_ptr(k, 0);
    // ptr[0..in_ch*seq] = activations
    ane_unlock_input(k, 0);

    ane_eval(k, ANE_QOS_BACKGROUND);
    // ... backward + SGD on W ...
}

ane_free(k); free(mil);
```

</details>

Full API documentation: **[libane/README.md](libane/README.md)**

---

## Benchmark Results (M3 Pro)

| Metric | Value | |
|:---|---:|:---|
| Measured Peak FP16 | **5.46 TFLOPS** | 128x stacked conv (amortized dispatch) |
| Best Single Kernel | **3.64 TFLOPS** | 768×2048 sp256 |
| Sustained Peak (5s) | **2.95 TFLOPS** | Continuous eval, no interruption |
| Apple Marketing Spec | **18 TOPS** | INT8, theoretical |
| Efficiency vs Spec | **30.3%** | Real fp16 matmul vs Apple INT8 TOPS |
| Training Stories110M | **91–183 ms/step** | Depending on vocab size |
| Dispatch Overhead | **~0.25 ms** | Minimum latency per kernel call |
| QoS Background vs Default | **42% faster** | Less scheduling overhead |

> [!NOTE]
> Apple's "18 TOPS" is an INT8 peak theoretical number. Real FP16 matmul throughput is lower due to dispatch overhead, memory bandwidth, and the fp16 data path being narrower than INT8. **Run `./ane bench` to measure your chip's actual peak.**

<details>
<summary><i>TFLOPS vs TOPS — what's the difference?</i></summary>

&nbsp;

**TFLOPS** = Tera Floating-Point Operations Per Second — counts multiply+add as 2 ops.<br>
**TOPS** = Tera Operations Per Second — counts each operation individually.<br>
That's why TOPS can be higher than TFLOPS on the same hardware. Apple publishes TOPS (INT8), we measure TFLOPS (FP16).

</details>

---

## Chip Comparison — All Apple Silicon Generations

> [!NOTE]
> **The ANE is identical across base/Pro/Max.** Within a generation: same 16-core Neural Engine. Pro/Max only add more GPU + bandwidth. **Only Ultra doubles the ANE** (32 cores). Memory bandwidth does **not** help the ANE as long as tensors stay within the ~32MB SRAM.

| Chip | Arch | ANE TOPS | Mem BW | TFLOPS\* | INT8 | SRAM |
|:---|:---|---:|---:|---:|:---|---:|
| **M1** | H13 | 11 | 68 GB/s | ~5.5 | weights only | ~32 MB |
| M1 Pro | H13 | 11 | 200 GB/s | ~5.5 | weights only | ~32 MB |
| M1 Max | H13 | 11 | 400 GB/s | ~5.5 | weights only | ~32 MB |
| M1 Ultra | H13 x2 | **22** | 800 GB/s | ~11 | weights only | ~64 MB |
| | | | | | | |
| **M2** | H14 | 15.8 | 100 GB/s | ~8 | weights only | ~32 MB |
| M2 Pro | H14 | 15.8 | 200 GB/s | ~9.0 | weights only | ~32 MB |
| M2 Max | H14 | 15.8 | 400 GB/s | ~9.2 | weights only | ~32 MB |
| M2 Ultra | H14 x2 | **31.6** | 800 GB/s | ~18 | weights only | ~64 MB |
| | | | | | | |
| **M3** | H15 | 18 | 100 GB/s | ~9.4 | weights only | ~32 MB |
| **M3 Pro** | H15 (h15g) | 18 | 150 GB/s | **9.4** | 1.0–1.14x | ~32 MB |
| M3 Max | H15 (h15p) | 18 | 300–400 GB/s | ~9.5 | weights only | ~32 MB |
| M3 Ultra | H15 x2 | **36** | 819 GB/s | ~19 | weights only | ~64 MB |
| | | | | | | |
| **M4** | H16 (h16g) | **38** | 120 GB/s | ~11 | **1.88x (W8A8)** | ~32 MB |
| M4 Pro | H16 (h16p) | **38** | 273 GB/s | ~12 | **1.88x (W8A8)** | ~32 MB |
| M4 Max | H16 | **38** | 546 GB/s | ~11 | **1.88x (W8A8)** | ~32 MB |
| | | | | | | |
| **M5** | — | n/a | 153 GB/s | ~12–14† | TBD | ~32 MB |
| M5 Pro | — | n/a | 307 GB/s | ~12–14† | TBD | ~32 MB |
| M5 Max | — | n/a | 614 GB/s | ~12–14† | TBD | ~32 MB |

<sub>\* Measured FP16 TFLOPS on ANE (Conv 1x1). M2/M4 values from maderix/ANE benchmarks.</sub><br>
<sub>† M5 ANE estimate based on M3→M4 trend. Apple has not published separate ANE TOPS.</sub>

<details>
<summary><b>What does "INT8 weights only" vs "W8A8" mean?</b></summary>

&nbsp;

**M1/M2/M3 (INT8 weights only):** The ANE loads INT8 weights but **computes in FP16**. This saves memory and bandwidth but provides **no compute speedup**. That's why INT8 on M3 Pro only yields 1.0–1.14x.

**M4+ (W8A8 = Weights AND Activations INT8):** Only from M4 (H16) onwards can **both** — weights and activations — be processed in INT8. This doubles the effective throughput: **1.88x speedup**. INT8 training becomes realistic.

</details>

<details>
<summary><b>Estimated Training Performance per Chip</b></summary>

&nbsp;

Based on Stories110M (124 vocab, compacted), measured on M3 Pro = 91ms/step:

| Chip | Estimated ms/step | INT8 ms/step | Factor vs M3 Pro |
|:---|---:|---:|:---|
| M1 | ~160 ms | — | 0.6x |
| M2 | ~105 ms | — | 0.9x |
| **M3 Pro** | **91 ms** | — | _Baseline_ |
| M3 Ultra | ~50 ms | — | 1.8x |
| **M4** | **~45–55 ms** | **~25–30 ms** | **2x (4x with INT8)** |
| M4 Pro | ~45–55 ms | ~25–30 ms | 2x |
| M4 Max | ~40–50 ms | ~22–28 ms | 2–2.5x |
| M5 | ~35–45 ms | TBD | ~2.5x |
| M5 Pro | ~35–45 ms | TBD | ~2.5x |
| M5 Max | ~30–40 ms | TBD | ~3x |

</details>

<details>
<summary><b>Optimization Tips per Generation</b></summary>

&nbsp;

**M1 (11 TOPS)** — Lowest throughput. Tensor tiling is critical — keep working sets under 24MB. Use FP16, INT8 provides no benefit. QoS Background (9) yields the largest relative gain here.

**M2 (15.8 TOPS)** — 44% faster than M1, same constraints. No INT8 compute advantage. Same SRAM limits (~32MB).

**M3 / M3 Pro (18 TOPS)** — 14% above M2. No W8A8. Particularly flexible SRAM management (no hard cliff, gradual drop up to 73.5MB). Optimize for FP16, Conv 1x1 instead of matmul.

**M4 (38 TOPS)** — **The big leap**: 2x TOPS, true W8A8 INT8. Training steps are halved. INT8 quantization pays off for the first time. Same ~32MB SRAM.

**M5 (Fusion Architecture)** — ANE itself slightly faster, but the GPU Neural Accelerators (10–40 per chip) are the game-changer. Usable via Metal/MLX, not via libane. For maximum performance: hybrid ANE+GPU approach.

**Ultra Variants (M1–M3)** — Double ANE (32 cores, 2x TOPS). The only variant where Pro/Max → Ultra makes a real ANE difference. Both dies work in parallel — ideal for larger models that don't fit on 16 cores.

</details>

> [!IMPORTANT]
> The GPU Neural Accelerators in the M5 are accessible via **Metal/MLX**, **not** via the private ANE APIs that `libane` uses. For `libane` users, the primary benefits are the faster Neural Engine + higher memory bandwidth. For the full ~133 TOPS, Apple's MLX framework is needed — which is official and stable.

---

## Hardware Constraints & Bottlenecks

### Known ANE Limits

| Constraint | Value | Impact |
|:---|:---|:---|
| **SRAM On-Chip** | ~32 MB (all generations) | Tensors >32MB spill to DRAM → 30% throughput drop |
| **Compilation Limit** | ~119 per process | Then silent failures. Solved by Dynamic Spatial Packing |
| **IOSurface Minimum** | ~49 KB | Smaller tensors must be padded |
| **IOSurface Sorting** | Alphabetical by MIL name | Wrong order = silent failures |
| **`concat` Op** | Rejected | Must be split into separate programs |
| **`gelu` Op** | Not supported | Use tanh approximation |
| **Conv 1x1 vs matmul** | Conv is 3x faster | Express all matmuls as 1x1 Conv |
| **FP16 Overflow** | max ±65504 | Clamp activations before Softmax/RMSNorm |
| **Causal Masking** | Not native | `where()` MIL op as workaround |

<sub>Sources: Own research + [Orion Paper](https://arxiv.org/abs/2603.06728) (20 documented constraints)</sub>

### Bottleneck Analysis (M3 Pro, Stories110M)

```
Training Step = 91ms:

  ANE Forward          ██████░░░░░░░░░░░░░░  22ms  (24%)
  ANE Backward (dx)    ████████░░░░░░░░░░░░  15ms  (16%)
  CPU dW Gradients     ██████████░░░░░░░░░░  20ms  (22%)  ← Bottleneck
  CPU Attention/RoPE   ██████░░░░░░░░░░░░░░   8ms   (9%)
  CPU RMSNorm          ███░░░░░░░░░░░░░░░░░   5ms   (5%)
  CPU Adam Update      ██░░░░░░░░░░░░░░░░░░   3ms   (3%)
  Overhead             █████████░░░░░░░░░░░  18ms  (20%)

  ANE: 41%  ·  CPU: 59%
```

### Implemented Optimizations

| | Optimization | Result |
|:---|:---|:---|
| **done** | **Dynamic Spatial Packing** — Weights as IOSurface input | 60 compiles → **1 compile**, ~119 limit bypassed |
| **done** | **FP16 Overflow Protection** — Output/gradient sanitization | Prevents NaN/Inf divergence |
| **done** | **SRAM Budget Tracking** — Warning at >32MB | Diagnostics in `ane_compile()` |
| **done** | **Compile Budget Warning** — Warning at 110/119 | Safety net for legacy code |

### Open Optimizations

| Optimization | Gain | Effort |
|:---|:---|:---|
| **Pipeline Parallelism** — CPU Backward ‖ ANE Forward | ~40% latency | High |
| **Attention on ANE** — via `where()` MIL op | ~5ms CPU savings | High |
| **RMSNorm on ANE** — as MIL program | ~5ms CPU savings | Medium |
| **LoRA Adapter-as-Input** — Hot-swap fine-tuning | Zero recompile | Medium |

> Full optimization plan: **[ROADMAP.md](ROADMAP.md)**

---

## Project Structure

```
ANE-Training/
│
├── ane ······························· CLI Entry Point (./ane)
├── README.md ·························· This document
├── ARCHITECTURE.md ···················· 4-Layer Platform Architecture
├── ROADMAP.md ························· Optimization Plan (P0 completed)
├── RESEARCH_ANE_COMPLETE.md ··········· Complete Research Documentation
├── SUMMARY_TECHNICAL.md ·············· Technical Summary
├── SUMMARY_SIMPLE.md ················· Non-Technical Summary
├── LICENSE ···························· MIT
├── install.sh ························· One-Liner Installer
│
├── examples/ ·························· Runnable Demos
│   ├── demo_train.c                     Training Demo (Dynamic Spatial Packing)
│   ├── bench.c                          Auto-Benchmark
│   ├── generate.c                       Text Generation (Dynamic Spatial Packing)
│   ├── explore.m                        ANE Explorer
│   └── Makefile
│
└── libane/ ···························· Our C API
    ├── ane.h                            Stable API (never changes)
    ├── ane.m                            Implementation + Version Detection
    ├── test_ane.c                       Test Suite (3/3 passed)
    ├── README.md                        API Documentation
    └── Makefile
```

---

<details>
<summary><h2>Glossary</h2></summary>

&nbsp;

| Term | Explanation |
|:---|:---|
| **ANE** | Apple Neural Engine — AI accelerator in Apple Silicon |
| **MIL** | Model Intermediate Language — Apple's text format for models |
| **MLIR** | Multi-Level Intermediate Representation — compiler intermediate format |
| **LLIR** | Low-Level IR — machine-level representation before compilation |
| **HWX** | Hardware Executable — final binary for the ANE |
| **IOSurface** | macOS zero-copy shared memory between CPU and ANE |
| **QoS** | Quality of Service — priority level for ANE computations |
| **TFLOPS** | Tera Floating-Point Ops/Sec (10¹² FP ops) |
| **TOPS** | Tera Operations/Sec (counts each individual op) |
| **FP16** | 16-bit Float — the ANE's native compute format |
| **Conv 1x1** | 1x1 Convolution — 3x faster than matmul on ANE |
| **Dynamic Spatial Packing** | Weights as IOSurface input instead of BLOBFILE → 1x compile, ∞ training steps |

</details>

<details>
<summary><h2>Troubleshooting</h2></summary>

&nbsp;

### `ane_init()` returns -1 — Framework not found

```bash
ls /System/Library/PrivateFrameworks/AppleNeuralEngine.framework/
```

If empty: macOS too old or Apple changed the path.

### `ane_init()` returns -2 — Classes not found

Apple renamed private classes. `ane_init()` lists all found ANE classes on stderr. Enter new names in `libane/ane.m`, recompile.

### Compile error: `framework not found`

```bash
xcode-select --install
# or:
sudo xcode-select --reset
```

### `uname -m` shows `x86_64`

Intel Mac or Rosetta. ANE only exists on Apple Silicon:

```bash
sysctl sysctl.proc_translated 2>/dev/null
# 1 = Rosetta, 0 or error = native
```

### Lower TFLOPS than expected

- Don't run other compute-intensive processes
- Use QoS Background (9) — 42% faster than Default
- First run is slower due to kernel compilation (520ms one-time)

</details>

---

## Related Projects

| Project | Description |
|:---|:---|
| [maderix/ANE](https://github.com/maderix/ANE) | First training on ANE — inspiration for this project |
| [Orion Paper](https://arxiv.org/abs/2603.06728) | Academic paper: Delta Compilation, LoRA, 20 ANE constraints |
| [NeuralForge](https://github.com/Khaeldur/NeuralForge) | On-device LLM fine-tuning, process restart, GGUF export |
| [ANEMLL](https://github.com/Anemll/Anemll) | ANE Machine Learning Library |
| [hollance/neural-engine](https://github.com/hollance/neural-engine) | Community documentation (supported devices, internals) |
| [eiln/ane](https://github.com/eiln/ane) | Linux kernel driver — E5 binary format analysis |
| [SqueezeBits Yetter](https://blog.squeezebits.com/) | Disaggregated inference: ANE prefill + GPU decode |

---

> [!CAUTION]
> This project uses Apple's **private, undocumented APIs**. These can change with any macOS update. `libane` has version detection as protection — if Apple renames classes, only `ane.m` needs to be updated. Your code against `ane.h` remains unchanged.

<div align="center">

MIT License

</div>
