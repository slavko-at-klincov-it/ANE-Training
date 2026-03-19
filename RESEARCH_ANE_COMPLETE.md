# ANE Research — Complete API & Hardware Analysis
## Apple M3 Pro | macOS 26.3.1 | 2026-03-15

---

## 0. M3 PRO ANE HARDWARE IDENTITY (Discovered via Runtime Probing)

```
ANE Architecture Type:  h15g
ANE Sub Type:           h15
ANE Variant:            g       (likely "Pro" variant indicator)
ANE Cores:              16
ANE Units:              1
Board Type:             192
Virtual Machine:        NO
Power Drain When Idle:  NO      (hard power gating confirmed)
ANE Client:             Direct hardware (_ANEClient, not virtual)
Mach Service:           com.apple.appleneuralengine
```

### QoS Priority Levels (Discovered Values)
| Level | Value | Use Case |
|-------|-------|----------|
| Real Time | 0 | Lowest latency inference |
| Background | 9 | **Training (use this!)** |
| Utility | 17 | Batch inference |
| Default | 21 | Normal (repo uses this) |
| User Initiated | 25 | User-triggered inference |
| User Interactive | 33 | Real-time UI responses |

### ANE Compilation Pipeline (Discovered File Types)
```
model.mil → compile → model.bc.mlir → model.llir.bundle → model.hwx
(MIL text)           (MLIR bitcode)   (Low-Level IR)      (HW executable)
```

| File | Purpose |
|------|---------|
| `model.mil` | MIL source (input) |
| `model.bc.mlir` | MLIR bitcode (intermediate) |
| `model.llir.bundle` | Low-Level IR bundle (intermediate) |
| `model.hwx` | Hardware executable (final ANE binary) |
| `net.plist` | ANE CIR (circuit/compiler IR?) |
| `net_options.plist` | Compiler options |
| `weight.bin` | Weight data |
| `model.retain` | Cache retention marker |
| `model.src` | Source store reference |

### System Paths
| Path | Purpose |
|------|---------|
| `/Library/Caches/com.apple.aned/tmp` | System ANE temp |
| `/Library/Caches/com.apple.aneuserd/tmp` | User ANE temp |
| `/Library/Caches/com.apple.aned` | Model data vault |
| `/Library/Caches/com.apple.aned/clones` | Cloned models |
| `InMemoryModelCache` | In-memory model cache name |
| `com.apple.SHARED_SYSTEM_MODELS` | Shared system models cache |

### Architecture Generation Map
| Chip | Architecture | Sub Type | Notes |
|------|-------------|----------|-------|
| M3 Pro | **h15g** | h15 variant g | Your machine |
| M4 | h16g | h16 variant g | Repo reference (from substack) |
| M1/M2 | h13/h14 (estimated) | Unknown | Not tested |

---

## 1. YOUR M3 PRO BENCHMARK RESULTS

> **Note:** These are early benchmark results with suboptimal parameters (small spatial, untuned channels).
> After systematic tuning (see [ANE_PERFORMANCE_TUNING.md](docs/ANE_PERFORMANCE_TUNING.md)):
> - ANE Silicon Peak: **12.79 TFLOPS** (512ch sp128 depth128)
> - Sustained Single-Kernel: **5.01 TFLOPS**
> - Real Training (Stories-110M): **2.15 TFLOPS** / 80.9 ms/step (pipeline), **1.87 TFLOPS** / 93 ms/step (sequential)

### Throughput (Single Conv)
| Config | Weight (MB) | ms/eval | TFLOPS |
|--------|------------|---------|--------|
| 256ch x64sp | 0.1 | 0.248 | 0.03 |
| 512ch x64sp | 0.5 | 0.219 | 0.15 |
| 1024ch x64sp | 2.0 | 0.236 | 0.57 |
| 2048ch x64sp | 8.0 | 0.334 | 1.61 |
| 3072ch x64sp | 18.0 | 0.494 | 2.45 |
| 4096ch x64sp | 32.0 | 0.736 | 2.92 |

### Peak Sustained (Sequential Convs) — Historical (early measurements)

> **Note:** Current peak is **12.79 TFLOPS** (512ch sp128 depth128, after systematic sweep).
> These early results were measured before the performance tuning in v2.0.0.

| Config | Weight (MB) | GFLOP | ms/eval | TFLOPS |
|--------|------------|-------|---------|--------|
| 128x conv 512ch sp64 | 64.0 | 4.29 | 0.459 | **9.36** |
| 256x conv 256ch sp64 | 32.0 | 2.15 | 0.303 | 7.09 |
| 128x conv 384ch sp64 | 36.0 | 2.42 | 0.314 | 7.70 |

### INT8 vs FP16 (Large Spatial 64x64)
| Config | FP16 TOPS | INT8 TOPS | Ratio |
|--------|-----------|-----------|-------|
| 128x conv 512ch | 18.23 | 18.25 | 1.00x |
| 256x conv 256ch | 15.16 | 17.23 | 1.14x |
| 128x conv 256ch | 14.79 | 16.74 | 1.13x |

### SRAM Behavior
No sharp cliff detected — throughput scales gradually up to 73.5MB total, then drops at 129MB. M3 Pro appears to have more flexible SRAM/cache management than M4.

### M3 Pro vs M4 Comparison
| Metric | M3 Pro | M4 |
|--------|--------|-----|
| Apple marketing TOPS | 18 | 38 |
| Measured peak (stacked benchmark) | 12.79 TFLOPS | 15.8 TFLOPS |
| Measured peak (large spatial) | 18.23 TOPS | 19.0 TOPS |
| INT8 acceleration | 1.0-1.14x | 1.88x |
| INT8 peak | 18.25 TOPS | 35.1 TOPS |
| ANE cores | 16 | 16 |
| Key difference | INT8 barely helps | INT8 nearly doubles throughput |

---

## 2. COMPLETE PRIVATE API SURFACE (35 Classes, macOS 26.3.1)

### Classes Used by maderix/ANE (4 of 35)
| Class | Methods Used | Total Methods Available |
|-------|-------------|----------------------|
| `_ANEInMemoryModelDescriptor` | 1 | 17 |
| `_ANEInMemoryModel` | 6 | 44 |
| `_ANERequest` | 1 | 27 |
| `_ANEIOSurfaceObject` | 1 | 16 |

### UNDISCOVERED Classes with High Potential

#### `_ANEChainingRequest` — Pipeline Multiple Kernels Without CPU Roundtrip
```objc
+chainingRequestWithInputs:outputSets:lbInputSymbolId:lbOutputSymbolId:
                          procedureIndex:signalEvents:transactionHandle:
                          fwEnqueueDelay:memoryPoolId:
```
- Has **loopback** input/output — output of one kernel feeds input of next
- **signalEvents** — synchronization between chained operations
- **memoryPoolId** — shared memory pool across chain
- **Potential**: Chain forward pass kernels without CPU copy between layers

#### `_ANEClient` — Full Hardware Client (46 instance methods)
```objc
+sharedConnection              // Get shared ANE client
-evaluateRealTimeWithModel:    // Real-time evaluation (lower latency?)
-loadRealTimeModel:            // Real-time model loading
-unloadRealTimeModel:          // Real-time unload
-prepareChainingWithModel:     // Set up model chaining
-buffersReadyWithModel:        // Async buffer readiness
-enqueueSetsWithModel:         // Enqueue output sets
-mapIOSurfacesWithModel:       // Map IOSurface for model
-unmapIOSurfacesWithModel:     // Unmap IOSurface
-sessionHintWithModel:         // Session hints (caching?)
-beginRealTimeTask             // Enter real-time mode
-endRealTimeTask               // Exit real-time mode
```

#### `_ANEPerformanceStats` — Hardware Performance Counters
```objc
-hwExecutionTime               // Actual hardware execution time (ns)
-perfCounterData               // Raw performance counter data
-performanceCounters            // Parsed performance counters
-stringForPerfCounter:          // Human-readable counter names
+driverMaskForANEFMask:         // Filter which counters to enable
```
- Can measure real ANE utilization, not just wall clock time
- Counter data could reveal cache misses, stalls, utilization per core

#### `_ANEDeviceInfo` — Hardware Introspection
```objc
+hasANE                         // Check ANE presence
+numANECores                    // Number of ANE cores
+numANEs                        // Number of ANE units
+aneArchitectureType            // Architecture family
+aneSubType                     // Chip-specific subtype
+aneBoardType                   // Board identifier
+buildVersion                   // System build
+isExcessivePowerDrainWhenIdle  // Power management flag
```

#### `_ANESharedEvents` — GPU↔ANE Synchronization
```objc
-signalEvents                   // Events to signal on completion
-waitEvents                     // Events to wait for before starting
```
With `_ANESharedSignalEvent` and `_ANESharedWaitEvent`:
- Enable GPU→ANE→CPU pipeline without polling
- Could synchronize training steps across accelerators

#### `_ANEQoSMapper` — 6 Quality-of-Service Levels
```objc
+aneRealTimeTaskQoS            // Highest priority
+aneUserInteractiveTaskQoS     // Interactive
+aneUserInitiatedTaskQoS       // User-initiated
+aneDefaultTaskQoS             // Default
+aneUtilityTaskQoS             // Background utility
+aneBackgroundTaskQoS          // Lowest priority (for training!)
```
- Training should use `aneBackgroundTaskQoS` to not interfere with system
- `aneRealTimeTaskQoS` for inference serving

#### `_ANEProgramIOSurfacesMapper` — Memory Mapping
```objc
-mapIOSurfacesWithModel:request:cacheInference:error:
-unmapIOSurfacesWithModel:request:error:
```
- The `cacheInference:` parameter suggests pre-mapped I/O for repeated inference
- Could reduce per-eval overhead

#### `_ANEWeight` — Weight Management
```objc
-initWithWeightSymbolAndURL:weightURL:
-updateWeightURL:              // Update weights without recompile?
-SHACode                       // SHA verification for weight integrity
```

#### `_ANEInputBuffersReady` — Async Input Notification
```objc
+inputBuffersWithProcedureIndex:inputBufferInfoIndex:inputFreeValue:executionDelay:
```
- **executionDelay** — schedule evaluation with delay
- Could enable pipeline overlap: prepare next batch while current runs

#### `_ANEIOSurfaceObject` — Extended Surface Creation
```objc
+createIOSurfaceWithWidth:pixel_size:height:                  // Proper pixel-aware creation
+createIOSurfaceWithWidth:pixel_size:height:bytesPerElement:   // With element size
+objectWithIOSurface:startOffset:                              // Offset into shared surface
```
- The `startOffset:` method means multiple tensors could share ONE IOSurface
- Could reduce surface creation overhead dramatically

---

## 3. UNDISCOVERED MIL OPERATIONS

### Operations Used in Repo (17)
conv, matmul, reshape, transpose, slice_by_size, concat, sigmoid, softmax, mul, add, sub, pow, reduce_sum, cast, const, div (implicit)

### Additional MIL Operations Available (from coremltools)
Based on Apple's coremltools MIL specification, these additional ops exist but are NOT used in the repo:

**Activations:** relu, leaky_relu, elu, prelu, gelu (tanh/erf approximation), thresholded_relu, celu, selu, scaled_tanh, linear_activation, hard_sigmoid, hard_swish
**Normalization:** layer_norm, batch_norm, instance_norm, l2_norm, local_response_norm
**Pooling:** avg_pool, max_pool, l2_pool
**Reduce:** reduce_mean, reduce_max, reduce_min, reduce_prod, reduce_l2_norm, reduce_argmax, reduce_argmin
**Tensor ops:** gather, scatter, tile, pad, expand_dims, squeeze, stack, split, reverse, cumsum, fill_like, non_zero, topk, argsort
**Conv variants:** conv_transpose, depthwise_conv
**Element-wise:** abs, ceil, floor, round, sign, clip, cos, sin, tan, exp, log, sqrt, rsqrt, logical_and, logical_or, logical_not, equal, not_equal, greater, greater_equal, less, less_equal, where (ternary select)
**Quantization:** constexpr_affine_dequantize, constexpr_lut_to_dense, constexpr_sparse_to_dense, quantize, dequantize
**Other:** einsum, one_hot, shape, random (normal/uniform), range_1d, cond (conditional), while_loop, list operations

### Key Unlockable Operations for Training
1. **gelu** — could replace SiLU for BERT-style models
2. **layer_norm** — native op instead of manual mul/reduce_sum/pow
3. **gather/scatter** — could move embedding lookup to ANE
4. **where** (ternary select) — could implement causal masking on ANE
5. **reduce_mean** — simplify RMSNorm computation
6. **cumsum** — could help with position-dependent operations
7. **topk** — could accelerate sampling during generation
8. **einsum** — flexible tensor contractions

---

## 4. CODE GAPS & WHAT NEEDS FIXING

### Critical for M3 Pro
1. **Hardcoded 15.8 TFLOPS** → updated to 12.79 peak / 2.15 real training for M3 Pro
   - Files: train_large.m, tiny_train.m, dashboard.py
2. **INT8 training path missing** — only inference quantization exists; and on M3 Pro INT8 barely helps anyway

### Training Accuracy Issues
3. **RMSNorm FFN backward approximation** — uses `act_x[l]` instead of `x[l] + attn_residual`, causing gradient error
4. **No NaN/Inf detection** — gradient explosion goes unnoticed
5. **No validation set** — can't detect overfitting

### Missing Features
6. **Qwen3 pretrained weight loading** — errors out, only --scratch works
7. **No standalone inference** — dashboard.py requires training loop
8. **No data preprocessing pipeline** — user must pre-tokenize manually
9. **No multi-batch** — single sequence per step

### Architecture Limitations
10. **Static training doesn't support GQA** — only dynamic path does
11. **CPU attention** — causal masking forces attention to CPU
12. **~119 compile limit** — exec() restart is fragile

---

## 5. WHAT WE CAN BUILD: OUR OWN API

### Proposed Architecture: `libane.h`

Based on the 35 discovered classes, we can build a comprehensive C API:

```c
// === Device Discovery ===
typedef struct {
    int has_ane;
    int num_cores;
    int num_ane_units;
    const char *arch_type;      // "H16G" for M4, etc.
    const char *sub_type;
    const char *board_type;
} ANEDeviceInfo;
ANEDeviceInfo ane_get_device_info(void);

// === Compilation ===
typedef struct ANEKernel ANEKernel;
ANEKernel *ane_compile(const char *mil_text,
                        ANEWeightBlob *weights, int n_weights,
                        ANEQoS qos);

// === QoS Levels ===
typedef enum {
    ANE_QOS_BACKGROUND,         // For training (lowest priority)
    ANE_QOS_UTILITY,            // For batch inference
    ANE_QOS_DEFAULT,            // Normal
    ANE_QOS_USER_INITIATED,     // User-triggered inference
    ANE_QOS_USER_INTERACTIVE,   // Real-time UI
    ANE_QOS_REALTIME            // Lowest latency
} ANEQoS;

// === Tensor I/O ===
typedef struct ANETensor ANETensor;
ANETensor *ane_tensor_create(int channels, int height, int width, ANEDtype dtype);
ANETensor *ane_tensor_from_surface(IOSurfaceRef surf, size_t offset);  // Shared surface
void ane_tensor_write(ANETensor *t, const void *data, size_t bytes);
void ane_tensor_read(ANETensor *t, void *data, size_t bytes);

// === Execution ===
void ane_eval(ANEKernel *k, ANETensor **inputs, int n_in,
              ANETensor **outputs, int n_out);

// === Chaining (NEW — pipeline without CPU roundtrip) ===
typedef struct ANEChain ANEChain;
ANEChain *ane_chain_create(void);
void ane_chain_add(ANEChain *c, ANEKernel *k, int proc_idx);
void ane_chain_set_loopback(ANEChain *c, int in_sym, int out_sym);
void ane_chain_execute(ANEChain *c);

// === Performance Monitoring (NEW) ===
typedef struct {
    uint64_t hw_execution_ns;
    double ane_utilization;
    // per-counter data TBD
} ANEPerfStats;
ANEPerfStats ane_get_perf_stats(ANEKernel *k);

// === Events (NEW — GPU/CPU synchronization) ===
typedef struct ANEEvent ANEEvent;
ANEEvent *ane_event_create(void);
void ane_eval_with_events(ANEKernel *k, ANETensor **in, int n_in,
                          ANETensor **out, int n_out,
                          ANEEvent *signal_on_complete,
                          ANEEvent *wait_before_start);

// === Weight Management ===
ANEWeightBlob *ane_weight_fp16(const float *data, int rows, int cols);
ANEWeightBlob *ane_weight_int8(const float *data, int rows, int cols, float *scale_out);
void ane_kernel_update_weights(ANEKernel *k, ANEWeightBlob *w);  // Via _ANEWeight::updateWeightURL

// === MIL Generation Helpers ===
char *ane_mil_linear(int in_ch, int out_ch, int seq, const char *weight_name);
char *ane_mil_attention(int dim, int heads, int seq, bool gqa, int kv_heads);
char *ane_mil_ffn(int dim, int hidden, int seq, const char *activation);
char *ane_mil_rmsnorm(int dim, int seq);
char *ane_mil_rope(int dim, int seq, int max_pos);

// === Cleanup ===
void ane_kernel_free(ANEKernel *k);
void ane_tensor_free(ANETensor *t);
void ane_chain_free(ANEChain *c);
```

---

## 6. HARDWARE ARCHITECTURE INSIGHTS

### ANE Execution Model
- **E5 binary format**: FlatBuffer-structured, remarkably compact (~2.7KB for 1024×1024 matmul)
- **Parameterized compute primitives**: Binary size barely changes with matrix size → ANE uses parameterized engines, not custom microcode
- **Conv is the primary primitive**: All linear algebra maps to 1×1 convolutions internally
- **Queue depth: 127**: Can have 127 evaluation requests in flight simultaneously
- **Independent DVFS**: ANE clocks independently from CPU/GPU
- **Hard power gating**: 0mW when idle — no penalty for having ANE available

### Memory Layout Requirements
- Tensors: 4D, channels-first `[1, C, 1, S]`
- **Last axis must be 64-byte aligned** — misalignment causes 32-64× overhead!
- IOSurface: shared DMA-capable memory, zero-copy between CPU/GPU/ANE
- Weight blobs: custom binary format with 64-byte headers + FP16/INT8 data

### Compilation Pipeline
```
MIL text → _ANEInMemoryModelDescriptor → compile → E5 binary → load → evaluate
           (typed SSA IR)                         (FlatBuffer)   (ANE slots)
```
- First compile: 20-40ms
- Cache hits: effectively free
- ~119 compilations per process before ANE refuses

### M3 Pro Specific
- 16 ANE cores (same as M3, M4)
- Architecture type queryable via `_ANEDeviceInfo::aneArchitectureType`
- INT8 acceleration minimal (1.0-1.14x vs M4's 1.88x)
- ANE Silicon Peak: 12.79 TFLOPS (128x stacked benchmark), Real Training: 2.15 TFLOPS (80.9 ms/step)
- Suggested training QoS: `aneBackgroundTaskQoS` to avoid system interference

---

## 7. VERIFIED: TRAINING WORKS ON M3 PRO

```
$ ./train --scratch
=== ANE Dynamic Training: Stories110M (12 layers, GQA 12/12 heads) ===
dim=768 q_dim=768 kv_dim=768 hd=64 hidden=2048 seq=256 vocab=32000
Params: 109.5M (transformer 85.0M + embed 24.6M)
Kernels: 10 compiled ✓

=== ANE Dynamic Training: Qwen3-0.6B (28 layers, GQA 16/8 heads) ===
dim=1024 q_dim=2048 kv_dim=1024 hd=128 hidden=3072 seq=256 vocab=151936
Params: 596.0M (transformer 440.5M + embed 155.6M)
Kernels: 10 compiled ✓
```

Both model architectures compile and initialize successfully. Training requires pre-tokenized data file (tinystories_data00.bin).

---

## 8. WEB RESEARCH FINDINGS — OTHER PROJECTS & PAPERS

### Orion Paper (arxiv:2603.06728, March 2026)
First academic paper on ANE programming. Key discoveries:
- **Conv 1x1 is 3×  faster than matmul** on ANE — conv is the native primitive
- **`concat` MIL op is REJECTED** by ANE compiler — must split into separate programs
- **GELU activation unsupported** — use tanh approximation
- **M4 Max SRAM: ~32 MB** — 30% throughput drop when exceeded
- **~119 compilation limit** confirmed with silent failures
- **20 ANE programming constraints** documented (14 newly discovered)
- Achieved: 170+ tok/s GPT-2 124M inference, trained 110M model in 22 minutes
- Implements **LoRA adapter-as-input** (hot-swap without recompilation)

### Other Reverse Engineering Projects

| Project | What It Does |
|---------|-------------|
| **eiln/ane** (Asahi Linux) | Full Linux kernel driver (`ane.ko`) + userspace lib + Python bindings |
| **Khaeldur/NeuralForge** | SwiftUI app for on-device LLM fine-tuning using maderix/ANE |
| **Anemll/Anemll** | Model conversion → ANE inference pipeline, 47-62 tok/s on 1B models |
| **antgroup-skyward/ANETools** | **ANE Disassembler** — disassembles .hwx files, prints all registers/bits |
| **freedomtan/coreml_to_ane_hwx** | Direct CoreML → HWX conversion |
| **smpanaro/more-ane-transformers** | Optimized transformer deployment via CoreML |
| **hollance/neural-engine** | Community documentation of everything known about ANE |
| **mdaiter/ane** | Early Python/ObjC reverse engineering |

### Orion's 20 ANE Programming Constraints

**MIL Restrictions:**
1. `concat` op rejected by ANE compiler
2. GELU unsupported (use tanh approximation)
3. Convolution doesn't support bias (use separate `add`)
4. 32K+ channel convolutions rejected
5. matmul transpose flags require named constants
6. Output variables must reference live post-optimization nodes

**Memory & I/O:**
7. Multi-output buffers must have **uniform sizes**
8. Outputs ordered **alphabetically** by MIL variable name
9. Minimum ~49 KB IOSurface for evaluation
10. Multi-input surfaces need uniform allocation sizes
11. Inputs ordered alphabetically by parameter name
12. BLOBFILE offset uses `uint64(64)`, not 128
13. MIL text requires `NSData*`, not `NSString*`
14. Weight dictionary must be `@{}`, never `nil`

**Compilation:**
15. ~119 compilations per process, then silent failures
16. `exec()` restart overhead ~50ms
17. Weights baked at compile time (recompile or delta reload needed)

**Performance:**
18. **Conv 1x1 is 3× faster than matmul**
19. M4 Max SRAM ~32 MB (M3 Pro likely ~16 MB based on perf knee)
20. ANE reads flat buffers as packed `[1,C,1,S]` from byte 0

### ANE Hardware Architecture (from Apple Patents)

**US20190340486 — Multiply-Accumulate Architecture:**
- Neural engine cores with MAC circuits, data buffer, kernel fetcher
- Data buffer broadcasts same input to all activated engines
- Each engine applies different kernel coefficients
- Work hierarchy: Slices → Tiles → Work units (sized to fit accumulator)

**US20190340491 — Scalable Processing Engine:**
- Selectively activate/deactivate cores (2, 8, or 16)
- Kernel DMA and Buffer DMA for data transfer
- Power management via selective activation

**US20210103803 — Multi-Mode Planar Engine:**
- Three modes: pooling, elementwise, reduction
- Handles within-channel operations (I/O-bound)
- Neural engines handle cross-channel operations (compute-heavy)
- Components: ReLU, transposition, broadcasting, normalization, sqrt, inversion

### E5 Binary Format
- **FlatBuffer structure**, ~2.7 KB regardless of matrix size
- **Parameterized compute primitives** — describes which fixed primitives to chain, not custom microcode
- Cached at: `~/Library/Caches/<app>/com.apple.e5rt.e5bundlecache/<build>/<hash>/H16G.bundle/`

### ANE-Verified MIL Operations (from Orion)
Working: `conv1x1`, `matmul`, `add`, `sub`, `mul`, `neg`, `relu`, `tanh`, `sigmoid`, `exp`, `pow`, `sqrt`, `rsqrt`, `reduce_sum`, `reduce_mean`, `reduce_max`, `reshape`, `transpose`, `split`, `pad`, `slice`, `cast`, `softmax`

**Broken/rejected:** `concat` (compilation failure), `gelu` (unsupported), `gather` (CPU only), large-channel conv (>32K)

### Key Optimization Insight
> "Conv 1x1 is 3× faster than matmul on ANE."

This means the maderix repo's approach of mapping all linear layers as 1×1 convolutions is correct and optimal. The dynamic path's use of matmul (for spatial-packed weights) may be leaving performance on the table.

---

## 9. COMPLETE MIL OPERATION CATALOG (from coremltools)

| Category | Operations |
|----------|-----------|
| **Activations** | relu, leaky_relu, elu, prelu, gelu, silu, sigmoid, softmax, softplus, softsign, tanh, relu6, scaled_tanh, thresholded_relu, clamped_relu, hard_sigmoid, hard_swish, sigmoid_hard, linear_activation, softplus_parametric |
| **Linear Algebra** | conv, conv_transpose, matmul, linear, einsum |
| **Elementwise Binary** | add, sub, mul, real_div, floor_div, mod, pow, maximum, minimum, equal, not_equal, greater, greater_equal, less, less_equal, logical_and, logical_or, logical_xor |
| **Elementwise Unary** | abs, ceil, floor, round, sign, clip, cos, sin, tan, exp, exp2, log, sqrt, rsqrt, erf, cast, inverse, square, acos, asin, atan, atanh, cosh, sinh, tanh, threshold, logical_not |
| **Reduction** | reduce_sum, reduce_mean, reduce_max, reduce_min, reduce_prod, reduce_argmax, reduce_argmin, reduce_l1_norm, reduce_l2_norm, reduce_log_sum, reduce_log_sum_exp, reduce_sum_square |
| **Shape** | reshape, transpose, expand_dims, squeeze, concat, split, stack, slice_by_size, slice_by_index, pad, tile, reverse, reverse_sequence, flatten2d |
| **Gather/Scatter** | gather, gather_along_axis, gather_nd, scatter, scatter_along_axis, scatter_nd |
| **Pooling** | avg_pool, max_pool, l2_pool |
| **Normalization** | batch_norm, instance_norm, l2_norm, layer_norm, local_response_norm |
| **Tensor Ops** | argsort, band_part, cumsum, fill, non_zero, one_hot, range_1d, shape, topk, non_maximum_suppression, sliding_windows, identity |
| **Spatial** | batch_to_space, space_to_batch, depth_to_space, space_to_depth, pixel_shuffle |
| **Quantization** | quantize, dequantize, constexpr_affine_dequantize, constexpr_cast, constexpr_lut_to_dense, constexpr_sparse_to_dense |
| **Control Flow** | cond, while_loop, select, make_list, list_* |
| **Recurrent** | gru, lstm, rnn |
| **Transformers (iOS18+)** | scaled_dot_product_attention |

**Bold = verified working on ANE.** Most others fall back to CPU/GPU.

---

## 10. RESEARCH SOURCES

- [maderix/ANE](https://github.com/maderix/ANE) — Primary reverse engineering project
- [maderix Substack Part 1](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine) — API discovery
- [maderix Substack Part 2](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615) — Benchmarks
- [Orion Paper (arxiv:2603.06728)](https://arxiv.org/html/2603.06728v1) — Academic paper on ANE programming
- [hollance/neural-engine](https://github.com/hollance/neural-engine) — Community documentation
- [eiln/ane](https://github.com/eiln/ane) — Asahi Linux ANE driver
- [ANETools + Disassembler](https://github.com/antgroup-skyward/ANETools) — HWX disassembly
- [Apple ML Research: Deploying Transformers on ANE](https://machinelearning.apple.com/research/neural-engine-transformers)
- [coremltools MIL Ops Reference](https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html)
- [nst/iOS-Runtime-Headers](https://github.com/nst/iOS-Runtime-Headers/tree/master/PrivateFrameworks/AppleNeuralEngine.framework)
- [Apple Patent US20190340486](https://uspto.report/patent/app/20190340486) — MAC architecture
- [Apple Patent US20190340491](https://patents.google.com/patent/US20190340491A1/en) — Scalable engine
- [Apple Patent US20210103803](https://patents.google.com/patent/US20210103803A1/en) — Planar engine
- [Black Hat Asia 2021 — Wish Wu: ANE Internals](https://i.blackhat.com/asia-21/Friday-Handouts/as21-Wu-Apple-Neural_Engine.pdf)
- [Anemll](https://github.com/Anemll/Anemll) — Model conversion + ANE inference
- [NeuralForge](https://github.com/Khaeldur/NeuralForge) — SwiftUI fine-tuning app

---

## 11. PHASE A TEST RESULTS (Chaining & Advanced APIs)

### TEST 1: Basic ANE Eval — WORKS
Identity convolution (256ch × 64sp): input 1.0 → output 1.0. ANE computation is correct.

### TEST 2: QoS Impact on Latency — SURPRISING RESULT
| QoS Level | Value | ms/eval | Notes |
|-----------|-------|---------|-------|
| **Background** | **9** | **0.143** | **FASTEST — use for training!** |
| Utility | 17 | 0.227 | |
| Default | 21 | 0.248 | What the repo uses |
| UserInitiated | 25 | 0.249 | |
| UserInteractive | 33 | 0.247 | |

**Background QoS is 42% faster than Default!** Higher priorities add scheduling overhead.
For training, use QoS=9 (Background) — it's both fastest AND lowest system impact.

### TEST 3: IOSurface Offsets — PARTIAL
- `objectWithIOSurface:startOffset:` with offset=0 works
- Non-zero offsets crash (calling convention issue, needs more investigation)

### TEST 4: _ANEBuffer — WORKS
Successfully created `_ANEBuffer` objects with IOSurface + symbolIndex binding.
```
_ANEBuffer: { ANEIOSurfaceObject=..., symbolIndex=0, ANEBufferProducerAgent=... }
```

### TEST 5: _ANEChainingRequest — PARTIALLY WORKS
- **Object creation: SUCCESS** — ChainingRequest created with input/output buffers
- **validate: FAILS** — Parameter type mismatches (some NSNumber args should be arrays)
- **prepareChainingWithModel: FAILS** — Same issue
- **Status:** Needs further reverse engineering of exact parameter types
- The validate method internally calls `count` on parameters, suggesting some scalar params should be arrays

### TEST 6: _ANEClient — WORKS
- Direct hardware client obtained
- Not virtual, no restricted access

### TEST 7: perfStats — NOT WORKING YET
- `setPerfStatsMask:` called but `perfStats` returns nil after eval
- May require entitlements or a different init path
- The `_ANEPerformanceStatsIOSurface` class exists but needs testing

### Key Takeaways
1. **Use QoS=9 for training** — 42% faster than default!
2. **_ANEBuffer and _ANEChainingRequest objects can be created** — the pipeline exists
3. **Chaining needs more reverse engineering** of parameter types in validate
4. **perfStats needs entitlements** or different activation method

---

## 12. NEXT STEPS

### Phase 1: Probe Undiscovered APIs
- [ ] Call `_ANEDeviceInfo` to get M3 Pro architecture identifiers
- [ ] Enable `_ANEPerformanceStats` to measure real ANE utilization
- [ ] Test `_ANEChainingRequest` for pipelined execution
- [ ] Test `_ANEQoSMapper` QoS levels and their impact on throughput/latency
- [ ] Test `_ANEIOSurfaceObject::createIOSurfaceWithWidth:pixel_size:height:` vs manual creation

### Phase 2: Build Our Own API (`libane`)
- [ ] Wrap all 35 classes into clean C API
- [ ] Add device introspection
- [ ] Add performance monitoring
- [ ] Add chaining/pipelining support
- [ ] Add proper QoS management
- [ ] Test shared IOSurface offsets for multi-tensor packing

### Phase 3: Training Pipeline
- [ ] Fix RMSNorm backward approximation
- [ ] Add NaN/Inf detection
- [ ] Add validation set support
- [ ] Build data preprocessing pipeline
- [ ] Test additional MIL ops (layer_norm, gelu, gather, where)
- [ ] Explore `_ANEChainingRequest` for layer-to-layer pipelining
- [x] Add proper M3 Pro TFLOPS values (12.79 peak, 2.15 real training)

### Phase 4: Personal AI System
- [ ] File watcher daemon (FSEvents)
- [ ] Tokenizer/preprocessor for mixed document types
- [ ] Nightly training scheduler (launchd + background QoS)
- [ ] Inference server for queries
- [ ] Incremental learning (fine-tune on new data only)
