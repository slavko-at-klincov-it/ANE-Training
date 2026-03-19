# Architecture — ANE Training Platform

## What is this?

This document describes the 4-layer architecture behind the project — from research to the finished application. It shows how everything fits together and why certain decisions were made.

The public repository contains **Layer 1 (Research) and Layer 2 (libane)** with runnable demos. Layer 3 (Training Pipeline) is based on [maderix/ANE](https://github.com/maderix/ANE) and is referenced here. Layer 4 (Personal AI) is a separate project that builds on top of the platform.

The platform is reusable. You could build a code assistant, a log analyzer, a local chatbot, or an image classifier on top of it.

---

## Layer 1: Research & Discovery

### What we did
Reverse engineering of Apple's private Neural Engine framework through:
- Runtime introspection of all ObjC classes via `objc_copyClassList`
- Systematic probing of methods via `objc_msgSend` with crash recovery
- Benchmark suite executed on our own M3 Pro hardware
- Web research on all known RE projects and the Orion paper

### What we discovered

**35 private classes** discovered (the original repo uses only 4):

| Class | What it does | Used? |
|-------|-------------|-------|
| `_ANEInMemoryModelDescriptor` | MIL text + Weights → Model descriptor | Yes (libane) |
| `_ANEInMemoryModel` | Compile → Load → Evaluate → Unload | Yes (libane) |
| `_ANERequest` | Binds IOSurface I/O to evaluation | Yes (libane) |
| `_ANEIOSurfaceObject` | Wrapper for IOSurface (zero-copy) | Yes (libane) |
| `_ANEClient` | Direct hardware connection to the ANE daemon | Yes (probing) |
| `_ANEDeviceInfo` | Hardware detection (architecture, cores, board) | Yes (libane) |
| `_ANEQoSMapper` | 6 QoS levels for priority control | Yes (probing) |
| `_ANEChainingRequest` | Kernel pipeline without CPU roundtrip | Partial (objects created, validate not yet) |
| `_ANEBuffer` | IOSurface + symbol index binding | Yes (probing) |
| `_ANEPerformanceStats` | Hardware performance counters | No (requires entitlements?) |
| 25 more | Daemon connection, virtualization, weight management, etc. | Cataloged |

**Hardware identity of the M3 Pro** (via `_ANEDeviceInfo`):
```
Architecture: h15g (M4 = h16g)
Cores:        16
Units:        1
Board type:   192
Power gating: 0mW when idle
```

**6 QoS levels** discovered and benchmarked:
```
Background(9):      0.143 ms  ← FASTEST! Use for training.
Utility(17):        0.227 ms
Default(21):        0.248 ms  ← What the original repo uses.
UserInitiated(25):  0.249 ms
UserInteractive(33):0.247 ms
```
**Finding:** Background QoS is 42% faster than Default because of less scheduling overhead.

**Compilation pipeline** (previously unknown):
```
model.mil → model.bc.mlir → model.llir.bundle → model.hwx
(MIL Text)   (MLIR Bitcode)  (Low-Level IR)     (HW Binary)
```

**Performance benchmarks** (M3 Pro):
- ANE Silicon Peak: 12.79 TFLOPS (128x stacked conv benchmark)
- Sustained Single-Kernel: 5.01 TFLOPS (continuous eval)
- Real Training: 2.15 TFLOPS / 80.9 ms/step (pipeline), 1.87 TFLOPS / 93 ms/step (sequential)
- INT8 yields only 1.0-1.14x (on M4: 1.88x) → not worth it on M3

**20 ANE constraints** (from the Orion paper):
- `concat` is rejected by the ANE compiler
- `gelu` not supported (use tanh approximation)
- Conv 1x1 is 3x faster than matmul
- ~119 compilations per process, then exec() restart required
- Last tensor axis must be 64-byte aligned

### Where is this documented?
- `RESEARCH_ANE_COMPLETE.md` — Complete research documentation (benchmarks, API surface, constraints, sources)
- `SUMMARY_TECHNICAL.md` — Technical summary of the maderix/ANE repo
- `SUMMARY_SIMPLE.md` — Non-technical summary with use cases

---

## Layer 2: libane — Our Own C API

### Why?
The maderix/ANE repo has its own bridge code (`bridge/ane_bridge.m`), but:
- No version detection (breaks if Apple renames classes)
- No device detection
- No QoS support (everything on Default=21)
- No clean API separation

### What we built

```
libane/
├── ane.h          ← Stable C API (NEVER changes)
├── ane.m          ← Implementation (changes when Apple API changes)
├── libane.dylib   ← Shared library (73KB, arm64)
├── test_ane.c     ← Test suite (3/3 passed)
└── Makefile
```

### How version detection works

```c
// ane.m tries known class names in order:
static const char *MODEL_CLASS_NAMES[] = {
    "_ANEInMemoryModel",    // current (macOS 15-26)
    "_ANEModel",            // in case Apple renames
    "ANEInMemoryModel",     // without underscore
    "ANEModel",             // completely new
    NULL
};
```

On each `ane_init()`:
1. Load framework (3 known paths are tried)
2. Resolve classes (try alternatives)
3. Resolve selectors (try method names)
4. Determine API version (1 = known, 0 = unknown)
5. If unknown: list all ANE classes for debugging

**If Apple changes the API:**
- `ane.h` stays the same (your code doesn't change)
- Add new class/method names in `ane.m`
- Recompile, done

### API Overview

```c
// Initialization + hardware detection
ane_init();
ANEDeviceInfo info = ane_device_info();   // h15g, 16 cores, etc.
ANEAPIInfo api = ane_api_info();          // version, classes found, selectors resolved
ane_print_diagnostics();                  // Prints everything to stderr

// Compilation
ANEWeight w = ane_weight_fp16("@model_path/weights/w.bin", data, rows, cols);
char *mil = ane_mil_linear(in_ch, out_ch, seq, weight_name);
ANEKernel *k = ane_compile(mil, len, &w, 1, 1, &in_sz, 1, &out_sz, ANE_QOS_BACKGROUND);

// Evaluation
ane_write(k, 0, input_data, bytes);       // Data → IOSurface
ane_eval(k, ANE_QOS_BACKGROUND);          // Execute ANE
ane_read(k, 0, output_data, bytes);       // IOSurface → Data

// Zero-copy (faster, no memcpy)
ane_lock_input(k, 0);
float *ptr = (float *)ane_input_ptr(k, 0);
ptr[0] = 42.0f;                           // Write directly to IOSurface
ane_unlock_input(k, 0);

// Cleanup
ane_free(k);
ane_weight_free(&w);
```

### Architecture Decisions

| Decision | Reason |
|----------|--------|
| C API (not ObjC/Swift) | Maximum portability, usable from any language |
| Cached selectors | Zero overhead at runtime for method calls |
| IOSurface for I/O | Zero-copy between CPU and ANE (no staging buffer) |
| Conv 1x1 instead of matmul | 3x faster on ANE (Orion paper confirms) |
| QoS=9 (Background) | 42% faster than Default, lowest system load |
| FP32 input/output, FP16 internal | ANE computes in FP16, but CPU needs FP32 for gradients |

---

## Layer 3: Training Pipeline

> **Note:** Layer 3 is based on the [maderix/ANE](https://github.com/maderix/ANE) repo and is not included in this repository. This chapter documents our findings and adaptations.

### What we use
The maderix/ANE repo (`training/training_dynamic/`) — a working Transformer trainer that runs directly on the ANE.

### What we adapted
- Dashboard TFLOPS: 15.8 (M4) → 12.79 peak / 2.15 real training (M3 Pro)
- Created synthetic test data (500K tokens)
- Pulled tokenizer via git-lfs
- Verified: 50 steps stable, no NaN, no crashes

### How training works (simplified)
```
1. Generate MIL programs (10 kernel types per layer)
2. Compile once on ANE (520ms for all 10)
3. Per training step:
   a. Pack weights into IOSurface spatial dimension (CPU)
   b. Forward pass: ANE evaluates kernel (22ms)
   c. Attention + Softmax + RoPE on CPU (because ANE can't do causal masking)
   d. Backward pass: ANE for dx gradients (30ms), CPU for dW via CBLAS
   e. Adam optimizer update on CPU
   f. Weights are updated directly in IOSurface (no recompile!)
```

### Key Innovation: Dynamic Spatial Packing
Weights are packed **alongside** the activations in the spatial dimension of the input tensor:
```
IOSurface: [1, DIM, 1, SEQ + WEIGHT_COLS]
                        ↑Data  ↑Weights

Inside ANE kernel:
  data    = slice(input, [0,0,0,0], [1,DIM,1,SEQ])
  weights = slice(input, [0,0,0,SEQ], [1,DIM,1,OC])
  output  = matmul(data, weights)
```
→ **One compile for all weight updates.** Without this trick, you would have to recompile all kernels after every Adam step (119-compile limit!).

### Performance on M3 Pro
| Model | Parameters | ms/step | Compile time |
|-------|-----------|---------|--------------|
| Stories110M (32K vocab) | 109.5M | 183ms | 520ms |
| Stories110M (124 vocab, compacted) | 109.5M | 91ms | 520ms |
| Qwen3-0.6B | 596M | ~412ms (estimated) | ~800ms |

---

## Layer 4: Solution (Personal AI)

> **Note:** Layer 4 is a separate project and is not included in this repository.

The solution builds on ALL three underlying layers:
- **Layer 1** (Research) provided the knowledge about QoS, vocab compaction, constraints
- **Layer 2** (libane) provides the stable API for future ANE usage
- **Layer 3** (Training) provides the working trainer

---

## File Structure (this repository)

```
ANE-Training/
│
├── ARCHITECTURE.md              ← This document (platform architecture)
├── RESEARCH_ANE_COMPLETE.md     ← Full research documentation
├── SUMMARY_TECHNICAL.md         ← Technical summary
├── SUMMARY_SIMPLE.md            ← Simple summary
├── LICENSE                      ← MIT
├── install.sh                   ← One-liner installer
│
├── examples/                    ← Runnable demos
│   ├── demo_train.c             ← ANE Training Demo (make demo)
│   ├── bench.c                  ← Auto-Benchmark (make bench)
│   ├── generate.c               ← Text Generation (make generate)
│   ├── explore.m                ← ANE Explorer (make explore)
│   └── Makefile
│
└── libane/                      ← Our C API
    ├── ane.h                    ← Stable API interface
    ├── ane.m                    ← Implementation with version detection
    ├── test_ane.c               ← Test suite
    ├── README.md                ← API documentation
    └── Makefile
```

### External References (not in this repo)
- **[maderix/ANE](https://github.com/maderix/ANE)** — Layer 3: Training pipeline with Dynamic Spatial Packing

---

## Chronology

| When | What | Why | Result |
|------|------|-----|--------|
| Phase A | API tests & probing | Learn what the ANE hardware can do | 35 classes, QoS levels, h15g identity |
| Phase B | Built libane | Stable, version-safe API | ane.h/ane.m, 3/3 tests passed |
| Phase C | Verified training | Proof that it works on M3 Pro | 91-183ms/step, 50 steps stable |
| Phase D | Built Personal AI | Concrete application | Collect→Tokenize→Train→Query pipeline (separate project) |

## What else can you build on this?

The platform (Layers 1-3) is the foundation. You can build on top of it:

- **Code assistant**: Learns your code patterns, suggests completions
- **Log analyzer**: Trained on system/app logs, detects anomalies
- **Document search**: Semantic search across all your files
- **Meeting notes**: Learns your vocabulary, creates better summaries
- **Local chatbot**: Fine-tuned on your writing style
- **Federated learning**: Multiple devices train locally, share only gradients

All 100% local, 100% private, on your Mac's ANE.
