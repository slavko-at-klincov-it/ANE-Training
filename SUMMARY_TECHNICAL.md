# ANE Training — Technical Deep Dive

## What This Project Does

This project **reverse-engineers Apple's private Neural Engine (ANE) APIs** to enable **neural network training** on hardware Apple officially restricts to inference-only (via CoreML). It demonstrates a complete forward+backward training loop running transformer models (Stories110M and Qwen3-0.6B) with the compute-heavy linear algebra offloaded to the ANE.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                    train.m (main loop)               │
│  embedding → [layer×N] → final_norm → classifier    │
│              ↓ forward    ↑ backward                 │
├─────────┬───────────────────────────────┬───────────┤
│  CPU    │         ANE (via bridge)      │   CPU     │
│ RMSNorm │  QKV projection (1×1 conv)    │  RoPE     │
│ SiLU    │  Output projection            │  Softmax  │
│ Adam    │  FFN W1/W3 (parallel conv)    │  Attention│
│ Grads   │  FFN W2                       │  Masking  │
│         │  Classifier                   │           │
├─────────┴───────────────────────────────┴───────────┤
│              bridge/ane_bridge.m                      │
│  dlopen(AppleNeuralEngine.framework)                 │
│  NSClassFromString → _ANEInMemoryModel*              │
│  IOSurface zero-copy I/O                             │
└─────────────────────────────────────────────────────┘
```

## Private API Surface

Four private low-level classes resolved at runtime via `NSClassFromString()` (the public high-level training API, MLCompute with `MLCDevice.ane()`, was deprecated by Apple without replacement):

| Class | Purpose |
|-------|---------|
| `_ANEInMemoryModelDescriptor` | Creates model from MIL text + weight blobs |
| `_ANEInMemoryModel` | Compile → Load → Evaluate → Unload lifecycle |
| `_ANERequest` | Binds IOSurface I/O tensors to execution request |
| `_ANEIOSurfaceObject` | Wraps `IOSurfaceRef` for ANE consumption |

### Key Messages (objc_msgSend)

```
_ANEInMemoryModelDescriptor:
  +modelWithMILText:weights:optionsPlist:

_ANEInMemoryModel:
  +inMemoryModelWithDescriptor:
  -hexStringIdentifier
  -compileWithQoS:options:error:       (QoS=21)
  -loadWithQoS:options:error:          (QoS=21, retry after 100ms on fail)
  -evaluateWithQoS:options:request:error:
  -unloadWithQoS:error:
  -state

_ANEIOSurfaceObject:
  +objectWithIOSurface:

_ANERequest:
  +requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:
```

## MIL (Model Intermediate Language)

All ANE operations are expressed as MIL programs — Apple's internal IR for neural network graphs. Linear layers are mapped as **1×1 convolutions** (ANE's sweet spot):

```mil
program(1.3) {
  func main<ios18>(tensor<fp16, [1, 768, 1, 256]> x) {
    // Weight from binary blob
    tensor<fp16, [768, 768, 1, 1]> W = const()[
      val = BLOBFILE(path="@model_path/weights/wq.bin", offset=uint64(64))
    ]
    // Linear = 1x1 conv: [1,768,1,256] @ [768,768,1,1] → [1,768,1,256]
    tensor<fp16, [1, 768, 1, 256]> y = conv(weight=W, x=x);
  } -> (y);
}
```

### Tensor Layout

ANE uses channel-first `[1, C, 1, S]` layout (batch=1, channels=features, height=1, width=sequence). This matches IOSurface format and eliminates transpose overhead.

## Two Compilation Strategies

### 1. Static (Baked Weights) — `ane_mil_gen.h`

Weights embedded in BLOBFILE references at compile time. **Requires recompilation when weights change** (i.e., every training step). Limited by ~119 compile budget per process.

### 2. Dynamic (Spatial Packing) — `mil_dynamic.h`

**The key innovation.** Weights packed alongside activations in the spatial dimension of the input tensor:

```
Input IOSurface: [1, DIM, 1, SEQ + WEIGHT_COLS]
                              ↑act    ↑weights

Inside MIL kernel:
  activation = slice(input, [0,0,0,0], [1,DIM,1,SEQ])
  weight     = slice(input, [0,0,0,SEQ], [1,DIM,1,OC])
  output     = matmul(activation, weight)
```

**Advantage:** Single kernel compilation handles arbitrary weight values. Weight updates via CPU memcpy into the spatial region — no ANE recompile needed.

## Training Loop Detail

### Forward Pass (per layer)

1. **RMSNorm** (CPU, vectorized via vDSP)
2. **QKV projections** → ANE (3 parallel 1×1 convs or fused SDPA kernel)
3. **RoPE** (CPU, precomputed frequency table)
4. **Attention** (CPU — causal masking unsupported on ANE)
5. **Output projection** → ANE
6. **Residual add** (CPU)
7. **RMSNorm** (CPU)
8. **FFN W1/W3** → ANE (parallel gates)
9. **SiLU gate** (CPU)
10. **FFN W2** → ANE
11. **Residual add** (CPU)

### Backward Pass (reverse order, per layer)

- **dW gradients**: CPU via CBLAS (`sgemm`) — matrix multiply of cached activations × upstream gradients
- **dx gradients**: ANE (same kernel structure as forward, transposed weights)
- **Adam optimizer**: CPU (m/v moment tracking, bias correction)
- **Gradient clipping**: L2 norm with configurable threshold

### Weight Update Cycle (Static Path)

```
train_step() {
  forward()          // ANE eval existing kernels
  backward()         // CPU gradients + ANE dx
  adam_update()       // CPU weight modification
  recompile_kernels() // Compile new kernels with updated weights
                      // Only swap if ALL compile successfully
}
```

## IOSurface Zero-Copy I/O

```c
IOSurfaceRef surf = IOSurfaceCreate(@{
    kIOSurfaceWidth: @(bytes),
    kIOSurfaceHeight: @1,
    kIOSurfaceBytesPerElement: @1,
    kIOSurfacePixelFormat: @0   // raw bytes
});

// Write: lock → memcpy → unlock
IOSurfaceLock(surf, 0, NULL);
memcpy(IOSurfaceGetBaseAddress(surf), data, bytes);
IOSurfaceUnlock(surf, 0, NULL);
```

IOSurface provides DMA-capable shared memory between CPU and ANE without staging buffers.

## Weight Blob Binary Format

```
FP16:
  [0-63]    Global header (buf[0]=0x01, buf[4]=0x02)
  [64-127]  Chunk header (magic=0xDEADBEEF, size, data_offset=128)
  [128+]    _Float16 data

INT8 (quantized):
  [0-63]    Header (magic 0xEFBEADDE, buf[10]=0x08)
  [64+]     int8_t data
  Quantization: symmetric, scale = max(|w|) / 127
```

## Known Constraints

| Constraint | Workaround |
|-----------|------------|
| ~119 kernel compilations per process | `execl()` self-restart with checkpoint resume |
| No causal masking in ANE SDPA | Attention computed on CPU |
| FP16 gradient underflow | Global loss scaling |
| Multi-input ANE requests fail | Pack all inputs into single spatial tensor |
| ANE SRAM ~32MB | Performance cliff when exceeded; spatial packing helps locality |

## Performance (M3 Pro)

| Model | Params | ms/step | Kernels/layer | Architecture |
|-------|--------|---------|---------------|-------------|
| Stories110M | 109M | 80.9ms (pipeline) / 93ms (sequential) | 6 (MHA) | Llama2-style |
| Qwen3-0.6B | 596M | 412ms | 10 (GQA) | Grouped-Query Attention |

INT8 quantization (M4): **1.88x throughput**, peak **35.1 TOPS** (vs ~19 TOPS FP16). M3 Pro INT8 shows only ~1.0-1.14x improvement.

Inference vs CPU: ANE averages **722 GFLOPS** (FP16) vs CPU **1449 GFLOPS** (FP32/AMX). CPU wins at all shapes up to 1024x1024. ANE's advantage is power efficiency (~300 mW vs ~5W), not raw speed. See [docs/INFERENCE_BENCHMARK.md](docs/INFERENCE_BENCHMARK.md).

## File Map

```
repo/
├── api_exploration.m          # Runtime introspection of ANE private classes
├── inmem_bench.m              # Conv throughput: 6 sizes × 50 evals → TFLOPS
├── inmem_peak.m               # Sequential depth sweep → sustained peak measurement
├── sram_bench.m               # Working set sweep → SRAM capacity detection
├── ane_int8_bench.m           # INT8 vs FP16 throughput comparison
├── bridge/
│   ├── ane_bridge.h           # C API: init/compile/eval/free + weight blob builders
│   └── ane_bridge.m           # ObjC implementation: dlopen + objc_msgSend
├── training/
│   ├── train.m                # Simple training entry point
│   ├── model.h                # LayerWeights, LayerActs, LayerGrads structs
│   ├── forward.h              # model_forward() — ANE+CPU hybrid
│   ├── backward.h             # model_backward() — analytical gradients
│   ├── ane_mil_gen.h          # Static MIL generators (baked weights)
│   ├── stories_config.h       # DIM=768, HEADS=12, LAYERS=12, SEQ=256
│   ├── dashboard.py           # Real-time TUI: loss, ms/step, power, generation
│   └── training_dynamic/
│       ├── train.m            # Dynamic pipeline entry point
│       ├── mil_dynamic.h      # Spatial-packing MIL generators
│       ├── config.h           # Model-agnostic config macros
│       ├── cpu_ops.h          # RMSNorm, RoPE, softmax, Adam
│       ├── io.h               # Checkpoint I/O (binary, seekable)
│       └── models/
│           ├── stories110m.h  # MHA config (6 kernels/layer)
│           └── qwen3_06b.h    # GQA config (10 kernels/layer)
└── benchmarks/
    └── ANE_BENCHMARK_REPORT.md
```

## Checkpoint Format

```c
struct CkptHdr {
    int magic;          // 0x424C5A54 ("BLZT")
    int version;        // 2
    int step, total_steps;
    int n_layers, vocab_size, dim, hidden_dim, n_heads, seq_len;
    float lr, loss;
    double cum_compile, cum_train, cum_wall;
    int cum_steps, cum_batches, adam_t;
};
// Followed by: per-layer weights (fp32) + RMS scales + Adam m/v moments
```
