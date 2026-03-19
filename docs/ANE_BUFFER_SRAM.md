# ANE Buffer & Memory Architecture — Deep Findings

> Research results from probing `_ANEBuffer`, `intermediateBufferHandle`, memory mapping structs,
> and SRAM management. Tested on M3 Pro (h15g), macOS 26.3.1, Build 25D2128.

---

## Key Findings Summary

| Finding | Impact |
|:---|:---|
| **128 IOSurface buffer slots per execution** | Hard limit on model I/O complexity |
| **8 internal priority queues** | More QoS granularity than the 5 public levels |
| **`memoryPoolId` on chaining requests** | Shared intermediate buffers between chained models |
| **`_ANECompilerAnalytics`** | Per-layer SRAM usage and spill data — optimization oracle |
| **No userspace SRAM control** | Firmware/driver manages SRAM entirely |

---

## 128-Slot IOSurface Buffer Limit

The `ANEMemoryMappingParamsStruct` — passed to the ANE driver for IOSurface mapping — has a fixed array of **128 buffer slots**:

```c
struct ANEBufferStruct {
    IOSurfaceRef surface;  // pointer to IOSurface
    uint32_t field1;       // likely: buffer type or flags
    int32_t  field2;       // likely: symbol index
    int32_t  field3;       // likely: offset
    uint32_t field4;       // likely: size or alignment
};

struct ANEMemoryMappingParamsStruct {
    ANEBufferStruct buffers[128];  // ← fixed 128-slot array
    uint64_t field1;               // likely: total mapped size
    uint32_t field2;               // likely: buffer count
    uint32_t field3;               // likely: flags
    uint64_t field4;               // likely: reserved
};
```

Discovered via type encoding of `_ANEProgramIOSurfacesMapper -prepareANEMemoryMappingParams:request:`.

### What This Means

- A single model execution can reference **at most 128 IOSurface buffers** (inputs + outputs + weights + intermediates combined)
- The `VirtANEModel` kernel-interface struct confirms this: `[32I]` for input slots, `[32Q]` for output slots, `[64I]` for buffer mappings — totaling the 128-slot capacity
- For training with Dynamic Spatial Packing: each kernel uses ~3 IOSurfaces (input, output, weights), so the practical limit is ~40 concurrent kernels per execution — more than enough
- This matches the hardware register file layout of the ANE

---

## 8 Internal Priority Queues

`_ANEClient` internally creates **8 dispatch queues** for ANE task submission:

```
com.apple.anef.p0
com.apple.anef.p1
com.apple.anef.p2
com.apple.anef.p3
com.apple.anef.p4
com.apple.anef.p5
com.apple.anef.p6
com.apple.anef.p7
```

### What This Means

The public API exposes 5 QoS levels:

| Public Name | Value | Internal Queue |
|:---|:---|:---|
| Background | 9 | p0 or p1 |
| Utility | 17 | p2 or p3 |
| Default | 21 | p3 or p4 |
| User Initiated | 25 | p5 or p6 |
| User Interactive | 33 | p6 or p7 |

There are potentially **3 additional internal QoS levels** not exposed through the standard API. The exact mapping between QoS values and internal queues is done by `_ANEQoSMapper` (not yet fully probed).

### Action Item

- Probe `_ANEQoSMapper` to discover the exact mapping
- Test QoS values between the known ones (e.g., 10-16, 18-20, 22-24, 26-32) to see if they map to different internal queues
- Test if values 0-8 (below Background) or 34+ (above User Interactive) access the remaining queues

---

## `memoryPoolId` on `_ANEChainingRequest`

`_ANEChainingRequest` has a `memoryPoolId` property (`NSNumber`). This is the **most promising memory management surface** discovered.

### Context

When chaining multiple ANE models (e.g., forward pass → backward pass), each model normally gets its own intermediate buffer allocation in DRAM. The `memoryPoolId` appears to allow chained models to **share a single memory pool** for intermediates.

### API Surface

```objc
// _ANEChainingRequest properties:
@property memoryPoolId   (NSNumber, read-write)
@property models         (NSArray of _ANEModel)
@property requests       (NSArray of _ANERequest)

// Used via _ANEClient:
- prepareChainingWithModel:options:chainingReq:qos:error:
```

### Potential Benefits

1. **Reduced DRAM usage** — Chained models share intermediate buffers instead of allocating separately
2. **Reduced dispatch overhead** — If chaining bypasses the per-kernel dispatch cost (~0.17ms), this could close the gap between single-kernel peak (11.64 TFLOPS) and stacked peak (12.79 TFLOPS)
3. **Pipeline parallelism** — Output of model A flows directly as input to model B without CPU round-trip

### Status

Being probed by the Chaining Agent (separate investigation). The `prepareChainingWithModel:` method exists on `_ANEClient` and `_ANEVirtualClient` — it's a real API, not a stub.

---

## `_ANECompilerAnalytics` — SRAM Spill Oracle

This class was discovered during the buffer probe. It contains analytics about how the compiler allocated resources:

### Known Sub-Structures

```
_AnalyticsProcedureInfo  — per-procedure (whole model) stats
_AnalyticsLayerInfo      — per-layer resource usage
_AnalyticsTaskInfo       — per-task (HW operation) details
```

### Why This Matters

The ANE compiler decides at compile time how to tile operations across the ~32MB SRAM. When a tensor doesn't fit, it "spills" to DRAM, causing a ~30% throughput drop. Currently we have **no visibility** into whether this happens or which layers cause it.

If `_ANECompilerAnalytics` exposes per-layer SRAM usage, we could:
1. **Identify bottleneck layers** — Which layers spill to DRAM?
2. **Optimize model architecture** — Reshape layers to fit SRAM
3. **Make informed tiling decisions** — Split large layers into SRAM-sized chunks
4. **Validate training configs** — Confirm tensors stay within 32MB budget

### Action Item

Probe `_ANECompilerAnalytics` fully — dump all methods, try to access it after compilation, check if per-layer SRAM/spill data is readable on consumer macOS.

---

## `_ANEBuffer` — Metadata Wrapper (Not SRAM Control)

Despite the name, `_ANEBuffer` is **not** a memory management primitive:

```objc
@interface _ANEBuffer : NSObject <NSSecureCoding>
@property (readonly) _ANEIOSurfaceObject *ioSurfaceObject;
@property (readonly) NSNumber *symbolIndex;  // tensor slot in model
@property (readonly) int64_t source;         // enum: input/output/intermediate
@end

// Only factory method:
+ bufferWithIOSurfaceObject:symbolIndex:source:
// alloc/init returns nil
```

It's a binding record that maps an IOSurface to a model's tensor slot. Used internally for IPC with the `aned` daemon (hence `NSSecureCoding` conformance).

---

## `intermediateBufferHandle` — Daemon-Managed

The `intermediateBufferHandle` property on `_ANEModel` and `_ANEInMemoryModel`:

- **Type:** `uint64_t` (not an object pointer)
- **Value after load:** `0` (zero) — assigned by the daemon during `loadModel:`
- **Setter exists** but setting arbitrary values has no effect on eval
- It's a **kernel-space opaque handle** to a DRAM region for intermediate activations

```
After compile+load:
  program = _ANEProgramForEvaluation {
    programHandle = 5209596762943
    intermediateBufferHandle = 0
    queueDepth = 127
  }
```

**Conclusion:** The daemon/driver allocates and manages intermediate buffers. We cannot provide our own, control size, or influence allocation strategy from userspace.

---

## Classes That Do NOT Exist

All of these were tested and returned `NOT FOUND`:

```
_ANEMemoryPool          _ANESRAMManager        _ANEBufferPool
_ANETileInfo            _ANECache              _ANEDMAInfo
_ANEIntermediateBuffer  _ANESharedMemory       _ANEDeviceMemory
_ANEAllocation          _ANEWeightsBuffer      _ANEProgramCache
_ANEBufferDescriptor    _ANEResourcePool       _ANEProgramHandle
_ANEModelHandle
```

Apple has not exposed any SRAM management APIs in userspace. The tiling and SRAM allocation is fully handled by the compiler + firmware.

---

## Optimization Levers (What We CAN Do)

| Lever | Method | Impact |
|:---|:---|:---|
| **Keep tensors < 32MB** | Architecture design | Prevents SRAM → DRAM spill (~30% drop) |
| **Use `_ANECompilerAnalytics`** | Post-compile analysis | Identify which layers spill |
| **Use chaining with `memoryPoolId`** | `_ANEChainingRequest` | Share DRAM intermediates, reduce dispatch |
| **Kernel fusion** | Stacked conv, fused ops | Amortize dispatch overhead |
| **`queueDepth` tuning** | `_ANEModel.queueDepth` | Default 127 — try lower for latency? |

---

*Last updated: 2026-03-18 | M3 Pro (h15g), macOS 26.3.1 (25D2128)*
*Source: `repo/training/test_buffer.m`*
