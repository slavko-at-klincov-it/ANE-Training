# ANE QoS, SharedEvents, Session Hints & Model Attributes — Deep Findings

> Research results from probing `_ANEQoSMapper`, `_ANESharedEvents`, session hints,
> and model attributes. Tested on M3 Pro (h15g), macOS 26.3.1, Build 25D2128.

---

## Key Findings Summary

| Finding | Impact |
|:---|:---|
| **76 ANE classes in runtime** (previously knew 35) | Much larger API surface than documented |
| **QoS mapping fully decoded** | 6 levels, 8 queues, queues 0/1/7 reserved |
| **SharedEvents = GPU↔ANE sync** | Uses `IOSurfaceSharedEvent` (same as Metal) |
| **Session hints fail on consumer** | All tested strings rejected silently |
| **queueDepth=127 default** | Changing has no measurable effect |

---

## QoS Mapping — Fully Decoded

### Named Levels

| Name | QoS Value | Program Priority | Queue Index |
|:---|:---|:---|:---|
| **RealTime** | 0 | 2 | 2 |
| **Background** | 9 | 6 | 6 |
| **Utility** | 17 | 5 | 5 |
| **Default** | 21 | 5 | 5 |
| **UserInitiated** | 25 | 4 | 4 |
| **UserInteractive** | 33 | 3 | 3 |

### Internal Queue Architecture

8 dispatch queues (`com.apple.anef.p0` through `p7`):

```
Queue 0 — RESERVED (unused)
Queue 1 — RESERVED (unused)
Queue 2 — RealTime (QoS=0, highest HW priority)
Queue 3 — UserInteractive (QoS=33)
Queue 4 — UserInitiated (QoS=25)
Queue 5 — Utility/Default (QoS=17, 21)
Queue 6 — Background (QoS=9, fastest for compute)
Queue 7 — RESERVED (unused)
```

### Priority Reverse Mapping

`qosForProgramPriority:` returns:
- Priority 0-3 → QoS 33 (UserInteractive)
- Priority 4 → QoS 25 (UserInitiated)
- Priority 5 → QoS 21 (Default)
- Priority 6 → QoS 9 (Background)
- All others → QoS 21 (Default)

### Why Background (Queue 6) is Fastest

Background has the **lowest scheduling priority** (priority 6), which means:
- Less preemption from the OS scheduler
- Longer uninterrupted execution windows
- No arbitration overhead with other QoS levels

For training workloads that want maximum throughput (not minimum latency), this is ideal.

### No Hidden QoS Levels

Any unrecognized QoS value defaults to priority 5 / queue 5 (same as Default).
Testing values between the known ones (10-16, 18-20, etc.) confirmed no additional mappings exist.

---

## `_ANESharedEvents` — GPU↔ANE Synchronization

### Purpose

SharedEvents is the ANE's synchronization primitive for **cross-accelerator coordination**,
using `IOSurfaceSharedEvent` (the same mechanism Metal uses for GPU sync).

This is how Apple chains GPU→ANE→GPU in CoreML pipelines without CPU round-trips.

### Class Structure

```objc
@interface _ANESharedEvents
@property (readonly) NSArray *signalEvents;  // _ANESharedSignalEvent
@property (readonly) NSArray *waitEvents;    // _ANESharedWaitEvent
+ sharedEventsWithSignalEvents:waitEvents:
@end

@interface _ANESharedSignalEvent
@property sharedEvent;       // IOSurfaceSharedEvent
@property value;             // uint64 — signal value
@property symbolIndex;       // uint32 — which tensor I/O symbol
@property agentMask;         // uint64 — which ANE cores participate
@property eventType;         // int64 — sync mode
+ signalEventWithValue:symbolIndex:eventType:sharedEvent:
@end

@interface _ANESharedWaitEvent
@property sharedEvent;       // IOSurfaceSharedEvent
@property value;             // uint64 — wait value
@property eventType;         // uint64 — sync mode
+ waitEventWithValue:sharedEvent:
+ waitEventWithValue:sharedEvent:eventType:
@end
```

### Key Details

- `symbolIndex` on signal events maps to specific tensor I/O symbols (input 0, output 0, etc.)
- `agentMask` likely selects which ANE cores participate in the signal
- `eventType` suggests multiple synchronization modes
- `alloc/init` returns nil — requires an `IOSurfaceSharedEvent` to construct
- Conforms to `NSSecureCoding` (used for IPC with `aned` daemon)

### Related Runtime Classes

- `GraphANESharedEventHandler` — MPSGraph integration
- `MPSGraphAneSessionDescriptor` — MPSGraph↔ANE session

### Practical Relevance

SharedEvents enable GPU↔ANE synchronization, which Apple uses internally in CoreML.

For this project: **not beneficial.** GPU benchmarking showed Metal is 3-8x slower than CPU/AMX
for training-relevant operations. Combining ANE + GPU adds synchronization overhead without
throughput gain. ANE's value is background training (GPU stays free), not GPU coordination.

---

## Session Hints — Apple-Internal Only

### API

```objc
// On _ANEClient:
- sessionHintWithModel:hint:options:report:error:
```

### Test Results

All 31 tested hint strings returned `ok=NO` with no error:
```
"preload", "prefetch", "warmup", "realtime", "cache", "pin",
"priority", "batch", "streaming", "low_latency", "high_throughput",
"compile", "optimize", "persistent", "shared", "exclusive",
"background", "foreground", "power_save", "performance",
"memory_pool", "dedicated", "turbo", "fast", "slow",
"default", "auto", "manual", "hybrid", "sync", "async"
```

- `nil` hint → explicit error: "Bad argument error"
- Method validates input (requires non-nil string + non-nil model)
- Valid hint strings are likely Apple-internal only
- May require model loaded via daemon XPC path (not in-memory)

---

## Model Attributes

### Default After Compile+Load

A rich dictionary with two keys:

**`ANEFModelDescription`:**
- Procedure map
- Input/output symbol arrays
- Alignment arrays

**`NetworkStatusList`:**
Per-procedure array with `LiveInputList` and `LiveOutputList`, each tensor containing:
```
BatchStride, Batches, Channels, Depth, DepthStride,
Height, Width, Interleave, PlaneCount, PlaneStride,
RowStride, Symbol, Type
```

### Model State Properties

| Property | Value | Notes |
|:---|:---|:---|
| `programHandle` | Large uint64 | Kernel address/handle |
| `intermediateBufferHandle` | 0 | Assigned by daemon |
| `queueDepth` | 127 | Default, type `char` (-128 to 127) |
| `state` | 3 | Loaded state |
| `perfStatsMask` | 0 | Perf counters disabled |

### `queueDepth`

- Default: 127 (effectively unbounded in-flight requests)
- Readable and writable via `setQueueDepth:`
- **No measurable performance difference** across values 0-127
- Likely controls max in-flight evaluation requests before blocking
- Value 0 may mean "synchronous only"

### `setModelAttributes:`

Accepts any `NSDictionary` and stores it. **Replaces** the entire dict (not merge).
Custom keys are accepted but have no effect on already-loaded programs.
Attributes are only consulted at load time by the daemon.

---

## 76 ANE Classes Discovered

Notable previously unknown classes beyond the original 35:

| Class | Purpose |
|:---|:---|
| `_ANEOutputSetEnqueue` | Output buffer wrapper for chaining |
| `_ANEInputBuffersReady` | Pre-staging input buffers |
| `_ANEModelInstanceParameters` | Multi-instance model loading |
| `_ANEDeviceController` | Direct HW control: start/stop, privileged connection |
| `_ANEProgramForEvaluation` | Compiled program wrapper |
| `_ANEProgramIOSurfacesMapper` | IOSurface↔program mapping |
| `_ANECloneHelper` | Model cloning |
| `_ANEDataReporter` | Telemetry/analytics |
| `_ANESandboxingHelper` | Sandbox policy |
| `_ANEDaemonConnection` | XPC to `aned` daemon |
| `_ANEModelToken` | Model identity tokens |
| `MPSGraphAneSessionDescriptor` | MPSGraph↔ANE integration |
| `GraphANESharedEventHandler` | MPSGraph shared events |

### Most Promising for Performance

1. **`_ANEDeviceController`** — `start`/`stop`, `sharedPrivilegedConnection`, `isPrivileged`
2. **`doEvaluateWithModel:...completionEvent:error:`** on `_ANEVirtualClient` — async eval with event signaling
3. **`loadModelNewInstance:options:modelInstParams:qos:error:`** — multi-instance model loading
4. **`buffersReadyWithModel:inputBuffers:options:qos:error:`** — pre-staging

---

*Last updated: 2026-03-18 | M3 Pro (h15g), macOS 26.3.1 (25D2128)*
*Source: `repo/training/test_session_hints.m`*
