# ANE Compiler Analytics — Deep Findings

> Research results from probing `_ANECompilerAnalytics` and its sub-structures for
> per-layer SRAM usage and spill data.
> Tested on M3 Pro (h15g), macOS 26.3.1, Build 25D2128.

---

## Key Findings Summary

| Finding | Impact |
|:---|:---|
| **17 analytics metric types discovered** | DRAM/L2 traffic, NE/DRAM domain time, spill flags |
| **Per-layer cost data exists** | `_AnalyticsLayerInfo` has `float value` per layer |
| **`ViolatesMaxLatency` flag** | Would indicate SRAM spills — exactly what we need |
| **Buffer is daemon-side only** | `aned` generates it but doesn't return it to client |
| **`EspressoProfilingANEcompilerAnalytics`** | May write analytics to disk when profiling enabled |

---

## Analytics Metric Types

`_ANECompilerAnalytics +stringForAnalyticsType:` reveals 17 named metrics:

| ID | Name | Purpose |
|:---|:---|:---|
| 0 | Unsupported | |
| 1 | Start Time Stamp | Compilation start |
| 2 | End Time Stamp | Compilation end |
| 3 | **DRAMTraffic** | DRAM bandwidth consumed |
| 4 | **L2Traffic** | L2 cache traffic |
| 5 | **Static Analytics - NE domain time** | Neural Engine compute time |
| 6 | **Static Analytics - L2 domain time** | L2 cache domain time |
| 7 | **Static Analytics - DRAM domain time** | DRAM access time |
| 8 | **Static Analytics - Total elapsed time** | End-to-end latency |
| 9 | **Static Analytics - Procedure Latency** | Per-procedure latency |
| 10 | TaskId | Task identifier |
| 11 | **ViolatesMaxLatency** | **Spill/latency violation flag** |
| 12 | **Static Analytics - NE Frequency** | ANE clock frequency |
| 13 | **Static Analytics - L2 Frequency** | L2 cache frequency |
| 14 | **Static Analytics - DRAM Bandwidth** | DRAM bandwidth spec |
| 15 | Ident String | Model identifier |
| 16 | **MAX TD Latency** | Max task dispatch latency |

## Class Hierarchy

```
_ANECompilerAnalytics
  ├─ analyticsBuffer (NSData)          — raw binary from compiler
  ├─ populateAnalytics                 — parses buffer → procedureAnalytics
  └─ procedureAnalytics                — NSArray<_ANEAnalyticsProcedure>
      └─ _ANEAnalyticsProcedure
          ├─ identifier (NSString)
          ├─ procedureMetrics (NSDictionary)
          └─ groupInfo                 — NSArray<_ANEAnalyticsGroup>
              └─ _ANEAnalyticsGroup
                  ├─ groupID (NSNumber)
                  ├─ layerInfo         — NSArray<_ANEAnalyticsLayer>
                  │   └─ _ANEAnalyticsLayer
                  │       ├─ layerName (NSString)
                  │       └─ weight (float)   ← PER-LAYER COST
                  └─ taskInfo          — NSArray<_ANEAnalyticsTask>
                      └─ _ANEAnalyticsTask
                          └─ metrics (NSDictionary)
```

## Struct Layouts (from type encodings)

```c
// Per-layer info: 132 bytes
struct _AnalyticsLayerInfo {
    char name[64];     // layer name
    char type[64];     // layer type
    float value;       // cost/weight metric
};

// Per-procedure info: 48 bytes
struct _AnalyticsProcedureInfo {
    uint32_t field0;   // procedure index?
    uint32_t field1;   // layer count?
    uint32_t field2;   // group count?
    uint32_t field3;   // task count?
    uint32_t field4;
    uint64_t field5;
    uint32_t field6;
    uint64_t field7;
};

// Per-task info: 16 bytes
struct _AnalyticsTaskInfo {
    uint32_t taskId;
    uint64_t metric;   // likely latency or traffic
};

// Per-group info: 32 bytes
struct _AnalyticsGroupInfo {
    uint32_t groupId;
    uint64_t field1;
    uint32_t field2;
    uint64_t field3;
};
```

## The Problem: Buffer is Daemon-Side

`_ANECompilerAnalytics` is a **parser**, not a producer. It takes an `NSData` buffer via
`+objectWithBuffer:` and parses it into the hierarchy above.

The buffer is generated inside `aned` daemon during `compileModel:sandboxExtension:options:qos:withReply:`
on `_ANEDaemonConnection`. The daemon's XPC reply does **not** pass the analytics buffer back to
the client in the normal compilation flow.

## XPC Reply Interception Results (CONFIRMED)

### Discovered XPC Protocol

`_ANEDaemonProtocol` (required instance methods):
```
compileModel:sandboxExtension:options:qos:withReply:
compiledModelExistsFor:withReply:
loadModel:sandboxExtension:options:qos:withReply:
loadModelNewInstance:options:modelInstParams:qos:withReply:
prepareChainingWithModel:options:chainingReq:qos:withReply:
purgeCompiledModel:withReply:
unloadModel:options:qos:withReply:
reportTelemetryToPPS:playload:
```

### Compile Reply Block Signature

From block descriptor introspection:
```
v36@?0B8@"NSDictionary"12@"NSString"20@"NSError"28
```
Decoded: `^(BOOL success, NSDictionary *result, NSString *info, NSError *error)`

**Compile result dictionary (4 keys):**
| Key | Type | Value |
|:---|:---|:---|
| `ErrorList` | NSArray | Empty on success |
| `ModelMaxDramUsage` | NSNumber | 278980 (for 256x256 conv) |
| `CompiledInputSourceFileName` | NSString | Path to model.mil |
| `NetworkStatusList` | NSArray | Per-procedure I/O descriptors (see below) |

**No analytics buffer is present.** The compiler analytics buffer never leaves the daemon process.

### Load Reply Block Signature

```
v56@?0B8@"NSDictionary"12Q20Q28c36@"NSString"40@"NSError"48
```
Decoded: `^(BOOL success, NSDictionary *result, uint64_t programHandle, uint64_t intermediateHandle, char queueDepth, NSString *info, NSError *error)`

**Load result dictionary (2 keys):**
| Key | Type | Value |
|:---|:---|:---|
| `NetworkStatusList` | NSArray | Per-procedure I/O descriptors |
| `ANEFModelDescription` | NSDictionary | Input/output symbols, alignment, procedure names |

**Load reply extra fields:**
| Field | Type | Example | Meaning |
|:---|:---|:---|:---|
| `programHandle` | uint64 | `0x4d697c3e1f2` | ANE program slot identifier |
| `intermediateHandle` | uint64 | `0` | **SRAM spill buffer handle** (0 = no spill) |
| `queueDepth` | char | `127` | Max queue depth for this program |

### NetworkStatusList Per-Procedure Entry

Each procedure (e.g., "main") contains:
```
Name: "main"
LiveInputList:  [{Symbol, Name, Type, Channels, Width, Height, Depth,
                   Batches, Interleave, PlaneCount,
                   BatchStride, PlaneStride, RowStride, DepthStride}]
LiveOutputList: [same fields]
```

### Key Insight: intermediateBufferHandle IS the Spill Signal

The load reply's `intermediateHandle` serves the same purpose as the analytics
`ViolatesMaxLatency` flag. When non-zero, it indicates the compiler allocated an
intermediate DRAM buffer because layer activations exceeded SRAM capacity.

`intermediateHandle = 0` for all single-conv models (256-2048 channels), confirming
they fit in SRAM. Multi-layer models may produce non-zero handles.

### _ANEClient Internal Structure

```
_ANEClient
  ._conn: _ANEDaemonConnection (main XPC connection)
  ._fastConn: _ANEDaemonConnection (fast/priority connection)
  ._virtualClient: _ANEVirtualClient (direct IOUserClient path)
  ._connections: NSMutableDictionary (model -> connection map)
  ._connectionsUsedForLoadingModels: NSMutableDictionary
  ._priorityQ: NSArray
  ._lock: os_unfair_lock
```

`_ANEDaemonConnection` wraps `NSXPCConnection` to `com.apple.appleneuralengine` (aned, pid varies).

## Remaining Options for Analytics Buffer

### Option 1: Intercept Inside aned Daemon (DYLD_INSERT)
The analytics buffer is generated inside the daemon process. Use `DYLD_INSERT_LIBRARIES`
to inject into `/usr/libexec/aned` and intercept from the daemon side.
**Blocked by SIP** on standard macOS installs.

### Option 2: CoreML Profiling Environment
`EspressoProfilingANEcompilerAnalytics` has a `compiler_analytics_file_names` property (NSArray),
suggesting analytics are written to disk files when profiling is enabled.
Try: `COREML_PROFILING=1 ./ane bench` or similar environment variables.

### Option 3: _ANEVirtualClient Direct Path
`_ANEVirtualClient.compileModel:options:qos:error:` bypasses the daemon entirely and
calls the ANE IOUserClient directly. The `updatePerformanceStats:` method on
`_ANEVirtualClient` takes a `VMData*` struct and returns an NSObject (likely NSDictionary
or NSData with perf stats). This path may include analytics data that the daemon path filters out.

### Option 4: Construct Buffer Manually
Since we know the struct layouts (`_AnalyticsLayerInfo` = 132 bytes, etc.), we could
construct a synthetic analytics buffer if we can obtain the raw values through other means.

---

*Last updated: 2026-03-18 | M3 Pro (h15g), macOS 26.3.1 (25D2128)*
*Source: `repo/training/test_compiler_analytics.m`, `repo/training/test_analytics_xpc.m`*
