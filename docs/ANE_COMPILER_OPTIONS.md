# ANE Compiler & Runtime Options — Deep Findings

> Research results from probing `compileWithQoS:options:`, `loadWithQoS:options:`,
> `_ANEStrings`, and the ANECompiler framework.
> Tested on M3 Pro (h15g), macOS 26.3.1, Build 25D2128.

---

## Key Findings Summary

| Finding | Impact |
|:---|:---|
| **`KeepMemoryWired=YES` on load** | **Not beneficial** — adds load overhead, ~18% slower in benchmarks |
| **No hidden compiler optimization flags** | Compiler code generation is fixed per HW target |
| **8 runtime option keys discovered** | Scheduling, memory, power hints to daemon |
| **`net_options.plist`** default compiler config | Can be overridden per-model |
| **Entitlement-gated power saving** | `aggressivePowerSaving` requires special entitlement |

---

## Discovered Runtime Option Keys

These keys are accepted in the `options:` dict of `compileWithQoS:`, `loadWithQoS:`, and `evaluateWithQoS:`.
They are **runtime hints** to the ANE daemon, not compiler code-generation flags.

### Performance-Relevant

| Key | Purpose | Values | Effect |
|:---|:---|:---|:---|
| **`kANEFKeepModelMemoryWiredKey`** | Pin model memory in RAM | `@YES`/`@NO` | **Not beneficial** — adds ~18% load overhead, may help for single long-lived models |
| `kANEFDisableIOFencesUseSharedEventsKey` | Use shared events instead of IO fences | `@YES`/`@NO` | ~12-28% faster combined with Wired |
| `kANEFEnableLateLatchKey` | Deferred execution scheduling | `@YES`/`@NO` | No consistent effect measured |
| `kANEFEnablePowerSavingKey` | Power saving mode | `@YES`/`@NO` | No effect for small kernels |
| `kANEFEnableFWToFWSignal` | Firmware-to-firmware signaling | `@YES`/`@NO` | For chaining workloads |
| `kANEFSkipPreparePhaseKey` | Skip prepare phase during compile | `@YES`/`@NO` | Not measured |
| `kANEFPerformanceStatsMask` | Enable hardware perf counters | `@(uint)` | Returns nil on consumer builds |
| `ANEClientEnergyEfficientWorkload` | Energy-efficient workload hint | `@YES`/`@NO` | Not measured |

### Model Identity / Caching

| Key | Purpose |
|:---|:---|
| `kANEFModelType` | Model format identifier |
| `kANEFIsInMemoryModelTypeKey` | In-memory model hash |
| `kANEFInMemoryModelIsCachedKey` | Whether compiled model is cached |
| `kANEFModelInstanceParameters` | NSCoded instance parameters |
| `kANEFCompilerOptionsFilenameKey` | Custom compiler options plist filename |
| `kANEFIntermediateBufferHandleKey` | Intermediate buffer handle |
| `kANEFMemoryPoolIDKey` | Memory pool ID (for chaining) |
| `kANEFconstantSurfaceID` | Constant surface ID |

### Model Type Constants

```
kANEFModelMIL           — MIL text format (what we use)
kANEFModelMLIR          — MLIR intermediate
kANEFModelPreCompiled   — Pre-compiled HWX binary
kANEFModelANECIR        — ANECIR format
kANEFModelCoreML        — CoreML model
kANEFModelLLIRBundle    — LLIR bundle
```

---

## Recommended Configuration for Training

```objc
// On loadWithQoS:options:error: — empty dict is optimal
NSDictionary *loadOpts = @{};

// On evaluateWithQoS:options:request:error:
NSDictionary *evalOpts = @{};  // no eval-time options needed
```

### KeepMemoryWired — Not Recommended

Despite initial micro-benchmark results suggesting 15-30% improvement, controlled A/B testing
with the full benchmark showed `kANEFKeepModelMemoryWiredKey: @YES` is **~18% slower**:

| Config | Run 1 | Run 2 | Run 3 | Avg Peak TFLOPS |
|:---|:---|:---|:---|:---|
| Without Wired (default) | 3.96 | 4.28 | 4.86 | **4.37** |
| With Wired | 3.22 | 3.71 | 3.81 | **3.58** |

<sub>Note: These are single-kernel peak benchmark TFLOPS, not training throughput. Real training achieves ~2.15 TFLOPS.</sub>

The wiring overhead during `loadWithQoS:` is not amortized for training workloads that compile
multiple kernels. It may help for a single model loaded once and evaluated millions of times
(e.g., production inference), but not for benchmarks or training pipelines.

---

## How Options Are Merged

`compilerOptionsWithOptions:isCompiledModelCached:` reveals the merge process:

```
Base options (auto-set):
{
    kANEFInMemoryModelIsCachedKey = 0;
    kANEFIsInMemoryModelTypeKey = "<model_hash>";
    kANEFModelType = kANEFModelMIL;
}

Your options get merged on top.
Unknown keys are silently ignored (not present in merged output).
```

---

## Compiler Configuration Files

| File | Purpose |
|:---|:---|
| `net_options.plist` | Default compiler options (looked for alongside model) |
| `compiler_options.plist` | ANECIR compiler options |

The `compilerOptionsFileName` property on `_ANEInMemoryModel` allows overriding the default filename.
However, these files configure the daemon-side compiler, not exposed to us.

---

## `optionsPlist` on Model Descriptor

```objc
+modelWithMILText:weights:optionsPlist:
```

The `optionsPlist` parameter expects `NSData` (serialized property list), not `NSDictionary`.
It gets hashed into the model identity — changing it forces recompilation.
This embeds compiler hints into the model identity, not for runtime tuning.

---

## Entitlement-Gated Features

| Entitlement | Feature |
|:---|:---|
| `com.apple.aned.private.aggressivePowerSaving.allow` | Aggressive power saving mode |
| `com.apple.ane.memoryUnwiringOptOutAccess.allow` | Opt out of memory unwiring |

These are not available to third-party apps.

---

## ANECompiler Framework

The `ANECompiler.framework` exports:
- `ANECCreateCompilerOptionDictionary` — requires model-specific input
- `ANECCreateCompilerOptionsCFString` — requires model-specific input
- `ANECCompile`, `ANECCompileJIT`, `ANECCompileOnline`, `ANECCompileOffline` — daemon-only

The actual compiler runs inside the `aned` daemon process. Client code cannot invoke it directly
or control code generation parameters beyond the options dict.

---

*Last updated: 2026-03-18 | M3 Pro (h15g), macOS 26.3.1 (25D2128)*
*Source: `repo/training/test_compiler_opts.m`*
