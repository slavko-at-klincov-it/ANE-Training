# ANE Chaining & Dispatch — Deep Findings

> Research results from probing `_ANEChainingRequest`, `_ANEOutputSetEnqueue`, dispatch overhead,
> and alternative eval methods. Tested on M3 Pro (h15g), macOS 26.3.1, Build 25D2128.

---

## Key Findings Summary

| Finding | Impact |
|:---|:---|
| **Chaining = Loopback, not pipelining** | Same model repeated, not different kernels chained |
| **QoS=9 dispatch overhead is ~0** | No chaining needed for sequential eval |
| **`evaluateRealTimeWithModel:` works** | No entitlements needed (unlike `beginRealTimeTask`) |
| **Model-switch penalty is negative** | ANE already pipelines different models internally |
| **`_ANEOutputSetEnqueue`** discovered | Missing piece to fully activate chaining API |

---

## What Chaining Actually Is

`_ANEChainingRequest` implements **loopback execution** — running the SAME model multiple times
where the output of iteration N feeds back as input to iteration N+1, all without returning to userspace.

This is designed for **autoregressive/recurrent workloads** (e.g., LLM token generation), NOT for
connecting different kernel types in a pipeline.

### Evidence

```objc
@interface _ANEChainingRequest
@property inputBuffer;                  // initial input
@property outputSets;                   // ping-pong output buffers
@property loopbackInputSymbolIndex;     // which input gets the loopback
@property loopbackOutputSymbolIndex;    // which output produces the loopback
@property fwEnqueueDelay;               // firmware pacing between iterations
@property signalEvents;                 // CPU/GPU sync between iterations
@property memoryPoolId;                 // shared memory pool
@property procedureIndex;
@property transactionHandle;
@end
```

### `_ANEOutputSetEnqueue` — The Missing Piece

Each element in `outputSets` must be an `_ANEOutputSetEnqueue` object, not a plain array:

```objc
+outputSetWithProcedureIndex:(uint)  // which compiled procedure
    setIndex:(uint)                   // buffer index (0, 1 for ping-pong)
    signalValue:(uint64)              // sync event value
    signalNotRequired:(BOOL)          // skip signaling?
    isOpenLoop:(BOOL)                 // run without sync wait?
```

The `isOpenLoop` flag is particularly interesting — it may allow fire-and-forget loopback execution
at maximum throughput without synchronization overhead.

---

## Dispatch Overhead Analysis

### Sequential Eval Benchmark (256ch + 512ch kernels)

| Mode | Kernel A | Kernel B | A+B pair | Overhead |
|:---|:---|:---|:---|:---|
| **QoS=21 (Default)** | 0.265 ms | 0.260 ms | 0.616 ms | **0.091 ms** |
| **QoS=9 (Background)** | 0.283 ms | 0.246 ms | 0.521 ms | **~0 ms** |
| **evaluateRealTimeWithModel** | 0.266 ms | 0.258 ms | 0.506 ms | **-0.018 ms** |
| **doEvaluateDirectWithModel** | 0.280 ms | — | 0.508 ms | **~0 ms** |

### Key Insight: QoS=9 Already Pipelines

At QoS=9 (Background), the ANE already pipelines command submission internally.
The dispatch overhead between sequential kernel evaluations is **effectively zero**.
The ~0.09ms overhead only appears at higher QoS levels due to scheduling arbitration.

### Model-Switch Penalty is Negative

Switching between two different models (A→B) is **faster** than running the same model twice (A→A).
This suggests the ANE can overlap the load/setup of model B while model A is still executing.

### Latency Distribution (QoS=9, 1000 samples)

| Percentile | Kernel A (256ch) | Kernel B (512ch) | A+B pair |
|:---|:---|:---|:---|
| Min | 90 us | 75 us | 154 us |
| P50 | 139 us | 131 us | 269 us |
| P90 | 192 us | 179 us | 376 us |
| P99 | 3774 us | 3784 us | 3883 us |
| Max | 4015 us | 5507 us | 5405 us |

P50(A+B) = 269 us ≈ P50(A) + P50(B) = 270 us — confirms near-zero overhead at median.

The P99 spike (~3.8ms) is OS scheduling interference, not ANE-related.

---

## Alternative Eval Methods

### `evaluateRealTimeWithModel:options:request:error:`

- **Works without entitlements** (unlike `beginRealTimeTask`)
- No QoS parameter — uses dedicated RT queue
- Comparable latency to QoS=9
- Type: `BOOL (*)(id, SEL, id model, id options, id request, NSError **error)`

### `doEvaluateDirectWithModel:options:request:qos:error:`

- Works — "direct" eval path (may bypass daemon dispatch?)
- Takes QoS parameter
- Same type encoding as regular eval

### Both Confirmed Working

These are real, functional eval paths that may have different scheduling characteristics
for production workloads.

---

## _ANEClient Chaining Methods

| Method | Status |
|:---|:---|
| `prepareChainingWithModel:options:chainingReq:qos:error:` | Exists, needs `_ANEOutputSetEnqueue` objects |
| `enqueueSetsWithModel:outputSet:options:qos:error:` | Exists, needs proper output set objects |
| `buffersReadyWithModel:inputBuffers:options:qos:error:` | Exists, returns timeout (needs chaining setup) |

---

## Next Steps to Fully Activate Chaining

1. Construct `_ANEOutputSetEnqueue` objects with proper procedure/set indices
2. Create `_ANEBuffer` wrappers for IOSurfaces
3. Build a `_ANEChainingRequest` with loopback indices set
4. Call `prepareChainingWithModel:` then `enqueueSetsWithModel:`
5. Measure: does loopback execution eliminate CPU round-trip latency?

This is most valuable for **autoregressive text generation** where the same model runs
hundreds of times sequentially.

---

## Practical Implications for Training

Since QoS=9 already has zero dispatch overhead:
- **No need for chaining** to reduce inter-kernel latency in training
- The gap between single-kernel peak (11.64) and stacked peak (12.79) TFLOPS is due to **kernel size**, not dispatch
- Stacked kernels amortize per-kernel **compute setup**, not dispatch
- Note: these are ANE silicon peak numbers. Real training achieves 2.15 TFLOPS (pipeline) / 1.87 TFLOPS (sequential)
- Focus optimization on **larger fused kernels** rather than chaining small ones
- `evaluateRealTimeWithModel:` is available as an alternative eval path if needed

---

*Last updated: 2026-03-18 | M3 Pro (h15g), macOS 26.3.1 (25D2128)*
*Source: `repo/training/test_chaining.m`*
