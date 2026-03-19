# ANE Monitoring & Telemetry — Deep Findings

> Research results from probing Apple's private ANE APIs for hardware monitoring capabilities.
> Tested on M3 Pro (h15g), macOS 26.3.1, Build 25D2128.

---

## What We Can Measure

| Metric | Method | Sudo? | Accuracy | Integrated |
|:---|:---|:---|:---|:---|
| **Thermal State** | `NSProcessInfo.thermalState` | No | 4 levels | `ane_thermal_state()` in libane |
| **Execution Latency** | `mach_absolute_time` / `clock_gettime` | No | Sub-microsecond | Benchmark, all tests |
| **TFLOPS** | Latency + GFLOP calculation | No | ~1% variance | Benchmark (peak, not training) |
| **ANE Power (mW)** | `sudo powermetrics --samplers ane_power` | Yes | ~1s interval | Dashboard only |
| **CPU/GPU Power (mW)** | `sudo powermetrics --samplers cpu_power,gpu_power` | Yes | ~1s interval | Dashboard only |
| **Compile Count** | Internal tracking | No | Exact | `ane_compile_count()` |
| **Device Identity** | `_ANEDeviceInfo` class methods | No | Exact | `ane_device_info()` |

## What We Cannot Measure

| Metric | Why Not | Workaround |
|:---|:---|:---|
| **ANE Frequency** | Not exposed in any userspace API | None — Apple controls DVFS internally |
| **ANE Voltage** | No SMC/IOKit path for ANE power rail | `powermetrics` shows total ANE power draw, not voltage |
| **ANE Temperature** | No per-block thermal sensor API | `NSProcessInfo.thermalState` gives system-level thermal pressure |
| **ANE Core Utilization %** | Not tracked by any public or private API | Infer from latency vs theoretical peak |
| **Hardware Perf Counters** | `perfStatsMask` works but counters return `nil` on consumer builds | Only available on Apple-internal macOS builds |
| **Real-Time Task Mode** | `beginRealTimeTask` requires entitlements | Only Apple-signed apps can use this |

---

## _ANEDeviceInfo — Full Class Dump

All class methods available on `_ANEDeviceInfo`:

```
+ hasANE               → BOOL    (YES)
+ numANEs              → uint    (1)
+ numANECores          → uint    (16)
+ aneArchitectureType  → NSString ("h15g")
+ aneSubType           → NSString ("h15")
+ aneSubTypeVariant    → NSString ("g")
+ aneSubTypeProductVariant → NSString ("")
+ aneBoardType         → int64   (192)
+ productName          → NSString ("macOS")
+ buildVersion         → NSString ("25D2128")
+ bootArgs             → NSString ("" — empty on consumer)
+ isExcessivePowerDrainWhenIdle → BOOL (NO)
+ isVirtualMachine     → BOOL    (NO)
+ isInternalBuild      → BOOL    (NO)
+ precompiledModelChecksDisabled → BOOL (NO)
+ isBoolBootArgSetTrue: → BOOL
+ isBootArgPresent:     → BOOL
```

**Key insight:** `isInternalBuild: NO` explains why performance counters are unavailable. Apple gates hardware telemetry behind internal builds.

### Architecture Naming Convention

| Arch | Chip | SubType | Variant |
|:---|:---|:---|:---|
| h13g | M1 / M1 Pro | h13 | g |
| h13p | M1 Max | h13 | p |
| h14g | M2 / M2 Pro | h14 | g |
| h14p | M2 Max | h14 | p |
| h15g | M3 / M3 Pro | h15 | g |
| h15p | M3 Max | h15 | p |
| h16g | M4 / M4 Pro | h16 | g |
| h16p | M4 Max | h16 | p |

Pattern: `h{generation}{variant}` — `g` = base/Pro, `p` = Max. Ultra = 2x base die.

---

## _ANEPerformanceStats — Deep Probe

### Available Methods

```
Class methods:
  + statsWithHardwareExecutionNS:                          — create from hw time
  + statsWithReconstructed:hardwareExecutionNS:aneStatsRawData: — create from raw data
  + statsWithRequestPerformanceBuffer:statsBufferSize:      — create from perf buffer
  + driverMaskForANEFMask:                                  — convert mask format

Instance properties:
  @property hwExecutionTime     (uint64, readonly)  — hardware execution nanoseconds
  @property perfCounterData     (NSData, readonly)  — raw performance counter bytes
  @property pStatsRawData       (NSData, readonly)  — raw stats data

Instance methods:
  - performanceCounters         — parsed counter dict
  - stringForPerfCounter:       — counter name by index
  - emitPerfcounterSignpostsWithModelStringID: — os_signpost integration
```

### Test Results

Setting `perfStatsMask` on `_ANEInMemoryModel` before evaluation:

| Mask Value | Mask Readback | perfStats after eval |
|:---|:---|:---|
| 0x00000000 | 0x0 | `nil` |
| 0x00000001 | 0x1 | `nil` |
| 0x000000FF | 0xFF | `nil` |
| 0x0000FFFF | 0xFFFF | `nil` |
| 0xFFFFFFFF | 0xFFFFFFFF | `nil` |

**Conclusion:** The mask is accepted and stored on the model object, but the ANE driver on consumer macOS does not populate the performance stats buffer. The `_ANEVirtualClient` method `updatePerformanceStats:performanceStatsLength:perfStatsRawIOSurfaceRef:performanceStatsRawLength:hwExecutionTime:` is what fills these — it requires the driver to write perf data into IOSurfaces, which only happens on internal builds.

---

## Real-Time Task Mode

```objc
_ANEClient *client = [_ANEClient sharedConnection];
BOOL ok = [client beginRealTimeTask];  // → FAILED
```

`beginRealTimeTask` and `endRealTimeTask` exist on `_ANEClient` but require special entitlements (likely `com.apple.ane.realtime` or similar). Apple-signed apps (Siri, Camera, etc.) likely use this for latency-critical inference.

Also available but untested:
- `evaluateRealTimeWithModel:options:request:error:` — eval variant for RT mode
- `loadRealTimeModel:options:qos:error:` — load variant for RT mode
- `unloadRealTimeModel:options:qos:error:` — unload variant

---

## Thermal Monitoring

### NSProcessInfoThermalState

| State | Value | Meaning |
|:---|:---|:---|
| Nominal | 0 | Cool — full speed |
| Fair | 1 | Warm — performance still OK |
| Serious | 2 | Hot — system may throttle |
| Critical | 3 | Throttling active |

### Sustained Load Test Results (M3 Pro)

10-second sustained benchmark with `768x768 sp256` kernel:

- **93,000 evaluations** in 10 seconds
- **Thermal state: Nominal (cool) throughout** — never changed
- Average: 2.81 TFLOPS peak benchmark, 0.108 ms/eval (single kernel, not training)
- One dip to 1.88 TFLOPS at ~2.5s (OS scheduling, not thermal)

**Conclusion:** The ANE is thermally efficient. Even sustained full-speed operation does not trigger thermal pressure on M3 Pro. The MacBook fan did not activate. This suggests the ANE's power envelope is well within the cooling solution's capacity.

---

## powermetrics (sudo required)

The only way to measure actual ANE power draw in watts:

```bash
sudo powermetrics --samplers ane_power,cpu_power,gpu_power -i 1000
```

Output format:
```
ANE Power: 142 mW
CPU Power: 2341 mW
GPU Power: 12 mW
```

Already integrated in `repo/training/dashboard.py` for the TUI training dashboard. Not integrated in the CLI benchmark (would require sudo).

### Typical ANE Power Draw (M3 Pro)

| State | ANE Power |
|:---|:---|
| Idle | ~0 mW |
| Single kernel eval | ~50–150 mW |
| Sustained peak | ~200–400 mW |
| Full stacked benchmark | ~400–800 mW |

For comparison: CPU draws 2–8W, GPU draws 1–15W during equivalent compute.

---

## What This Means for Tweaking

Since we can't access frequency, voltage, or per-core utilization, the optimization levers are:

1. **Kernel Shape** — Larger spatial dims amortize dispatch overhead. Sweet spot: `768x2048 sp256` (single kernel) or `128x stacked` (peak throughput). Note: real training throughput is ~6x lower than peak due to per-layer dispatch, IOSurface I/O, and CPU gradients
2. **Dispatch Minimization** — The ~0.17ms dispatch floor is the primary bottleneck for small kernels. Fuse operations into fewer, larger kernels
3. **QoS Background (9)** — Consistently 42% faster than Default (21). Less OS scheduling interference
4. **SRAM Budget** — Keep working sets under 32MB. Beyond that, performance drops ~30% as data spills to DRAM
5. **Thermal Monitoring** — Use `ane_thermal_state()` to detect throttling during long training runs and adapt (reduce batch size, insert pauses)

---

## libane API

```c
#include "ane.h"

// Thermal monitoring (no ane_init() required)
ANEThermalState state = ane_thermal_state();
printf("Thermal: %s\n", ane_thermal_state_str(state));

// Device info (after ane_init())
ANEDeviceInfo info = ane_device_info();
printf("Chip: %s (%s), %d cores, board %d\n",
    info.arch, info.product, info.num_cores, info.board_type);
```

---

*Last updated: 2026-03-18 | M3 Pro (h15g), macOS 26.3.1 (25D2128)*
