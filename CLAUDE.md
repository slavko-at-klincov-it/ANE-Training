# CLAUDE.md — Project Instructions for AI Assistants

## What This Project Is

Reverse-engineered Apple Neural Engine (ANE) training platform. We discovered 76+ private API classes
in AppleNeuralEngine.framework and built `libane` — a stable C API for compiling, loading, and
evaluating models directly on the ANE. Apple officially only allows inference via CoreML.

## Architecture

```
./ane                    CLI entry point (bash)
examples/                Runnable demos (bench.c, demo_train.c, generate.c, explore.m)
libane/                  Core C API (ane.h, ane.m)
training/                Training pipeline (train_pipeline.m, train_large_ane.m, headers)
docs/                    Research findings (ANE_MONITORING, CHAINING, COMPILER_OPTIONS, etc.)
assets/                  Model assets (tokenizer.bin)
```

## Build Commands

```bash
cd libane && make                    # Build libane
cd libane && make test               # Run libane tests
cd examples && make all              # Build all examples
cd training && make <target>          # Build specific training target
./ane                                # Interactive CLI (builds everything automatically)
./ane bench                          # Full benchmark
```

## Key Technical Facts

- **ANE has ~119 compile budget per process** — then silent failures. Solved by Dynamic Spatial Packing.
- **QoS Background (9) is fastest** — less OS scheduling interference, ~0 dispatch overhead.
- **Conv 1x1 is 3x faster than matmul** on ANE — express all matmuls as 1x1 conv.
- **SRAM is ~32MB** — tensors beyond this spill to DRAM with ~30% throughput drop.
- **ANE Silicon Peak: 12.79 TFLOPS** (128x stacked conv benchmark, M3 Pro).
- **Sustained Single-Kernel: 5.01 TFLOPS** (continuous eval, one kernel shape).
- **Real Training (Stories-110M): 2.15 TFLOPS** (80.9 ms/step, pipeline parallel).
- **Sequential Training: 1.87 TFLOPS** (93 ms/step, without pipeline).
- Peak numbers are benchmark-only — real training is ~6x lower due to per-layer dispatch, IOSurface I/O, and CPU gradients.
- **Thermal: always Nominal** — ANE never throttles under sustained compute load.

## What Does NOT Work (Don't Re-Investigate)

- `perfStatsMask` on `_ANEPerformanceStats` — returns nil on consumer macOS (Apple-internal only)
- `beginRealTimeTask` on `_ANEClient` — needs Apple entitlements, fails
- `kANEFKeepModelMemoryWiredKey: @YES` on load — ~18% slower for training, not beneficial
- `sessionHintWithModel:hint:` — all tested hint strings rejected on consumer macOS
- **ANE backward without loss scaling** — FP16 underflow zeros all gradients (see below)
- **CPU rmsnorm_bwd with w[i] on full expression** — old code had `dx = w[i] * rrms * (dy - x*dot)` which applies RMSNorm weight to the correction term. Correct: `dx = rrms * (w[i]*dy - x*dot)`. Bug is invisible at init (w=1.0) but corrupts gradients once RMSNorm weights train away from 1.0. Fixed in both `cpu_ops.h` and `stories_cpu_ops.h`. ANE MIL version (`ane_rmsnorm_bwd.h`) was already correct.
- **Activation explosion with res_alpha + Adam** — residual scaling `1/sqrt(2*NLAYERS)` combined with Adam's normalized steps (each ≈±lr) causes weights to grow unbounded. After 5000 steps, x_cur reaches [-800, 600+], attenuating gradients through RMSNorm to ~1e-3. **Fixed by removing res_alpha (standard residual) + GPT-2 style init (Wo, W2 scaled by 1/sqrt(NLAYERS)).**
- **ACCUM_STEPS < 100** — 86 kernels per batch vs ~119 compile limit = only 1 batch per exec()
- **Rapid exec() restart loops** — poisons ANE daemon system-wide, requires reboot
- **Small kernel dims (D<128)** — ANE error 0x1d (Program Inference error)
- No userspace SRAM control — firmware/driver manages it entirely
- No ANE frequency/voltage/temperature APIs — not exposed

## What DOES Work (Use These)

- **LOSS_SCALE=1024 for ANE backward** — must scale dlogits up before backward, unscale grads before Adam. Without this, FP16 underflow zeros all layer gradients silently.
- **RoPE in MIL** — working implementation exists in `training_dynamic/mil_dynamic.h` (lines 131-168), ready to port to `stories_mil.h`
- `ane_thermal_state()` — system thermal pressure (4 levels)
- `ane_device_info()` — arch, cores, board type, build
- `evaluateRealTimeWithModel:` — works without entitlements (alternative eval path)
- `doEvaluateDirectWithModel:` — works (possibly bypasses daemon dispatch)
- Dynamic Spatial Packing — weights as IOSurface input, compile once, train unlimited
- **Standard residual + GPT-2 init** — use `res_alpha=1.0` (no scaling) with output projections (Wo, W2) initialized at `1/sqrt(NLAYERS)`. This keeps activations stable (x stays [-3, 4]) and enables convergence. Loss drops from ln(V)=10.37 to ~9.94 on Tiny-ANE-15M after 1000 Adam updates.

## Coding Conventions

- libane public API: `ane.h` (C, stable, never changes for users)
- libane implementation: `ane.m` (Objective-C, uses objc_msgSend for private APIs)
- Research probes: `repo/training/test_*.m` (standalone Objective-C, each gets Makefile target)
- Benchmark: `examples/bench.c` (pure C, links against libane)
- CLI: `./ane` (bash, bilingual DE/EN with `t()` helper)
- Research docs: `docs/ANE_*.md` (detailed findings per topic)

## Language

User speaks German. Code comments and docs are English. CLI supports both (--lang=de/en).

## Research Findings Location

All deep RE findings are in `docs/`:
- `ANE_MONITORING.md` — Thermal, device info, what's measurable
- `ANE_BUFFER_SRAM.md` — Buffer architecture, 128 slots, SRAM limits
- `ANE_CHAINING.md` — Chaining = loopback, dispatch overhead analysis
- `ANE_COMPILER_OPTIONS.md` — Runtime option keys, KeepMemoryWired rejection
- `ANE_QOS_EVENTS.md` — 76 classes, QoS mapping, SharedEvents, model attributes
