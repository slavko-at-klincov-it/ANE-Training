# ANE Training Optimization — Experiment Log
**Date:** 2026-03-20
**Branch:** experiment/3h-optimize-session
**Baseline:** 77 ms/step, 2.26 TFLOPS, 0% overlap, loss flat at ~10.43

## What Works
- train_pipeline builds and runs (86 kernels, 5s compile)
- ANE dispatch at QoS 9 (Background) — confirmed fastest
- Dynamic Spatial Packing — compile once, unlimited training
- Steady-state 77 ms/step on Stories110M (12 layers, dim=768)
- **tiny_train learns** — loss 0.50 → 0.001 over 2000 steps (proof ANE training works)
- **Loss scaling (LOSS_SCALE=1024) fixes FP16 underflow** — gradients now non-zero
- **ffnBwd kernel works in isolation** — verified at full DIM=768/HIDDEN=2048/SEQ=256
- **Zero SRAM spills** — ANE tiles 1x1 convs automatically (204 configs tested)
- **RoPE MIL implementation exists** in training_dynamic/mil_dynamic.h (ready to port)

## What Doesn't Work
- ~~Loss not decreasing~~ → **FIXED by loss scaling** (gradients non-zero, slow convergence)
- Pipeline overlap = 0% (measurement bug + design limitation)
- Missing RoPE in SDPA kernels (limits model quality)
- **ANE compile budget is SYSTEM-WIDE** — rapid exec() loops poison the daemon, requires reboot
- Cannot reduce ACCUM_STEPS below ~100 (86 kernels per batch vs ~119 compile limit)

## Confirmed Bugs

### Bug 1: Pipeline Overlap Measurement Is Mathematically Zero (CONFIRMED)
**File:** training/train_pipeline.m, line ~900
**Root cause:** The formula `overlap = (step_bwd + step_fwd) - step_wall` is always 0 because
the three intervals are measured contiguously (no gap between bwd_end and fwd_start timestamps).
The math: `(t_bwd_end - t_start) + (t_fwd_end - t_bwd_end) - (t_fwd_end - t_start) = 0` always.

**But also:** Even with correct measurement, actual overlap is negligible because:
- dW tasks dispatched to serial `dw_q` complete before backward returns (plenty of synchronous work after dispatch)
- The big classifier backward sgemm (32000×768×256) runs synchronously on main thread, not dispatched
- `dw_q` is DISPATCH_QUEUE_SERIAL — only one sgemm at a time

**Fix needed:**
1. Fix measurement: check `dispatch_group_wait(dw_grp, DISPATCH_TIME_NOW)` at forward start
2. Make `dw_q` concurrent
3. Move classifier backward sgemm to `dw_q`
4. Move `dispatch_group_wait` from top of backward to just before gradient accumulation

### Bug 2: FFN Backward ANE Kernel Produces All-Zero Output (CONFIRMED — ROOT CAUSE)
**File:** training/stories_mil.h gen_ffn_bwd(), training/stories_io.h ane_eval()
**Root cause:** The `ffnBwd` kernel compiles and evaluates without error, but the output IOSurface
is entirely zeros. Verified by:
- Input is non-zero: |dffn_in|=0.094892, restored |h1|=259.37, |h3|=255.81
- Raw ioIn has data: dffn_sample=0.000578, h1_sample=0.642
- Raw ioOut is all zeros: first 1000 elements = 0.0
- ane_eval returns success (BOOL=YES, no NSError)
- This is a **silent failure** from the ANE

**Consequence:** ALL layer gradients are zero. Only embedding gradients are non-zero (from
classifier backward which runs on CPU). Loss stays flat because no weight in any transformer
layer is ever updated meaningfully.

**Gradient flow trace:**
```
classifier bwd (CPU, sgemm) → dy norm=0.094892 ✓
final rmsnorm bwd (CPU)     → dy norm=0.094892 ✓
For each layer L (11→0):
  ffnBwd (ANE)              → dx_ffn=0.000000 ✗ ← ALL ZEROS
  dW FFN (CPU, sgemm)       → all zero because dsilu/dh1/dh3 are zero
  rmsFFNBwd (ANE)           → dx2=0.000000 (input dx_ffn is zero)
  residual dx2 += dy        → dx2=0.094892 (only from residual passthrough)
  sdpaBwd1 (ANE)            → dv=?, dq/dk unclear
  dW QKV (CPU)              → zero because dq/dk/dv likely zero
  qkvBwd (ANE)              → dx_attn=0 (zero input)
  rmsAttBwd (ANE)           → dx_rms1=0
  dy = dx_rms1 + dx2        → dy=0.094892 (only residual)
```

**ROOT CAUSE CONFIRMED: FP16 underflow in ANE backward pass!**

The kernel WORKS in isolation with larger inputs (norm 8.77 output). In the pipeline, gradient
values are max ~0.000425 → after W2^T conv (weights ~0.036), intermediates drop to ~1.5e-5 →
after SiLU derivative chain, values underflow below fp16 precision → ANE flushes to zero.

**Proof:**
- 1000x gradient scaling → ffnBwd produces |dx|=1.703210 (non-zero!)
- Original gradient scale → ffnBwd produces |dx|=0.000000 (underflow)
- Isolated test with 0.1-scale inputs → works fine (norm 8.77)

**Fix needed: Loss scaling / gradient scaling (standard mixed-precision technique)**

## Under Investigation
- [x] Why is loss flat? → Bug 2 (FP16 underflow in backward, fixed by loss scaling)
- [x] Why is pipeline overlap 0ms? → Bug 1 (measurement + design limitation)
- [x] What hyperparameters work for tiny_train.m? → See Round 2
- [x] What kernel shapes give best ANE throughput? → See Round 6
- [x] Is ffnBwd failure a size/SRAM issue or MIL bug? → Neither! FP16 underflow (Round 4)
- [x] Missing RoPE in SDPA kernels → See Round 9 (working MIL impl exists in training_dynamic/)
- [x] Loss scaling validation — gradients non-zero! See Round 10
- [x] ANE compile budget is SYSTEM-WIDE — See Round 11 (critical discovery)

## Findings

### Round 1 — Pipeline Overlap Analysis (COMPLETE)
**Agent:** Diagnose zero pipeline overlap
**Duration:** ~2 min
**Result:** Two issues found:
1. Measurement bug — formula is algebraically always 0 regardless of real overlap
2. Design limitation — dW tasks complete within backward pass, leaving nothing to overlap with forward
**Next action:** Fix overlap after loss bug is diagnosed (no point optimizing speed if training doesn't learn)

### Round 2 — tiny_train Hyperparameter Sweep (COMPLETE)
**Agent:** Test tiny_train hyperparams
**Duration:** ~4 min
**Key result:** tiny_train DOES learn — loss 0.50 → 0.001 over 2000 steps. ANE training works!

**LR sweep (500 steps each):**
| LR  | Final Loss | Notes |
|-----|-----------|-------|
| 0.1 | 0.5007 | Way too slow |
| 0.5 | 0.4999 | Too slow at 500 steps |
| **1.0** | **0.1213** | **Best — stable, consistent** |
| 1.5 | 0.4210 | Unstable, bounces |
| 2.0 | 0.4919 | Wild oscillation |
| 5.0 | 0.5009 | Diverges, dead ReLUs |

**Long runs at LR=1.0:**
- 2000 steps: loss 0.0014, 7.5s wall
- 5000 steps: loss 0.0011, 21.5s wall (plateaus, likely FP16 limit)

**Training speed:** 0.6 ms/step compute, 3.7 ms/step wall (80-87% compile overhead)

**Critical insight:** tiny_train learns, train_pipeline doesn't → bug is in Stories-110M gradient/update code, NOT in ANE mechanism

**Files modified by agent:**
- training/Makefile — added tiny_train target
- training/tiny_train.m — added --lr= and --steps= CLI args

### Round 3 — Gradient Tracing (COMPLETE)
**Method:** Added diagnostic prints to train_pipeline.m backward pass
**Duration:** ~15 min hands-on debugging
**Key result:** FFN backward ANE kernel returns all zeros. This is THE root cause of flat loss.

**Evidence chain:**
1. Embed gradients non-zero (17.36 norm) — CPU classifier backward works
2. rms_final gradients non-zero (0.43 norm) — CPU rmsnorm backward works
3. ALL 12 layers: Wq, Wk, Wv, Wo, W1, W2, W3, rms_att, rms_ffn gradients = exactly 0
4. dy enters each layer at 0.094892, exits at 0.094892 — only residual passes through
5. ffnBwd kernel: input valid (non-zero), ane_eval succeeds, output = all zeros
6. This means: sdpaBwd also gets zero-like input (dx2 only from residual, no FFN contribution)
7. All dW accumulations sum to zero because activation gradients are zero

**Next action:** Testing ffnBwd in isolation with small dimensions to determine if it's SRAM/size issue

### Round 4 — FP16 Underflow Root Cause (COMPLETE)
**Method:** Isolation test of ffnBwd kernel + 1000x scaling experiment in pipeline
**Duration:** ~20 min hands-on debugging
**Key result:** The kernel works! Gradients are just too small for fp16.

**Evidence:**
- test_ffn_bwd_mini.m standalone test: |output|=8.77 with 0.1-scale inputs → kernel works
- POST-COMPILE test in pipeline with sin() data: |output|=0.001412 → tiny but non-zero
- 1000x scaled gradients in pipeline: |dx|=1.703210 → clearly works
- Original gradients (max 0.000425): |dx|=0.000000 → fp16 underflow

**Chain of underflow:**
```
dlogits → dffn (max ≈ 4.25e-4)
  → fp16 conversion (OK, 4.25e-4 representable)
  → W2^T conv (weights ~0.036): output ≈ 1.5e-5
  → SiLU derivative: more multiplications
  → W1^T/W3^T conv: values below fp16 subnormal threshold
  → FLUSHED TO ZERO
```

**Fix: Loss scaling (standard mixed-precision technique)**
- Multiply dlogits by LOSS_SCALE=1024 before backward
- Divide accumulated gradients by LOSS_SCALE before Adam update
- Agent implementing this now

### Round 5 — Loss Scaling Implementation (IN PROGRESS)
**Agent:** Implementing LOSS_SCALE=1024 in train_pipeline.m
**Expected result:** Loss should decrease from ~10.43

### Round 6 — ANE Kernel Shape Benchmarks (COMPLETE)
**Agent:** Benchmark kernel shapes
**Duration:** ~8 min
**Key results:**

**Peak throughput:**
- Single kernel: **7.61 TFLOPS** (768×2048, spatial 4096)
- Stacked conv: **11.65 TFLOPS** (384ch, sp128, 256 depth)

**Stories-110M shapes at SEQ=256:**
| Shape | TFLOPS |
|-------|--------|
| QKV 768×768 | 0.87 |
| FFN-up 768×2048 | 3.52 |
| FFN-down 2048×768 | 2.07 |

**Critical findings:**
- Zero SRAM spills across all 204 configs (ANE tiles 1x1 convs automatically)
- Dispatch overhead ~0.3ms/kernel — need ≥1 GFLOP per dispatch for >50% efficiency
- Spatial >= 256 needed for reasonable throughput
- 768ch stacked convs hit performance cliff vs 384ch (weight fetch bottleneck)
- FFN-down is ~25% slower than FFN-up for same FLOPs

**Files created:**
- docs/ANE_KERNEL_SHAPE_SWEEP.md — full benchmark report
- training/sweep_training.c — Stories-110M focused sweep (66 configs)

### Round 7 — ffnBwd Isolation Tests (COMPLETE)
**Agent:** Test ffnBwd kernel in isolation
**Duration:** ~5 min
**Key result:** Kernel works perfectly at full size! DIM=768, HIDDEN=2048, SEQ=256:
dx L2=1.91, dh1 L2=4.37, dh3 L2=4.12 with ~99.9% non-zero values.

**Interesting discovery:** Small dimensions FAIL — D=64/H=128/S=16 causes ANE error
`status=0x1d` (Program Inference error). ANE hardware has a minimum kernel size threshold.
All sizes from D=128 upward work.

**Files created:**
- training/test_ffn_bwd_debug.m — standalone isolation test

### Round 8 — Loss Diagnosis Deep Dive (COMPLETE)
**Agent:** Full code path analysis of train_pipeline.m backward pass
**Duration:** ~11 min
**Additional findings beyond our own investigation:**

1. **Missing RoPE in SDPA kernels** — The ANE attention forward/backward kernels do NOT apply
   Rotary Position Embeddings, unlike the CPU-only `forward.h`/`backward.h` which include
   `cpu_rope()`/`cpu_rope_backward()`. This means the model has no positional encoding beyond
   the causal mask. Will severely limit learning quality even after loss scaling fix.

2. **Gradient flow is mathematically correct** — Exhaustive trace confirms all residual connections,
   dW accumulations, rmsnorm backward, and SDPA backward are correctly wired. The ONLY issue is
   FP16 precision.

3. **ACCUM_STEPS=100** confirmed — no weight update until step 100, so steps 0-99 use frozen
   weights by design. This is expected behavior, not a bug.

### Round 9 — RoPE Investigation (COMPLETE)
**Agent:** Investigate missing RoPE impact
**Duration:** ~2 min
**Key findings:**

1. **A working RoPE-in-MIL implementation already exists** at `training_dynamic/mil_dynamic.h`
   (lines 131-168). Uses precomputed cos/sin tables as baked blobs (32 KB each) and
   `rotate_half` via reshape+slice+concat+negate+mul+add.

2. **Low-medium complexity to port**: ~20 extra MIL ops, 64 KB of cos/sin tables, minimal
   kernel size increase. Forward is a direct port. Backward can stay on CPU initially
   (apply cpu_rope_backward to dQ/dK after ANE extraction).

3. **Without RoPE, model quality is severely limited**: the causal mask gives relative ordering
   but NOT absolute position awareness. The model cannot learn position-dependent patterns.

4. **Simpler alternative: ALiBi** — add position-dependent bias to attention scores.
   Zero backward complexity (constant bias, no gradient). But slightly worse quality than RoPE.

**Recommendation:** Add RoPE after loss scaling is validated. It's a port, not invention.

### Round 10 — Loss Scaling Validation (COMPLETE)
**Agent:** Implement loss scaling fix
**Duration:** ~7 min
**Result:** LOSS_SCALE=1024 successfully implemented. Gradient norms are now non-zero:
- |embed|=8070, |W1[0]|=1532, |W2[0]|=4564, |Wq[0]|=233 (pre-unscale)
- Previously ALL were 0.000000

**Loss trend with ACCUM_STEPS=100 (200 steps = 2 Adam updates):**
- Steps 0-99: loss ~10.40 (frozen weights, expected)
- Steps 100-199: loss ~10.39-10.42 (first Adam update applied)
- Over 1000 steps: loss floor drops from ~10.39 to ~10.33 (slow but positive trend)

**Changes to train_pipeline.m:**
- Added `#define LOSS_SCALE 1024.0f`
- Scale dlogits by LOSS_SCALE after loss computation
- Unscale gradients before Adam: `gsc = 1.0f / (steps_batch * LOSS_SCALE)`
- Removed all debug instrumentation

### Round 11 — ANE Compile Budget Discovery (COMPLETE — CRITICAL)
**Method:** Tried ACCUM_STEPS=10 to get faster weight updates
**Result:** SYSTEM-WIDE ANE COMPILE BUDGET EXHAUSTION

**What happened:**
- ACCUM_STEPS=10 → needs 86 kernel recompiles every 10 steps
- exec() restart loop fires rapidly (can't fit 86 kernels in budget)
- Rapid exec() loop creates dozens of processes, each trying to compile
- ANE daemon reaches system-wide compile limit → ALL compiles start failing
- Even killing aned daemon doesn't recover — state persists
- **Likely requires system reboot to clear**

**New "What Doesn't Work" entry:**
- Rapid exec() restart loops poison the ANE daemon's compile budget
- The ~119 compile limit is NOT just per-process — it's system-wide (or at least per-daemon session)
- Once exhausted, ANE compiles fail randomly even for new processes
- Cannot reduce ACCUM_STEPS below ~100 without hitting the limit

**Implication:** ACCUM_STEPS=100 is actually the RIGHT value given the ~119 compile limit.
With 86-99 kernels per batch, we can only do 1 batch per exec() cycle.
To get more Adam updates, we need to reduce KERNELS_PER_LAYER (fewer backward kernels)
or find a way to share kernels across layers.
