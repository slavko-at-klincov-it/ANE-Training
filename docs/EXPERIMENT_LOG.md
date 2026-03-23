# ANE Training Optimization — Experiment Log
**Date:** 2026-03-20
**Branch:** experiment/3h-optimize-session
**Baseline:** 77 ms/step, 2.26 TFLOPS, 0% overlap, loss flat at ~10.43
**Status:** 12 rounds + convergence achieved. Stories-110M generates coherent text (loss 1.86, 41 tok/s). GPU verified not used (0%). Hardware monitor built.
**NOTE:** ANE compile budget exhausted during experiments — **reboot required** to recover.

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
- **Gradient direction verified correct** — numerical gradient check confirms backward produces loss-reducing gradients (verified with SGD at lr 1e-5 to 1.0)
- **Training convergence achieved** — Tiny-ANE-15M to loss 2.54, Stories-110M to loss 3.40 on 1B tokens (Round 12c)

## What Doesn't Work
- ~~Loss not decreasing~~ → **FIXED by loss scaling** (gradients non-zero, slow convergence)
- ~~CPU rmsnorm_bwd had wrong w[i] placement~~ → **FIXED** (Round 12)
- Pipeline overlap = 0% (measurement bug + design limitation)
- Missing RoPE in SDPA kernels (limits model quality)
- **ANE compile budget is SYSTEM-WIDE** — rapid exec() loops poison the daemon, requires reboot
- Cannot reduce ACCUM_STEPS below ~100 (86 kernels per batch vs ~119 compile limit)
- **Activation explosion** — x_cur grows to [-800, 600] over 5000 steps with res_alpha + Adam, attenuating all gradients
- **Long training activation explosion** — even without res_alpha, activations grow to thousands over 25K+ steps with Adam. Stories-110M diverges to NaN at step 45K. Need lower LR or weight clipping for stable long runs.

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

### Round 12 — RMSNorm Backward Bug + Convergence Investigation (COMPLETE)
**Date:** 2026-03-21
**Branch:** experiment/3h-optimize-session
**Method:** Full backward pass code review + CPU-only gradient verification

**Bug found: CPU `rmsnorm_bwd` applies w[i] to the entire gradient expression**

Both `cpu_ops.h` and `stories_cpu_ops.h` had the wrong formula:
```
Old: dx[i] = w[i] * rrms * (dy[i] - x[i] * dot)     ← w[i] wraps BOTH terms
New: dx[i] = rrms * (w[i] * dy[i] - x[i] * dot)     ← w[i] only on dy term
```

The ANE MIL version (`ane_rmsnorm_bwd.h`) was already correct — it computes
`dx = (dy*w - x*coeff) * rrms` which correctly limits w to the dy term.

**Impact assessment:**
- At initialization (w[i]=1.0 for all i), old and new code produce IDENTICAL results
- The bug only manifests after RMSNorm weights train away from 1.0
- Once w[i] ≠ 1.0, the correction term (which keeps gradients orthogonal to the
  normalization manifold) gets distorted by per-dimension w[i] scaling
- This corrupts gradient direction without changing magnitude much, explaining
  why gradients appeared "healthy" but didn't converge

**Gradient verification (CPU-only, no ANE):**
- Numerical gradient check confirms gradient direction is correct after fix
- SGD with computed gradient reduces loss at all learning rates tested (1e-5 to 1.0)
- Single step with lr=1.0: loss 10.40 → 8.69 (embedding-only model)
- Full model with transformer: SGD on 100 embed elements reduces loss by 2.97e-4

**Remaining convergence challenge: activation explosion**
- With Stories-110M (12 layers, dim=768), `x_cur` grows from [-0.13, 0.16] (step 0)
  to [-858, 647] (step 4990) — a ~5000x increase
- RMSNorm compensates by scaling rrms down to ~1/800, attenuating gradients (dy ~1e-3)
- Root cause: `res_alpha = 1/sqrt(2*NLAYERS)` combined with Adam's normalized steps
  (each ≈ ±lr per element) allows weights to grow unboundedly
- The loss on Stories-110M stays at ~10.3-10.5 over 5000 steps (no clear trend)
- Tiny-ANE-15M with 5000 steps at lr=3e-4 also shows no clear trend

**Why models appear stuck at ln(V):**
1. rmsnorm_bwd bug (fixed) — corrupts gradients once w trains away from 1.0
2. Activation explosion — attenuates gradients through deep networks
3. Large vocab (32K) with small dim (256-768) — loss landscape is nearly flat w.r.t.
   transformer weights at initialization (verified: numerical gradient of Wq ≈ 0 with eps=1e-4)
4. Small data (500K tokens) — insufficient for 13-110M param models

**Files changed:**
- `training/training_dynamic/cpu_ops.h` — fixed rmsnorm_bwd dx computation
- `training/stories_cpu_ops.h` — same fix (used by train_pipeline, train_large, etc.)
- `training/ane_rmsnorm_bwd.h` — comment fix (code was already correct)
- `CLAUDE.md` — documented rmsnorm_bwd bug and activation explosion in "What Doesn't Work"

**Next steps to achieve convergence:**
1. ~~Address activation explosion~~ → **DONE** (Round 12b below)
2. Start with smaller vocab: use a byte-level tokenizer (256 tokens) or char-level
3. More training data: 500K tokens is ~1000x too small for 110M params
4. Hyperparameter sweep: try lr=1e-4, wd=0.01, beta2=0.999 (more conservative)
5. Run 50000+ steps (5000+ Adam updates) to see through batch-to-batch noise

### Round 12b — Activation Explosion Fix + First Convergence (COMPLETE)
**Date:** 2026-03-21
**Method:** Remove res_alpha scaling, GPT-2 style output projection init

**Changes:**
1. Removed `res_alpha = 1/sqrt(2*NLAYERS)` — set to 1.0 (standard residual: `x + layer_output`)
2. GPT-2 style init: output projections (Wo, W2) scaled by `1/sqrt(NLAYERS)` instead of `1/sqrt(2*NLAYERS)`
3. Removed baked res_alpha from FFN fused MIL kernel (was `x2 + alpha*ffn_out`, now `x2 + ffn_out`)
4. Added per-layer gradient norm tracking (L0, Lmid, Llast)

**Results — Tiny-ANE-15M, 10K steps, accum=10, lr=3e-4:**
```
Step    Loss     x_range         Notes
   0    10.37   [-1.0, 1.1]     Random baseline (ln(32K)=10.37)
1000    10.29   [-3.1, 3.4]     Warmup complete
2000    10.41                    Batch noise
3000    10.30
4000    10.32
5000    10.27                    Clear downward trend
6000    10.01                    Breaking away from baseline!
7000     9.97   [-3.1, 3.4]     Best loss — 0.40 below baseline
8000    10.13
9000    10.19
9950     9.94                    Loss still improving at end
```

**Key improvements vs previous:**
- Activations: x stays at [-3, 4] (was [-800, 600] with res_alpha)
- Loss: 10.37 → 9.94 (was stuck at 10.4 indefinitely)
- Gradient flow: stable across all layers (L0=0.08, L3=0.01, L5=0.01)
- Training speed: 103.6 ms/step, 17.6 min wall for 10K steps

**Why lr=1e-3 doesn't work:**
- Tried lr=1e-3 with warmup=200: activations still explode to [-446, 3870]
- The standard residual (no scaling) amplifies each layer's output more,
  making the model more sensitive to learning rate
- lr=3e-4 is the right range for this architecture

**Files changed:**
- `training/training_dynamic/train.m` — res_alpha=1.0, GPT-2 init, per-layer grad norms
- `training/training_dynamic/mil_dynamic.h` — removed baked alpha from FFN fused kernel
- `CLAUDE.md` — updated findings

### Round 12c — 1B Token Training: Full Convergence Achieved (COMPLETE)
**Date:** 2026-03-21
**Method:** Full TinyStories dataset training with fixes from Round 12 + 12b

**Data upgrade:**
- Downloaded full TinyStories dataset: 50 shards, 1.025B tokens, 1.9GB (`tinystories_all.bin`)
- Previous data was only 500K tokens (`tinystories_data00.bin`) — 2000x more data now

**Results — Tiny-ANE-15M (6 layers, dim=256), accum=10, lr=3e-4:**
```
Step     Loss     Notes
    0    9.66     Below ln(32K)=10.37 (pretrained embed?)
 5000    ~6.5     Rapid descent
10000    ~4.5     Still dropping fast
15000    ~3.2     Approaching convergence
20000    2.54     20K steps = 2000 Adam updates, ~35 min wall
```

**Results — Stories-110M (12 layers, dim=768), accum=10, lr=3e-4:**
```
Step     Loss     Notes
    0    9.72     Near random baseline
 2500    ~6.0     Fast initial learning
 5000    ~4.5     Steady descent
 7500    ~3.8     Slowing but still improving
10000    3.40     10K steps = 1000 Adam updates, ~35 min wall
```

**Key observations:**
- Activations stable: Stories-110M x stays at [-63, 73] (was [-969, 773] on 500K tokens)
- Both models show clear continuous downward loss trend — no plateaus
- For reference: ln(32000) = 10.37 is random baseline, so loss 3.40 represents massive learning
- More data was the key missing ingredient — 500K tokens was ~2000x too small
- All fixes compounding: loss scaling (Round 10) + rmsnorm_bwd fix (Round 12) + res_alpha removal (Round 12b) + sufficient data

**What made convergence possible (all required):**
1. LOSS_SCALE=1024 — prevents FP16 underflow in ANE backward
2. Fixed rmsnorm_bwd w[i] placement — correct gradient direction
3. Removed res_alpha — stable activations (no explosion)
4. 1B tokens instead of 500K — sufficient data for model capacity
5. lr=3e-4, accum=10 — conservative but effective hyperparameters

### Round 12d — 50K Step Training + Text Generation (COMPLETE)
**Date:** 2026-03-21
**Method:** Extended Stories-110M training to 50K steps on 1B tokens (accum=10, lr=3e-4) + built text generation pipeline

**Results — Stories-110M 50K steps:**
```
Step     Loss     Notes
    0    9.72     Random baseline
 5000    4.80     Rapid descent
10000    3.78     Still improving
15000    3.57     Slowing
26200    3.00     Best checkpoint saved
35000    3.27     Best overall (moving avg)
45000    NaN      Diverged — activation explosion
```

**Loss curve:** 9.72 → 4.80 (5K) → 3.78 (10K) → 3.57 (15K) → 3.27 (35K best) → NaN (45K diverged)
**Best checkpoint:** step 26200, loss=3.00
**Model DOES learn** (loss 9.72→3.00 = massive improvement) but can't sustain long training

**Activation explosion analysis:**
- `x` grows from [-1, 1] to [-4400, 4700] at step 40K, then NaN
- Standard residual (`x + layer_output`) without res_alpha prevents early explosion but not long-term
- Over 25K+ steps, Adam pushes weights to grow, layer outputs grow, residuals accumulate
- Weight decay (wd=0.1) isn't constraining enough
- Needs: lower LR (1e-4), or weight clipping, or activation clamping for stable long training

**Text generation:**
- Built `generate.m` (~310 lines): load checkpoint, 3 ANE kernels, autoregressive generation
- Tiny-ANE-15M (loss 2.35): 3.2 tok/s, semi-coherent
- Stories-110M (loss 3.00): 4.0 tok/s, has structure but too much noise at this loss level

**Files created:**
- `training/training_dynamic/generate.m` — text generation with ANE inference

### Round 12e — Overnight 100K Training + Hardware Monitor (2026-03-22/23)

**Hardware monitor built (`hw_monitor.h`):**
Samples CPU, GPU, memory, thermal every second during training. Writes `hw_log.csv`.

**Definitive hardware measurements (Stories-110M, M3 Pro):**
| Metric | Value |
|--------|-------|
| GPU utilization | **0%** — GPU is NOT used |
| GPU memory | 0.1 MB — no Metal shaders |
| BLAS throughput | 988-1547 GFLOPS — AMX coprocessor, not GPU |
| ANE utilization | Not measurable (Apple locks on consumer macOS) |
| Thermal | Always Nominal |
| RAM | 2923 MB |

**Overnight training results (100K steps Stories-110M + 10K steps Qwen3):**

Stories-110M (100K steps, lr=1e-4, accum=10, maxact=100, ~4.4h):
```
Step     Loss
0        9.72
20K      2.61
40K      2.23
60K      2.11
80K      1.86  ← best
100K     2.13
```
- No NaN, activations stable, 121.5 ms/step
- Best checkpoint at loss 1.86: generates coherent children's stories at 41 tok/s
- Sample: "He loved to play with his toys and his friends. One day, Max saw a big,
  red ball on the ground. He was so happy..."

Qwen3-0.6B SEQ=128 (10K steps, lr=1e-4, accum=10, ~2.5h):
```
Step     Loss
0        9.66
1K       5.53
5K       5.75
9K       4.77
```
- Survives on 18GB Mac with SEQ=128, 334 ms/step
- Learning but slow — needs more steps or higher LR

**Activation clamping results:**
- `--maxact 50` + lr=3e-4: saturates, kills gradients → NaN at step 15K
- `--maxact 100` + lr=1e-4: **perfect** — clamp never triggers, acts as safety net
- Weight norm clamping (`--wnorm`) alone insufficient — other weights also grow

**Key insight:** lr=1e-4 is the primary stabilizer. maxact=100 is insurance for very long runs.
