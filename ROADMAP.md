# Roadmap — ANE-Training Optimizations

As of: March 2026. Based on analysis of the Orion Paper, NeuralForge, eiln/ane Linux driver, and our own codebase analysis.

---

## P0 — Completed

### 1. Dynamic Spatial Packing (Zero Recompilation) — DONE

**Problem:** ANE bakes weights into the HWX binary at compile time. Every weight update requires recompilation. Limit: ~119 compilations per process.

**Solution:** Weights as input IOSurface channels instead of BLOBFILE constants. Compile once, update weights via IOSurface write.

```
Before:      compile(mil + weights) → 60 Compiles for 60 Steps
After:       compile(mil_dynamic) → 1 Compile for ∞ Steps
```

**Implementation:**
- `ane_mil_linear_dynamic()` — MIL with weights-as-input (slice → reshape → matmul)
- `ane_write_dynamic_weights()` — packs W[out][in] into the correct IOSurface layout
- Input: `[1, in_ch + in_ch*out_ch, 1, seq]` — activations + weights together
- Both training demos (`demo_train.c`, `generate.c`) converted

**Result:** Compile count 1 instead of 60. ~119 limit completely bypassed. 0.1ms/step.

**RE insight:** Delta Compilation (disk patching after unload/reload) does NOT work — ANE bakes weights into HWX and does not re-read them from disk. `ane_reload_weights()` remains as EXPERIMENTAL API.

---

### 2. FP16 Overflow Protection — DONE

**Problem:** FP16 max = 65504. Softmax/RMSNorm can cause overflow → NaN → training diverges.

**Solution:** Clamp activations, sanitize gradients. Implemented in `demo_train.c` and `generate.c`.

```c
// Before Softmax/RMSNorm:
for (int i = 0; i < n; i++)
    x[i] = fminf(fmaxf(x[i], -65504.0f), 65504.0f);

// After gradient computation:
for (int i = 0; i < n; i++) {
    if (isnan(grad[i])) grad[i] = 0.0f;
    if (isinf(grad[i])) grad[i] = copysignf(65504.0f, grad[i]);
}
```

**Files:** `examples/demo_train.c`, `examples/generate.c`

---

### 3. Process Restart at ~119 Compilations — OBSOLETE (solved by DSP)

**Problem:** ANE allows ~119 compilations per process, then silent failures.

**Solution:** Compilation counter + `execl()` with resume flag.

```c
static int g_compile_count = 0;

ANEKernel *ane_compile(...) {
    if (++g_compile_count > 110) {  // Safety margin
        checkpoint_save(state);
        execl(argv[0], argv[0], "--resume", checkpoint_path, NULL);
    }
    // ... normal compile
}
```

**Note:** Dynamic Spatial Packing already bypasses this problem (1x compile, ∞ weight updates). Still worth implementing as a safety net.

**Files:** `libane/ane.m` (counter), `examples/demo_train.c` (checkpoint/resume)

---

### 4. Zero-Copy for Weight Updates — DONE (part of DSP)

**Problem:** `ane_write()` does lock → memcpy → unlock (55-65µs per call).

**Solution:** Write directly into IOSurface via `ane_input_ptr()`.

```c
// Instead of:
ane_write(k, 0, updated_weights, bytes);

// Better:
ane_lock_input(k, 0);
float *ptr = (float *)ane_input_ptr(k, 0);
// Only update changed weights (delta):
for (int i = 0; i < n_changed; i++)
    ptr[changed_idx[i]] = new_val[i];
ane_unlock_input(k, 0);
```

**Files:** `examples/demo_train.c`, `examples/generate.c`

---

### 5. SRAM Budget Tracking — DONE

**Problem:** M3 Pro SRAM ~16MB, throughput drop above 73.5MB, cliff at 129MB.

**Solution:** Track tensor sizes, warn when budget is exceeded.

```c
// In ane_compile():
size_t total_io = 0;
for (int i = 0; i < n_inputs; i++) total_io += input_sizes[i];
for (int i = 0; i < n_outputs; i++) total_io += output_sizes[i];
if (total_io > 16 * 1024 * 1024)
    fprintf(stderr, "libane: warning: total I/O %zuMB exceeds SRAM (~16MB)\n", total_io >> 20);
```

**Files:** `libane/ane.m`

---

### 6. Alphabetical IOSurface Sorting

**Problem:** Multi-input/output surfaces must be sorted alphabetically by MIL variable name. Wrong order = silent failures.

**Source:** Orion Constraint #3, #19

**Files:** `libane/ane.m` (sorting in `ane_compile()`)

---

## P1 — Medium-term (architectural changes)

### 7. Pipeline Parallelism (30-40% Latency)

**Problem:** CPU backward pass (30ms) blocks while ANE is idle.

**Solution:** Backward step N in parallel with forward step N+1.

```
Current (sequential):
  ANE forward [22ms] → CPU backward [30ms] → ANE forward [22ms] → ...
  Total: 52ms/step

Pipeline:
  ANE forward N [22ms] → ANE forward N+1 [22ms]
  CPU backward N [30ms]  ← parallel!
  Total: ~30ms/step (ANE is bottleneck instead of CPU)
```

**Implementation:** Async ANE eval via `_ANEClient::enqueueSetsWithModel:` + completion handler.

---

### 8. LoRA Adapter-as-Input

**Problem:** Weight updates require recompilation or delta compilation.

**Solution:** Base weights in BLOBFILE (compile once), LoRA adapters (A, B matrices) as IOSurface input.

```
MIL:
  base_W = file_value("@model_path/weights/base.bin")  // baked
  lora_A = input("lora_a", [rank, in_dim])              // IOSurface
  lora_B = input("lora_b", [out_dim, rank])              // IOSurface
  W_eff  = base_W + matmul(lora_B, lora_A)
  output = conv(input, W_eff)
```

**Advantage:** Zero recompilation for fine-tuning. Hot-swap of adapters.

**Source:** Orion Section 4.3

---

### 9. Kernel Chaining (`_ANEChainingRequest`)

**Problem:** Each kernel eval has a CPU roundtrip (~0.2ms). For a 12-layer transformer: 22 roundtrips = 4.4ms overhead.

**Status:** Object creation works, `validate()` fails (parameter type mismatch).

**Next steps:**
1. Identify NSNumber parameters that should actually be arrays
2. Correctly build `_ANEBuffer` objects for intermediate tensors
3. Test `_ANEClient::prepareChainingWithModel:chainingReq:qos:error:`

**Estimated gain:** 3-5x forward pass speedup (for deep networks).

---

### 10. RMSNorm on ANE

**Problem:** 24 RMSNorm operations per step (2 per layer x 12 layers) run on CPU (~4-6ms).

**Solution:** MIL program for RMSNorm:

```
reduce_mean(square(x)) → add(eps) → rsqrt → mul(x) → mul(gamma)
```

All ops are ANE-compatible. Compile as separate kernel or fuse into existing kernels.

---

### 11. Causal Masking via `where()` on ANE

**Problem:** Attention runs on CPU because ANE cannot do causal masking.

**Solution:** MIL `where()` operator:

```
causal_mask = constant([1, heads, seq, seq])  // Lower-triangular, baked
scores = matmul(Q, K_T) / sqrt(d)
masked = where(causal_mask, scores, constant(-65504.0))
attn = softmax(masked, axis=-1)
output = matmul(attn, V)
```

**Limitation:** Causal mask is fixed per sequence length. For variable length: pre-compile multiple kernels.

---

## P2 — ~~Long-term (Hybrid ANE + GPU)~~ REJECTED

> **Status: Explored and rejected (2026-03-19).**
> GPU (Metal) benchmarked at 3-8x slower than CPU/AMX for training-relevant matmuls.
> ANE + GPU combination tested — adds synchronization overhead without throughput gain.
> ANE's real USP is **background training** (GPU stays free), not competing with GPU on throughput.
> For max throughput, use MLX (GPU-only). For background training, use ANE (this project).
>
> See `docs/ANE_MODEL_SIZE_BENCHMARK.md` for detailed benchmark data.

---

## New Projects & Sources

| Project | Relevance |
|:---|:---|
| [Orion Paper](https://arxiv.org/abs/2603.06728) | Delta Compilation, LoRA-as-Input, 20 Constraints |
| [NeuralForge](https://github.com/Khaeldur/NeuralForge) | Process-Restart, GGUF-Export, Gradient Accumulation |
| [ANEMLL](https://github.com/Anemll/Anemll) | ANE ML Library |
| [SqueezeBits Yetter](https://blog.squeezebits.com/disaggregated-inference-on-apple-silicon-npu-prefill-and-gpu-decode-67176) | Disaggregated Inference (ANE Prefill + GPU Decode) |
| [Metal FlashAttention 2.0](https://engineering.drawthings.ai/) | GPU Attention |
| [Apple MLX M5 Research](https://machinelearning.apple.com/research/exploring-llms-mlx-m5) | GPU Neural Accelerator Benchmarks |
| [eiln/ane](https://github.com/eiln/ane) | E5 Binary Format Analysis |
| [tzakharko M5 Microbenchmarks](https://tzakharko.github.io/apple-neural-accelerators-benchmark/) | GPU-NA Tile-Size, FLOPS/Core |

---

## Current Bottleneck Analysis

```
Training Step (91ms, M3 Pro, Stories110M):

  ANE Forward      ████████████░░░░░░░░░░░░  22ms (24%)
  CPU Attention     ███░░░░░░░░░░░░░░░░░░░░░   5ms  (5%)
  ANE Backward     ██████████████░░░░░░░░░░  30ms (33%)  ← partially CPU (dW)
  CPU RMSNorm      ███░░░░░░░░░░░░░░░░░░░░░   5ms  (5%)
  CPU dW Gradients ████████████████████░░░░  20ms (22%)  ← CBLAS matmul
  CPU Adam Update  ██░░░░░░░░░░░░░░░░░░░░░░   3ms  (3%)
  Overhead         ██████░░░░░░░░░░░░░░░░░░   6ms  (7%)

  ANE: 41% | CPU: 59% ← CPU is bottleneck
```

**After P0+P1 optimizations (estimated):**

```
  ANE Forward      ████████████░░░░░░░░░░░░  22ms
  ANE Attention    ███░░░░░░░░░░░░░░░░░░░░░   3ms  ← was CPU
  ANE Backward     ██████████████░░░░░░░░░░  15ms  ← parallel
  CPU dW (parallel)████████████░░░░░░░░░░░░   0ms  ← hidden behind ANE
  Delta Reload     ████░░░░░░░░░░░░░░░░░░░░   5ms  ← was 4200ms

  Estimated step: ~30ms (3x faster)
```
