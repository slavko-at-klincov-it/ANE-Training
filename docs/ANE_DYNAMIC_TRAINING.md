# ANE Dynamic Training — `training_dynamic/`

The main working training pipeline for Apple Neural Engine. Trains transformer models
(Llama2-style MHA or Qwen3-style GQA) entirely on-device using reverse-engineered ANE APIs.

## Dynamic Spatial Packing

ANE has a ~119 compile budget per process — after that, compilations silently fail.
A naive approach compiles separate kernels per layer, exhausting this budget in one epoch.

**Solution: Dynamic Spatial Packing.** Pack weights into the IOSurface input alongside
activations. Compile 10 generic kernels once at startup. Each kernel slices its weight
matrices from the spatial dimension of the input tensor at runtime. Per-layer IOSurfaces
hold pre-staged weights; only activations change each step. Result: compile once, train
unlimited steps across any number of layers.

## File Layout

| File | Purpose |
|------|---------|
| `train.m` | Main training loop (forward, backward, Adam, checkpointing) |
| `config.h` | Model-agnostic structs (`LayerWeights`, `LayerActs`, `LayerGrads`, `AdamState`), derived sizes, ANE init |
| `io.h` | IOSurface creation, NEON fp16/fp32 conversion, per-kernel weight staging, GQA tile/reduce |
| `mil_dynamic.h` | MIL program generators for all 10 ANE kernels (RoPE, SDPA, matmuls) |
| `cpu_ops.h` | CPU operations: RMSNorm fwd/bwd, cross-entropy, Adam, embedding, RoPE backward, vocab compaction |
| `generate.m` | Autoregressive text generation from a trained checkpoint |
| `ane_train.h` | High-level C API: 18 functions for training, generation, monitoring |
| `hw_monitor.h` | Background hardware monitor (CPU, GPU, memory, thermal) writing CSV logs |
| `Makefile` | Build system with model selection via `MODEL=` |
| `models/*.h` | Per-model dimension configs (see below) |

## Build & Run

```bash
cd training/training_dynamic

# Build for a specific model
make MODEL=stories110m      # Stories-110M (default)
make MODEL=tiny_ane          # Tiny-ANE-15M (fast iteration)
make MODEL=qwen3_06b         # Qwen3-0.6B (GQA)

# Train from scratch
./train --scratch --steps 5000 --lr 3e-4 --accum 10

# Resume from checkpoint
./train --resume --steps 10000

# All CLI options
./train --scratch \
  --steps 10000    \   # total training steps
  --lr 3e-4        \   # peak learning rate (cosine schedule)
  --accum 10       \   # gradient accumulation steps
  --warmup 100     \   # linear warmup steps
  --clip 1.0       \   # gradient clipping (global norm)
  --wnorm 50       \   # weight norm clamp for Wo/W2 (0=disabled)
  --maxact 100     \   # activation clamp on residual stream (0=disabled)
  --data PATH      \   # path to tokenized binary data (uint16)
  --resume         \   # resume from checkpoint
  --scratch            # train from random init (Xavier/GPT-2 style)
```

## Model Configs

Each model is a header in `models/` defining these macros:

```c
#define MODEL_NAME "Tiny-ANE-15M"
#define DIM 256           // model dimension
#define HIDDEN 768        // FFN hidden dimension
#define HEADS 4           // number of query heads
#define KV_HEADS 4        // number of KV heads (< HEADS = GQA)
#define HD (DIM/HEADS)    // head dimension
#define GQA_RATIO 1       // HEADS / KV_HEADS
#define Q_DIM (HEADS*HD)  // total query dimension
#define KV_DIM (KV_HEADS*HD)
#define SEQ 256           // sequence length
#define NLAYERS 6         // transformer layers
#define VOCAB 32000       // vocabulary size
#define CKPT_PATH "ane_tiny_ckpt.bin"
#define DEFAULT_DATA_PATH "../tinystories_data00.bin"
```

Available models:

| Config | Params | Layers | GQA | Notes |
|--------|--------|--------|-----|-------|
| `tiny_ane` | 15M | 6 | MHA | Fast iteration |
| `small_ane` | ~30M | 8 | MHA | |
| `medium_ane` | ~60M | 10 | MHA | |
| `wide_ane` | ~60M | 8 | MHA | Wider dims |
| `stories110m` | 110M | 12 | MHA | Llama2-style |
| `qwen3_06b` | 600M | 28 | GQA 16/8 | Qwen3 architecture |

To add a new model: create `models/mymodel.h` with all the `#define`s above, then
`make MODEL=mymodel`.

## The 10 ANE Kernels

Compiled once at startup, shared across all layers via per-layer IOSurface requests.

**Forward (3 kernels):**

| Kernel | Operation | Input Shape |
|--------|-----------|-------------|
| `sdpaFwd` | QKV projection + RoPE + causal SDPA (no Wo) | `[1, DIM, 1, SEQ+Q_DIM+2*KV_DIM]` |
| `woFwd` | Attention output projection (attn_out @ Wo^T) | `[1, Q_DIM, 1, SEQ+DIM]` |
| `ffnFused` | W1+W3 projection, SiLU gate, W2 projection + residual | `[1, DIM, 1, 2*SEQ+3*HIDDEN]` |

**Backward (7 kernels):**

| Kernel | Operation | Input Shape |
|--------|-----------|-------------|
| `ffnBwdW2t` | dffn @ W2^T -> dsilu_raw | `[1, DIM, 1, SEQ+HIDDEN]` |
| `ffnBwdW13t` | dh1@W1^T + dh3@W3^T -> dx_ffn | `[1, HIDDEN, 1, 2*SEQ+2*DIM]` |
| `wotBwd` | dx2 @ Wo -> d_attn (gradient through Wo^T) | `[1, DIM, 1, SEQ+Q_DIM]` |
| `sdpaBwd1` | Q,K,V,da -> dV, attention probs, dp (weight-free, has mask) | `[1, 4*Q_DIM, 1, SEQ]` |
| `sdpaBwd2` | probs,dp,Q,K -> dQ, dK (weight-free) | `[1, 2*SCORE_CH+2*Q_DIM, 1, SEQ]` |
| `qBwd` | dQ @ Wq -> dx_q (backprop through Q projection) | `[1, Q_DIM, 1, SEQ+DIM]` |
| `kvBwd` | dK@Wk + dV@Wv -> dx_kv (fused KV backprop) | `[1, KV_DIM, 1, 2*SEQ+2*DIM]` |

Weight gradient computation (dW) runs on CPU via `cblas_sgemm` on a background dispatch
queue, overlapped with the next layer's ANE evaluation. Loss scaling (default 256) prevents
FP16 underflow during backward.

## Hardware Breakdown

Approximate per-step time distribution (Stories-110M, M3 Pro):

- **ANE ~40%** — all matmuls (QKV, Wo, FFN fwd/bwd, SDPA) via MIL kernels
- **CPU ~45%** — RMSNorm fwd/bwd, SiLU backward, cross-entropy loss, Adam optimizer,
  weight gradient accumulation (cblas_sgemm, may use AMX), embedding, RoPE backward
- **IO ~15%** — fp32->fp16 conversion and IOSurface staging (NEON vectorized)
- **GPU 0%** — not used directly (Accelerate BLAS may internally use GPU/AMX)

## Text Generation

```bash
make generate_stories110m
./generate_stories110m --ckpt ane_stories110M_dyn_ckpt.bin \
  --prompt "Once upon a time" --temp 0.8 --topp 0.9 --max_tokens 200
```

Uses only 3 forward kernels (sdpaFwd, woFwd, ffnFused). Supports argmax, temperature,
and top-p nucleus sampling. Loads checkpoint weights, skips Adam state.

CLI options: `--ckpt`, `--tokenizer`, `--prompt`, `--temp`, `--topp`, `--max_tokens`, `--seed`

## Hardware Monitoring

`hw_monitor.h` runs a background thread sampling system metrics at 1s intervals.
Writes `hw_log.csv` with columns:

```
timestamp_ms, step, loss, ane_ms, cpu_ms, io_ms,
cpu_user_ms, cpu_sys_ms, threads,
gpu_util, gpu_mem_mb, ane_util,
mem_rss_mb, mem_avail_mb, thermal
```

Prints a summary at training end (peak RSS, thermal state, thread count).
ANE utilization is not available on consumer macOS (always -1).

## Training Results

See `docs/EXPERIMENT_LOG.md` for detailed experiment results and loss curves.

## High-Level API — `ane_train.h`

`training_dynamic/ane_train.h` provides 18 public functions wrapping the entire training pipeline
into a session-based C API. No need to manage MIL programs, IOSurfaces, or kernels directly.

### API Surface

**Training:**
| Function | Description |
|----------|-------------|
| `ane_train_create(model, data_path)` | Create training session for a model |
| `ane_train_step(session)` | Run one training step (forward + backward + Adam) |
| `ane_train_run(session, steps)` | Run N training steps |
| `ane_train_save(session, path)` | Save checkpoint |
| `ane_train_destroy(session)` | Free all resources |

**Generation:**
| Function | Description |
|----------|-------------|
| `ane_gen_create(checkpoint)` | Load checkpoint for generation |
| `ane_gen_run(session, prompt, callback)` | Generate with streaming token callback |
| `ane_gen_destroy(session)` | Free generation session |

**Hardware Monitoring:**
| Function | Description |
|----------|-------------|
| `ane_hw_snapshot()` | Get current CPU, GPU, memory, thermal state |

### Build as Static Library

```bash
cd training/training_dynamic
make libane_train MODEL=stories110m   # → libane_train_stories110m.a
make libane_train MODEL=tiny_ane      # → libane_train_tiny_ane.a
```

The library is compiled per model because model dimensions are compile-time constants.
Link against the `.a` file from any C, Objective-C, or Swift project.

## Native macOS App

`app/ANETraining.swift` is a SwiftUI menu bar app built on top of `libane_train`.

- **MenuBarExtra** with brain icon — lives in the menu bar, not the dock
- **3 tabs:** Training (start/stop/resume), Generation (prompt + stream), Hardware Monitor
- **614 KB** binary, no Xcode project needed

### Build & Install

```bash
cd app && bash build.sh       # Compiles Swift + links libane_train
# Output: "ANE Training.app"
# Drag to /Applications to install
```

## Integration Guide — Using libane_train from Swift

To use the API from your own Swift app:

1. **Build the static library** for your target model:
   ```bash
   make libane_train MODEL=stories110m
   ```

2. **Add to your Swift project:**
   - Link `libane_train_stories110m.a`
   - Add a bridging header importing `ane_train.h`
   - Link frameworks: `Foundation`, `IOSurface`, `Accelerate`

3. **Call from Swift:**
   ```swift
   // Bridging header:
   // #include "ane_train.h"

   let session = ane_train_create("stories110m", "data.bin")
   defer { ane_train_destroy(session) }

   for _ in 0..<1000 {
       ane_train_step(session)
   }
   ane_train_save(session, "checkpoint.bin")

   // Hardware monitoring
   let hw = ane_hw_snapshot()
   print("CPU: \(hw.cpu_percent)%, Thermal: \(hw.thermal)")
   ```

4. **Generation with streaming:**
   ```swift
   let gen = ane_gen_create("checkpoint.bin")
   defer { ane_gen_destroy(gen) }

   ane_gen_run(gen, "Once upon a time") { token in
       print(String(cString: token), terminator: "")
   }
   ```

See `app/ANETraining.swift` for a complete working example.

## Key Constraints

- **LOSS_SCALE=256** (or 1024) required — FP16 underflow zeros all gradients without it
- **ACCUM_STEPS >= 100 recommended** — 86 kernels per forward+backward vs ~119 compile limit
- **Checkpoint format v4** — includes GQA fields (kv_heads, head_dim, q_dim)
- **Vocab compaction** — maps full vocab (e.g. 151K for Qwen3) to only tokens present in data
- **No rapid restart loops** — poisons ANE daemon system-wide, requires reboot
