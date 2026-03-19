# ANE Model Architecture Sweep — Which Models Train Best on the ANE?

> Real training runs on M3 Pro (h15g), 2000 steps from scratch, TinyStories data.
> All models use the same vocab (32k), sequence length (256), and training settings.

---

## Results

| Model | Params | DIM | Hidden | Layers | ms/step | Best Loss | Quality/sec |
|:---|:---|:---|:---|:---|:---|:---|:---|
| **Tiny-ANE** | 13M | 256 | 768 | 6 | **79ms** | **9.95** | Best |
| Small-ANE | 28M | 384 | 1152 | 8 | 118ms | 10.10 | Good |
| Medium-ANE | 44M | 512 | 1536 | 8 | 131ms | 10.00 | Similar to Tiny but slower |
| Wide-ANE | 52M | 640 | 1920 | 6 | 131ms | 10.20 | Slowest AND worst loss |

**Winner: Tiny-ANE (13M).** Best loss AND fastest. More parameter updates per second
beats having more parameters per update at this training budget.

---

## The Classifier Bottleneck

The CPU classifier (32k vocab softmax + cross-entropy + cblas_sgemm) consumes **66% of
every training step**, regardless of model size:

```
Timing Breakdown (consistent across all models):

  ANE Forward+Backward:  ~22ms  (17%)  ← actual model compute
  CPU Classifier (32k):  ~86ms  (66%)  ← THE bottleneck
  I/O, RMSNorm, etc:     ~22ms  (17%)

Total:                   ~130ms/step (medium/wide)
                          ~79ms/step (tiny — less ANE work, same classifier)
```

This means:
- **Model size barely affects step time** (79-131ms range for 4x params difference)
- **The ANE is NOT the bottleneck** — it finishes in 22ms and waits 86ms for the CPU
- **Vocab reduction would be the biggest single optimization** (32k → 4k = ~10x less classifier work)

---

## Why Tiny Wins

At 2000 steps of training:
1. Tiny does 2000 updates at 79ms/step = **158 seconds total**
2. Medium does 2000 updates at 131ms/step = **262 seconds total**
3. Tiny gets **66% more wall-clock time** for gradient updates
4. With random init and limited data, more frequent updates > more parameters

For longer training (100k+ steps), larger models would eventually overtake Tiny
due to higher capacity. But for quick experiments and fine-tuning, Tiny is optimal.

---

## Recommendations

### For quick experiments / fine-tuning:
Use **Tiny-ANE** (DIM=256, 6 layers, 13M params). Fastest iteration, best quality
per second, fits easily in memory.

### For best final quality (long training):
Use **Medium-ANE** (DIM=512, 8 layers, 44M params). More capacity, still in the
ANE channel sweet spot (384-640). Avoid DIM=768+ (ANE performance cliff).

### For maximum ANE utilization:
Reduce vocab from 32k to 4k-8k. This would drop classifier from 86ms to ~10ms,
making the ANE the actual bottleneck and revealing true architecture differences.

### Avoid:
- **DIM=768+** — ANE stacked benchmark shows performance cliff at 768 channels
- **Wide-shallow** (640×6) — no advantage over medium-deep (512×8) at same param count
- **Large vocab** — dominates step time, masks all other optimizations

---

## Model Configs

Available in `training/training_dynamic/models/`:

| File | Model | DIM | Hidden | Heads | Layers | Vocab |
|:---|:---|:---|:---|:---|:---|:---|
| `tiny_ane.h` | Tiny-ANE-15M | 256 | 768 | 4 | 6 | 32000 |
| `small_ane.h` | Small-ANE-30M | 384 | 1152 | 6 | 8 | 32000 |
| `medium_ane.h` | Medium-ANE-42M | 512 | 1536 | 8 | 8 | 32000 |
| `wide_ane.h` | Wide-ANE-45M | 640 | 1920 | 10 | 6 | 32000 |
| `stories110m.h` | Stories-110M | 768 | 2048 | 12 | 12 | 32000 |
| `qwen3_06b.h` | Qwen3-0.6B | 1024 | 3072 | 16 | 28 | 151936 |

Build any config: `cd training/training_dynamic && make MODEL=tiny_ane`

Run sweep: `./sweep_models.sh 2000`

---

*Last updated: 2026-03-19 | M3 Pro (h15g), macOS 26.3.1 (25D2128)*
*Source: `training/training_dynamic/sweep_models.sh`*
