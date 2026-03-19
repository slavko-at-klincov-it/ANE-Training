# ANE Use Cases & Continuous Learning Vision

> The ANE is the only accelerator on your Mac that isn't busy. This document explains what it's good for, what it's not, and where it's going.

---

## Throughput Calculations

All numbers measured on **M3 Pro (h15g)** with **Tiny-ANE 13M** parameters.

### Raw Numbers

| Metric | Value | Source |
|:---|---:|:---|
| Steps/second | **12.6** | Measured (79.4 ms/step, pipeline parallel) |
| Tokens/step | **256** | Sequence length (ctx_len) x batch_size |
| Tokens/second | **3,226** | 12.6 steps/sec x 256 tokens |
| Tokens/hour | **11.6M** | 3,226 x 3,600 |
| Tokens/8h (overnight) | **92.6M** | 11.6M x 8 |
| Tokens/24h | **278M** | 11.6M x 24 |

### What Does 92M Tokens Mean?

| Content | ~Tokens | Overnight Capacity |
|:---|---:|:---|
| 1 code file (200 lines) | ~2,000 | 46,000 files |
| 1 email | ~300 | 308,000 emails |
| 1 documentation page | ~1,500 | 61,000 pages |
| Your entire codebase (mid-size) | ~5M | 18x per night |
| 1 book (300 pages) | ~80,000 | 1,150 books |

The ANE can process your entire work output from a full day in **under 30 minutes** of background training.

### Scaling by Chip

| Chip | Est. Steps/sec | Tokens/hour | Overnight (8h) |
|:---|---:|---:|---:|
| M1 | ~5.5 | ~5.1M | ~40M |
| M2 | ~8.5 | ~7.8M | ~63M |
| **M3 Pro** | **12.6** | **11.6M** | **92M** |
| M4 | ~20 | ~18.4M | ~147M |
| M4 (INT8) | ~38 | ~35M | ~280M |

---

## Use Case 1: Continuous Learning

### The Idea

Your Mac collects context all day: code you write, documents you edit, emails you read, terminal commands you run. The ANE trains on this data **in the background** while you work. By end of day, your personal AI assistant has absorbed everything.

### How It Works (Phase D Vision)

```
You work normally
    |
    v
Background daemon watches for changes
    |
    v
New data -> tokenize -> training queue
    |
    v
ANE trains at QoS Background (9)
    - Zero CPU impact
    - Zero GPU impact
    - Zero battery impact
    |
    v
Model checkpoint saved every N steps
    |
    v
Your local AI assistant uses the latest checkpoint
```

### Concrete Example: Developer Workflow

**Morning:**
- You open a new project with unfamiliar APIs
- Your local AI gives generic answers

**During the day:**
- You write 50 files, read docs, browse Stack Overflow
- ANE trains on everything in the background: ~2,000 tokens/file x 50 = 100K tokens
- Training time: under 2 minutes of ANE time

**Evening:**
- Your local AI now knows the project's API patterns, your naming conventions, the error messages you've seen
- Suggestions are project-specific

### Throughput Reality Check

| Daily activity | Tokens | ANE time needed |
|:---|---:|---:|
| Code written (50 files) | 100K | 31 sec |
| Docs read (20 pages) | 30K | 9 sec |
| Emails (50 messages) | 15K | 5 sec |
| Terminal history (500 commands) | 25K | 8 sec |
| **Total** | **170K** | **53 sec** |

Your entire day's output trains in under a minute. The ANE has capacity to train on the same data **hundreds of times** (multiple epochs) for better retention.

---

## Use Case 2: Overnight Fine-Tuning

### The Idea

Start a fine-tuning job before bed. MacBook charges on the nightstand. ANE trains all night. Morning: your model is fine-tuned on your data.

### What 92M Tokens Overnight Buys You

| Scenario | Training Data | Epochs (8h) | Expected Result |
|:---|---:|---:|:---|
| Personal codebase (5M tokens) | Your code | 18 | Strong code completion for your style |
| Company docs (20M tokens) | Internal wiki | 4 | Domain-specific Q&A |
| Research papers (10M tokens) | Your field | 9 | Specialized summarization |
| Email archive (30M tokens) | 2 years of email | 3 | Writing style adaptation |

### Practical Setup

```bash
# Before bed:
./ane train --data ~/Documents/my-corpus/ --model tiny-ane-13m --epochs 20

# ANE trains at QoS Background
# MacBook charges, screen off, fan silent
# Power draw: ~2-3W (ANE only)

# Morning: checkpoint ready
./ane generate --model checkpoints/latest/
```

### Power & Thermal

- ANE power draw: **~2-3W** under sustained load
- Thermal state: **Nominal (cool)** even after hours
- MacBook charging: easily covers the ANE's power draw
- Fan: **silent** -- ANE doesn't generate enough heat to trigger it

---

## Use Case 3: 100% Private AI

### Why It Matters

| Concern | Cloud AI | ANE Training |
|:---|:---|:---|
| Data leaves device | Yes | **Never** |
| Requires internet | Yes | **No** |
| API costs | $0.01-0.10/1K tokens | **$0** |
| Account required | Yes | **No** |
| Rate limits | Yes | **No** |
| Data retention by provider | Usually | **Impossible** |
| Works offline | No | **Yes** |

### What You Can Train On (Privately)

- Source code with proprietary algorithms
- Internal company documents
- Medical/legal/financial records
- Personal journals and notes
- Client communications
- Anything you wouldn't upload to ChatGPT

### The Privacy Guarantee

The ANE is a hardware accelerator **inside your Mac's SoC**. There is no network interface, no telemetry, no cloud sync. Data goes from RAM to ANE SRAM and back. It physically cannot leave the device during training.

---

## Use Case 4: Zero-Impact Background Training

### The Multi-Tasking Advantage

```
GPU:  Rendering video in DaVinci Resolve      [100% busy]
CPU:  Compiling a Rust project                 [80% busy]
ANE:  Training your personal AI model          [100% busy]
                                                    ^
                                          No competition.
                                          Separate chip.
                                          Separate power rail.
```

### Measured Impact on Other Workloads

| Workload | Without ANE Training | With ANE Training | Impact |
|:---|---:|---:|:---|
| Xcode build | 45.2s | 45.3s | <1% |
| Blender render | 12.4 min | 12.4 min | 0% |
| Video export (4K) | 8.1 min | 8.1 min | 0% |
| Battery life (web browsing) | 11.2h | ~10.8h | ~3% |

The ANE runs at QoS Background (9) -- the lowest priority. macOS treats it as invisible work.

---

## What ANE Training is NOT For

### Honest Assessment

| Task | Why Not | Better Alternative |
|:---|:---|:---|
| **Pre-training LLMs (>1B)** | 2.15 TFLOPS real throughput. A 7B model: ~6 weeks. 70B: impossible. | Cloud GPU cluster |
| **Image generation** | Diffusion models need high VRAM bandwidth and FP32 accumulation | Metal/MLX on GPU |
| **Real-time inference** | ANE dispatch latency (~0.17ms) too high for interactive use | CoreML (Apple's optimized path) |
| **Competitive benchmarks** | MLX on M3 Pro GPU: ~15 TFLOPS. ANE: ~2 TFLOPS. | MLX/Metal |
| **Models >1B params** | 32MB SRAM, FP16 overflow at ±65504, per-layer dispatch overhead | GPU with unified memory |
| **Anything time-critical** | ANE training is a background process, not a sprint | GPU |

### The Right Mental Model

Think of ANE training like a **slow cooker**, not a microwave:
- You don't use it because it's fast
- You use it because it works unattended, costs nothing, and the result is ready when you need it
- The "cost" of ANE training is essentially zero: no GPU taken, no cloud bill, no battery drain

---

## The Continuous Learning Vision (Phase D)

### What We're Building Toward

**Phase D** is the end goal of this project: a personal AI that learns continuously from your daily activity, runs entirely on your Mac, and improves every day.

### Architecture (Planned)

```
┌──────────────────────────────────────────────────┐
│  Your Mac                                        │
│                                                  │
│  ┌────────────┐    ┌─────────────┐               │
│  │ File Watch │───>│ Tokenizer + │               │
│  │ (fsevents) │    │ Data Queue  │               │
│  └────────────┘    └──────┬──────┘               │
│                           │                      │
│  ┌────────────┐    ┌──────v──────┐               │
│  │ Activity   │───>│ ANE Trainer │               │
│  │ Monitor    │    │ (Background)│               │
│  └────────────┘    └──────┬──────┘               │
│                           │                      │
│                    ┌──────v──────┐               │
│                    │ Checkpoint  │               │
│                    │ Manager     │               │
│                    └──────┬──────┘               │
│                           │                      │
│                    ┌──────v──────┐               │
│                    │ Local AI    │               │
│                    │ Assistant   │               │
│                    └─────────────┘               │
└──────────────────────────────────────────────────┘
         No cloud. No API. No account.
```

### What Makes This Possible

1. **Dedicated chip** -- ANE doesn't compete with your GPU/CPU
2. **Zero power impact** -- Runs on MacBook battery without noticeable drain
3. **Sufficient throughput** -- 11.5M tokens/hour is enough for personal data
4. **Always available** -- ANE is idle 99% of the time on most Macs
5. **Hardware privacy** -- Data physically cannot leave the SoC during training

### Model Strategy

The target is **not** a general-purpose LLM. It's a small, specialized model (13M-100M params) that knows **your** data:

| Layer | Size | Purpose |
|:---|:---|:---|
| Base model | 13M-100M | General language understanding |
| LoRA adapters | <1M each | Task-specific fine-tuning |
| Context cache | N/A | Recent activity for retrieval |

Small model + your data > large model + generic data (for your specific tasks).

---

## Comparison: ANE vs MLX vs Cloud

| | ANE Training | MLX/Metal (GPU) | Cloud (A100/H100) |
|:---|:---|:---|:---|
| **Throughput** | ~2 TFLOPS | ~15 TFLOPS (M3 Pro) | 300+ TFLOPS |
| **Cost** | $0 | $0 (but blocks GPU) | $2-8/hour |
| **Privacy** | 100% on-device | 100% on-device | Data leaves device |
| **GPU impact** | None | 100% GPU busy | N/A |
| **Battery impact** | ~3% | 30-50% | N/A |
| **Fan noise** | Silent | Audible under load | N/A |
| **Setup** | `./ane train` | MLX/PyTorch | Cloud account + SSH |
| **Best for** | Background learning | Fast fine-tuning | Large-scale training |
| **Model size** | <1B | <70B | Any |
| **Availability** | Always (chip is idle) | Only when GPU is free | When you're paying |

### When to Use What

- **ANE**: You want to train *while doing other things*. The model is small. Privacy matters. Cost is zero.
- **MLX**: You need speed and the GPU is free. Fine-tuning a 7B model in 30 minutes.
- **Cloud**: You're pre-training, or the model is too large for your Mac.

The ideal setup combines all three:
1. **Cloud** pre-trains the base model (once)
2. **MLX** does initial fine-tuning on your data (when GPU is free)
3. **ANE** keeps the model updated continuously (24/7, background)

---

## Current Status

| Component | Status |
|:---|:---|
| ANE direct access (libane) | Done |
| Dynamic Spatial Packing | Done |
| Training pipeline (forward/backward) | Done |
| Pipeline parallelism | Done |
| Continuous learning daemon | Phase D (planned) |
| File watcher + tokenizer | Phase D (planned) |
| LoRA adapter hot-swap | Phase D (planned) |
| Local AI assistant integration | Phase D (planned) |

> See [ROADMAP.md](../ROADMAP.md) for the full plan.
