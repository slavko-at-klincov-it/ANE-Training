# ANE Training — Plain Language Summary

## The One-Sentence Version

Someone figured out how to **teach AI models new things** using a hidden chip inside every Mac (the Neural Engine) that Apple only officially allows for *running* AI, not *training* it.

---

## What Is the Neural Engine?

Every modern Mac and iPhone has a special chip called the **Apple Neural Engine (ANE)**. Think of it like a dedicated brain for AI tasks — it's incredibly fast and power-efficient at running neural networks.

Apple designed this chip for **inference** — meaning it can *use* a trained AI model (like recognizing your face in photos), but Apple never gave developers tools to *train* models on it (teach the model new things).

This project **cracks that restriction open**.

## How Does It Work? (Simple Version)

Imagine the ANE is a locked kitchen. Apple gives you a window to pass food through (inference), but won't let you inside to cook (training).

This project:
1. **Picks the lock** — discovers the hidden "private APIs" (secret commands) that Apple uses internally
2. **Sneaks recipes in** — feeds the ANE specially formatted instructions (called MIL programs) that look like normal inference work but actually perform training calculations
3. **Passes ingredients through the window** — uses a clever trick where training data (weights) are packed alongside normal input data, so the ANE processes both without knowing the difference

The result: an AI model that **learns** directly on the Neural Engine, something Apple never intended to be possible.

## What Can It Actually Train?

The project demonstrates two real transformer models:

| Model | Size | Speed | What It Is |
|-------|------|-------|-----------|
| **Stories110M** | 109 million parameters | 91ms per step | A small language model that can learn to write short stories |
| **Qwen3-0.6B** | 596 million parameters | 412ms per step | A medium-sized language model capable of more complex text |

For context, ChatGPT has hundreds of billions of parameters — these are much smaller, but they prove the concept works.

---

## Use Case Examples

### 1. Fine-Tuning a Personal AI Assistant — Offline, On-Device

**Scenario:** You want a small AI that understands your company's internal jargon, product names, and writing style — but you can't send confidential data to the cloud.

**How this helps:** You could take a base language model, feed it your internal documents, and fine-tune it directly on your MacBook. The training data never leaves your machine. The ANE does the heavy lifting efficiently without draining your battery like GPU training would.

### 2. Edge ML Research on Apple Silicon

**Scenario:** A researcher at a university wants to study how neural networks learn, but their lab can't afford expensive GPU clusters.

**How this helps:** Every M-series Mac already has an ANE. This project unlocks it as a training accelerator. A student with a MacBook Air could run training experiments that would otherwise require dedicated GPU hardware.

### 3. Real-Time Adaptive Models

**Scenario:** An app that adapts to user behavior in real-time — like a keyboard that learns your vocabulary, or a photo editor that learns your editing style.

**How this helps:** Instead of sending data to a server, the app could retrain small model components directly on the device using the ANE. The model improves as you use it, privately and instantly.

### 4. Federated Learning on Apple Devices

**Scenario:** A health app wants to improve its model using data from thousands of users, but health data is extremely sensitive.

**How this helps:** Each user's device trains locally on their own data using the ANE. Only the learned improvements (gradients) are shared — never the raw data. The ANE makes this practical by being fast and power-efficient.

### 5. Rapid Prototyping for ML Engineers

**Scenario:** An ML engineer wants to quickly test whether a model architecture works before committing to expensive cloud GPU training.

**How this helps:** Run small-scale training experiments locally at 91ms/step. Iterate on architecture, hyperparameters, and data preprocessing without waiting for cloud instances to spin up or paying per-hour GPU costs.

### 6. Understanding Apple's Hardware

**Scenario:** A hardware engineer or systems researcher wants to understand ANE's actual capabilities — SRAM size, throughput limits, quantization behavior.

**How this helps:** The benchmark suite (`sram_bench`, `inmem_peak`, `ane_int8_bench`) systematically probes the ANE's performance characteristics. This is information Apple doesn't publish. The SRAM probing benchmark, for example, reveals that ANE has roughly 4-6MB of fast on-chip memory by detecting where performance drops off.

---

## What This Is NOT

- **Not a product** — it's research code, a proof of concept
- **Not a replacement for PyTorch/TensorFlow** — those are complete ecosystems; this is a focused experiment
- **Not stable** — it uses Apple's private, undocumented APIs that could break with any macOS update
- **Not for large models** — you won't train GPT-4 on your MacBook; this works for small-to-medium models

## Why It Matters

This project demonstrates that **consumer hardware has untapped potential**. The Neural Engine in your Mac is a powerful AI accelerator that sits mostly idle. By showing that training is possible — not just inference — it opens a conversation about what Apple could officially enable if they chose to.

It's also a remarkable piece of reverse engineering: discovering undocumented APIs, understanding binary weight formats, working around hardware limitations (like the 119-compilation limit), and building a complete training pipeline from scratch — all without any documentation from Apple.

## Key Numbers

- **35.1 TOPS** peak throughput (INT8 quantized) — that's 35 trillion operations per second
- **1.88×** speedup with INT8 quantization vs FP16
- **Zero external dependencies** — just macOS system frameworks
- **MIT licensed** — anyone can fork, study, and build on it
