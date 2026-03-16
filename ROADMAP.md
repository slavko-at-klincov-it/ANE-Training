# Roadmap — ANE-Training Optimierungen

Stand: März 2026. Basierend auf Analyse des Orion Papers, NeuralForge, eiln/ane Linux-Driver, und eigener Codebase-Analyse.

---

## P0 — Erledigt

### 1. Dynamic Spatial Packing (Zero Recompilation) — DONE

**Problem:** ANE bakt Weights beim Compile ins HWX-Binary. Jeder Weight-Update erfordert Recompilation. Limit: ~119 Compilations pro Prozess.

**Lösung:** Weights als Input-IOSurface-Channels statt BLOBFILE-Konstanten. Compile einmal, Weights per IOSurface-Write updaten.

```
Vorher:      compile(mil + weights) → 60 Compiles für 60 Steps
Nachher:     compile(mil_dynamic) → 1 Compile für ∞ Steps
```

**Implementierung:**
- `ane_mil_linear_dynamic()` — MIL mit Weights-als-Input (slice → reshape → matmul)
- `ane_write_dynamic_weights()` — packt W[out][in] ins korrekte IOSurface-Layout
- Input: `[1, in_ch + in_ch*out_ch, 1, seq]` — Aktivierungen + Weights zusammen
- Beide Training-Demos (`demo_train.c`, `generate.c`) umgestellt

**Ergebnis:** Compile count 1 statt 60. ~119 Limit komplett umgangen. 0.1ms/step.

**RE-Erkenntnis:** Delta Compilation (Disk-Patching nach Unload/Reload) funktioniert NICHT — ANE bakt Weights ins HWX und liest sie nicht erneut von Disk. `ane_reload_weights()` bleibt als EXPERIMENTAL API erhalten.

---

### 2. FP16 Overflow Protection — DONE

**Problem:** FP16 max = 65504. Softmax/RMSNorm können Overflow verursachen → NaN → Training divergiert.

**Lösung:** Aktivierungen clampen, Gradienten sanitizen. Implementiert in `demo_train.c` und `generate.c`.

```c
// Vor Softmax/RMSNorm:
for (int i = 0; i < n; i++)
    x[i] = fminf(fmaxf(x[i], -65504.0f), 65504.0f);

// Nach Gradient-Berechnung:
for (int i = 0; i < n; i++) {
    if (isnan(grad[i])) grad[i] = 0.0f;
    if (isinf(grad[i])) grad[i] = copysignf(65504.0f, grad[i]);
}
```

**Dateien:** `examples/demo_train.c`, `examples/generate.c`

---

### 3. Process-Restart bei ~119 Compilations — OBSOLET (durch DSP gelöst)

**Problem:** ANE erlaubt ~119 Compilations pro Prozess, dann stille Fehler.

**Lösung:** Compilation-Counter + `execl()` mit Resume-Flag.

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

**Hinweis:** Dynamic Spatial Packing umgeht das Problem bereits (1x compile, ∞ weight-updates). Trotzdem als Safety-Net implementieren.

**Dateien:** `libane/ane.m` (Counter), `examples/demo_train.c` (Checkpoint/Resume)

---

### 4. Zero-Copy für Weight-Updates — DONE (Teil von DSP)

**Problem:** `ane_write()` macht Lock → memcpy → Unlock (55-65µs pro Call).

**Lösung:** Direkt in IOSurface schreiben via `ane_input_ptr()`.

```c
// Statt:
ane_write(k, 0, updated_weights, bytes);

// Besser:
ane_lock_input(k, 0);
float *ptr = (float *)ane_input_ptr(k, 0);
// Nur geänderte Weights updaten (delta):
for (int i = 0; i < n_changed; i++)
    ptr[changed_idx[i]] = new_val[i];
ane_unlock_input(k, 0);
```

**Dateien:** `examples/demo_train.c`, `examples/generate.c`

---

### 5. SRAM Budget Tracking — DONE

**Problem:** M3 Pro SRAM ~16MB, Throughput-Drop ab 73.5MB, Cliff bei 129MB.

**Lösung:** Tensor-Größen tracken, warnen wenn Budget überschritten.

```c
// In ane_compile():
size_t total_io = 0;
for (int i = 0; i < n_inputs; i++) total_io += input_sizes[i];
for (int i = 0; i < n_outputs; i++) total_io += output_sizes[i];
if (total_io > 16 * 1024 * 1024)
    fprintf(stderr, "libane: warning: total I/O %zuMB exceeds SRAM (~16MB)\n", total_io >> 20);
```

**Dateien:** `libane/ane.m`

---

### 6. Alphabetische IOSurface-Sortierung

**Problem:** Multi-Input/Output Surfaces müssen alphabetisch nach MIL-Variablenname sortiert sein. Falsche Reihenfolge = stille Fehler.

**Quelle:** Orion Constraint #3, #19

**Dateien:** `libane/ane.m` (Sortierung in `ane_compile()`)

---

## P1 — Mittelfristig (architektonische Änderungen)

### 7. Pipeline-Parallelismus (30-40% Latenz)

**Problem:** CPU Backward (30ms) blockiert während ANE idle ist.

**Lösung:** Backward Step N parallel zum Forward Step N+1.

```
Aktuell (sequentiell):
  ANE forward [22ms] → CPU backward [30ms] → ANE forward [22ms] → ...
  Total: 52ms/step

Pipeline:
  ANE forward N [22ms] → ANE forward N+1 [22ms]
  CPU backward N [30ms]  ← parallel!
  Total: ~30ms/step (ANE ist Bottleneck statt CPU)
```

**Implementierung:** Async ANE eval via `_ANEClient::enqueueSetsWithModel:` + Completion-Handler.

---

### 8. LoRA Adapter-as-Input

**Problem:** Weight-Updates erfordern Recompilation oder Delta-Compilation.

**Lösung:** Base-Weights in BLOBFILE (compile once), LoRA-Adapter (A, B Matrizen) als IOSurface-Input.

```
MIL:
  base_W = file_value("@model_path/weights/base.bin")  // baked
  lora_A = input("lora_a", [rank, in_dim])              // IOSurface
  lora_B = input("lora_b", [out_dim, rank])              // IOSurface
  W_eff  = base_W + matmul(lora_B, lora_A)
  output = conv(input, W_eff)
```

**Vorteil:** Zero Recompilation für Fine-Tuning. Hot-Swap von Adaptern.

**Quelle:** Orion Section 4.3

---

### 9. Kernel Chaining (`_ANEChainingRequest`)

**Problem:** Jede Kernel-Eval hat CPU-Roundtrip (~0.2ms). Bei 12-Layer Transformer: 22 Roundtrips = 4.4ms Overhead.

**Status:** Objekt-Erstellung funktioniert, `validate()` schlägt fehl (Parameter-Typ-Mismatch).

**Nächste Schritte:**
1. NSNumber-Parameter die eigentlich Arrays sein sollten identifizieren
2. `_ANEBuffer` Objekte für Zwischen-Tensoren korrekt aufbauen
3. `_ANEClient::prepareChainingWithModel:chainingReq:qos:error:` testen

**Geschätzter Gain:** 3-5x Forward-Pass Speedup (bei tiefem Netzwerk).

---

### 10. RMSNorm auf ANE

**Problem:** 24 RMSNorm-Operationen pro Step (2 pro Layer × 12 Layer) laufen auf CPU (~4-6ms).

**Lösung:** MIL-Programm für RMSNorm:

```
reduce_mean(square(x)) → add(eps) → rsqrt → mul(x) → mul(gamma)
```

Alle Ops sind ANE-kompatibel. Als separater Kernel kompilieren oder in bestehende Kernel fusionieren.

---

### 11. Causal Masking via `where()` auf ANE

**Problem:** Attention läuft auf CPU weil ANE kein Causal Masking kann.

**Lösung:** MIL `where()` Operator:

```
causal_mask = constant([1, heads, seq, seq])  // Lower-triangular, baked
scores = matmul(Q, K_T) / sqrt(d)
masked = where(causal_mask, scores, constant(-65504.0))
attn = softmax(masked, axis=-1)
output = matmul(attn, V)
```

**Einschränkung:** Causal Mask ist fix pro Sequenzlänge. Bei variabler Länge: mehrere Kernel vorcompilieren.

---

## P2 — Langfristig (Hybrid ANE + GPU)

### 12. ANE Forward + GPU Backward

```
┌─────────────────────────────┐
│  ANE (libane)               │
│  Linear, Conv, Embedding    │
│  Output → IOSurface         │
└────────────┬────────────────┘
             │ Zero-Copy (Unified Memory)
             ▼
┌─────────────────────────────┐
│  GPU (MLX / MPSGraph)       │
│  Attention, Softmax, RoPE   │
│  Backward Pass (Autodiff)   │
│  Adam Optimizer              │
└────────────┬────────────────┘
             │ Weight Update
             ▼
┌─────────────────────────────┐
│  Delta Compilation / LoRA   │
│  Weights → ANE              │
└─────────────────────────────┘
```

**Voraussetzung:** IOSurface-zu-Metal-Buffer Zero-Copy Mapping.

**Geschätzter Gain:** 2-3x Gesamt-Speedup.

---

### 13. GPU FlashAttention

Metal FlashAttention 2.0 existiert (Draw Things Engineering). Für lange Sequenzen (>512 Tokens) signifikant schneller als CPU-Attention.

**Gain:** 3-5x für Attention-Berechnung.

---

### 14. M5 GPU Neural Accelerators nutzen

M5 hat drei ML-Beschleuniger:
- **ANE** (16-core, ~38+ TOPS) — via libane
- **GPU Neural Accelerators** (10-40 pro Chip, ~95 TOPS auf M5 Pro) — via Metal 4 TensorOps
- **CPU** (NEON/AMX)

Für maximale Performance: ANE für Forward, GPU-NAs für Backward.

**Voraussetzung:** M5 Hardware + Metal 4 SDK.

---

## Neue Projekte & Quellen

| Projekt | Relevanz |
|:---|:---|
| [Orion Paper](https://arxiv.org/abs/2603.06728) | Delta Compilation, LoRA-as-Input, 20 Constraints |
| [NeuralForge](https://github.com/Khaeldur/NeuralForge) | Process-Restart, GGUF-Export, Gradient Accumulation |
| [ANEMLL](https://github.com/Anemll/Anemll) | ANE ML Library |
| [SqueezeBits Yetter](https://blog.squeezebits.com/disaggregated-inference-on-apple-silicon-npu-prefill-and-gpu-decode-67176) | Disaggregated Inference (ANE Prefill + GPU Decode) |
| [Metal FlashAttention 2.0](https://engineering.drawthings.ai/) | GPU Attention |
| [Apple MLX M5 Research](https://machinelearning.apple.com/research/exploring-llms-mlx-m5) | GPU Neural Accelerator Benchmarks |
| [eiln/ane](https://github.com/eiln/ane) | E5 Binary Format Analyse |
| [tzakharko M5 Microbenchmarks](https://tzakharko.github.io/apple-neural-accelerators-benchmark/) | GPU-NA Tile-Size, FLOPS/Core |

---

## Aktuelle Bottleneck-Analyse

```
Training Step (91ms, M3 Pro, Stories110M):

  ANE Forward      ████████████░░░░░░░░░░░░  22ms (24%)
  CPU Attention     ███░░░░░░░░░░░░░░░░░░░░░   5ms  (5%)
  ANE Backward     ██████████████░░░░░░░░░░  30ms (33%)  ← teilweise CPU (dW)
  CPU RMSNorm      ███░░░░░░░░░░░░░░░░░░░░░   5ms  (5%)
  CPU dW Gradients ████████████████████░░░░  20ms (22%)  ← CBLAS matmul
  CPU Adam Update  ██░░░░░░░░░░░░░░░░░░░░░░   3ms  (3%)
  Overhead         ██████░░░░░░░░░░░░░░░░░░   6ms  (7%)

  ANE: 41% | CPU: 59% ← CPU ist Bottleneck
```

**Nach P0+P1 Optimierungen (geschätzt):**

```
  ANE Forward      ████████████░░░░░░░░░░░░  22ms
  ANE Attention    ███░░░░░░░░░░░░░░░░░░░░░   3ms  ← war CPU
  ANE Backward     ██████████████░░░░░░░░░░  15ms  ← parallel
  CPU dW (parallel)████████████░░░░░░░░░░░░   0ms  ← hidden behind ANE
  Delta Reload     ████░░░░░░░░░░░░░░░░░░░░   5ms  ← war 4200ms

  Geschätzter Step: ~30ms (3x schneller)
```
