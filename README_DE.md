<div align="center">

```
     ___    _   __ ______
    /   |  / | / // ____/
   / /| | /  |/ // __/
  / ___ |/ /|  // /___
 /_/  |_/_/ |_//_____/  Training
```

### Reverse-Engineering Apples Neural Engine

**1 Compile · 9.4 TFLOPS · 35 Private Klassen · Zero Recompilation**

[![License: MIT](https://img.shields.io/badge/Lizenz-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/macOS_15+-111111.svg?logo=apple&logoColor=white)](https://www.apple.com/macos/)
[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-M1--M5-FF3B30.svg)](https://support.apple.com/en-us/116943)
[![Compile](https://img.shields.io/badge/Compile-1x_for_∞_Steps-34C759.svg)](#dynamic-spatial-packing--der-durchbruch)
[![TFLOPS](https://img.shields.io/badge/Peak-9.4_TFLOPS-007AFF.svg)](#benchmark-ergebnisse-m3-pro)
[![Classes](https://img.shields.io/badge/ANE_Klassen-35-FF9500.svg)](RESEARCH_ANE_COMPLETE.md)

</div>

<br>

> Die erste eigenständige C-API (`libane`) für Apples privaten Neural Engine. Ermöglicht **Training** direkt auf dem ANE — Apple beschränkt ihn offiziell auf Inference via CoreML. Vollständige Hardware-Forschung, lauffähige Demos, Benchmark-Suite.

---

<table>
<tr>
<td width="50%">

**Entdeckungen**

| | |
|:--|:--|
| `35` | Private API-Klassen entdeckt (bekannt: nur 4) |
| `42%` | Schneller mit QoS Background statt Default |
| `9.4` | TFLOPS Peak auf M3 Pro (FP16) |
| `1` | Compile reicht — unbegrenzt viele Training-Steps |
| `3x` | Conv 1x1 schneller als matmul auf ANE |

</td>
<td width="50%">

**libane — 73KB C-API**

| | |
|:--|:--|
| Dynamic | Weights via IOSurface, zero recompile |
| Zero-Copy | I/O direkt in ANE-Speicher |
| Version-Detection | Überlebt Apple API-Änderungen |
| 6 QoS-Level | Background (9) bis Realtime (0) |
| FP16 Safe | Overflow Protection eingebaut |

</td>
</tr>
</table>

> [!NOTE]
> **Compilation-Pipeline:** `MIL → MLIR → LLIR → HWX` — siehe [Glossar](#glossar) für alle Fachbegriffe.

---

## Voraussetzungen

| Was | Minimum | Getestet mit |
|:---|:---|:---|
| Mac | Apple Silicon (M1–M5) | M3 Pro |
| macOS | 15+ | 26.3.1 (Build 25D2128) |
| Xcode CLI Tools | Erforderlich | `xcode-select --install` |

```bash
uname -m           # → "arm64"
xcode-select -p    # → /Library/Developer/CommandLineTools
```

> [!WARNING]
> **Intel Mac?** Dieses Projekt funktioniert **nur** auf Apple Silicon. Der Neural Engine existiert nur in M-Serie Chips.

---

## Installation

<table>
<tr>
<td width="50%">

**Option A — One-Liner**

```bash
curl -sSL https://raw.githubusercontent.com/\
slavko-at-klincov-it/ANE-Training/\
main/install.sh | bash
```

Prüft Voraussetzungen, klont, baut, benchmarked.

</td>
<td width="50%">

**Option B — Manuell**

```bash
git clone https://github.com/slavko-at-klincov-it/ANE-Training.git
cd ANE-Training
./ane
```

Interaktives Menü — baut alles automatisch.

</td>
</tr>
</table>

> [!TIP]
> `./ane` erkennt deine Hardware, baut alle Binaries beim ersten Start, und führt dich durch alles. Du brauchst keine Makefiles oder Pfade zu kennen.

---

## Schnellstart

```bash
./ane                # Interaktives Menü
./ane train          # Training Demo (Y=2X, 1 Compile, 60 Steps)
./ane bench          # Auto-Benchmark (TFLOPS + Chip-Vergleich)
./ane generate       # Shakespeare Text-Generation auf ANE
./ane explore        # 35 ANE-Klassen interaktiv erkunden
./ane info           # Hardware-Erkennung
./ane test           # libane Test-Suite
```

<details open>
<summary><b>Training Demo</b> — <code>make demo</code></summary>

&nbsp;

Trainiert einen Linear-Layer direkt auf dem ANE mit **Dynamic Spatial Packing** — kompiliert einmal, trainiert 60 Steps ohne Recompilation. Forward auf Neural Engine, Backward + SGD auf CPU.

```
Hardware: h15g, 16 ANE cores
Compiled once (dynamic weights, no recompilation needed)

Goal: Train W so that Y = W @ X approximates Y = 2*X

step   loss       W[0,0]   W[1,1]   ms/step
0        1.3493    0.148    0.241   0.4
5        0.5334    0.847    0.868   0.1
10       0.2304    1.260    1.254   0.1
30       0.0164    1.841    1.825   0.1
59       0.0010    1.975    1.967   0.3

Diagonal average: 1.959 (converged!)
Compile count: 1 / 119 budget
```

**1 Compilation statt 60.** Weights werden per IOSurface-Write aktualisiert, nicht per Recompile. 0.1ms/step.

</details>

<details>
<summary><b>Auto-Benchmark</b> — <code>make bench</code></summary>

&nbsp;

Erkennt deinen Chip, misst TFLOPS, zeigt Vergleich:

```
Chip:   h15g (M3 Pro), 16 cores

---- Single Conv Sweep (1x1 conv, ch x ch) ----
256x256 sp64     0.1 MB   2.10  0.284 ms    7.38
4096x4096 sp64  32.0 MB  34.36  3.841 ms    8.94

---- Peak Sustained (Stacked Conv) ----
128x stacked     32.0 MB  34.36  3.647 ms    9.42

---- Performance Overview ----
>> h15g (M3 Pro)        9.42 TFLOPS  ████████████████████████████░░
   h16g (M4)           11.00 TFLOPS  █████████████████████████████░
```

</details>

<details>
<summary><b>Text Generation</b> — <code>make generate</code></summary>

&nbsp;

Bigram-Modell auf Shakespeare, Typewriter-Ausgabe. Kompiliert einmal, trainiert + generiert ohne Recompile:

```
Compiled once (dynamic weights, no recompilation needed)

Training bigram model on Shakespeare...
step   loss      perplexity
0       4.1589   64.00
29      3.1245   22.76

Generating text (200 chars, temperature=0.8)...
To be or not to be, that is the question...

Compiles used: 1 / 119
```

</details>

<details>
<summary><b>ANE Explorer</b> — <code>make explore</code></summary>

&nbsp;

Alle 35 ANE-Klassen kategorisiert, interaktive Inspektion:

```
Found 35 ANE classes

┌─ Core (Model compilation, loading, evaluation)
│  █ _ANEInMemoryModel
│  █ _ANEInMemoryModelDescriptor
│  █ _ANERequest
└─

Interactive Mode: Enter a class name to inspect
> _ANEInMemoryModel
Instance Methods (23):
  - compileWithQoS:options:error:
  - loadWithQoS:options:error:
  ...
```

</details>

---

## Dynamic Spatial Packing — Der Durchbruch

> [!IMPORTANT]
> **Das Problem:** Der ANE bakt Weights bei Compilation ins HWX-Binary. Jeder Weight-Update braucht eine Recompilation (~520ms). Hartes Limit: **~119 Compilations pro Prozess**, dann stille Fehler.
>
> **Die Lösung:** Weights nicht mehr ins Binary baken, sondern als **Input-Channels** neben den Aktivierungen packen. MIL-Code sliced sie auseinander → matmul. **1x kompilieren, unbegrenzt trainieren.**

<table>
<tr>
<td width="50%">

**Vorher — Standard**
```
Weights ──→ BLOBFILE ──→ Compile
Input   ──→ IOSurface ──→ Eval

Pro Step:  1 Compile + 1 Eval
Budget:    119 Steps max
Latency:   0.3–0.5 ms/step
```

</td>
<td width="50%">

**Nachher — Dynamic Spatial Packing**
```
Weights ──→ IOSurface ──→ Write
Input   ──→ IOSurface ──→ Eval

Pro Step:  1 Write + 1 Eval
Budget:    ∞ Steps
Latency:   0.1 ms/step
```

</td>
</tr>
</table>

<details>
<summary><b>Wie funktioniert das im Detail?</b></summary>

&nbsp;

**Input-Layout:** `[1, in_ch + in_ch*out_ch, 1, seq]`
- Channels `[0..in_ch)`: Aktivierungen (Trainingsdaten)
- Channels `[in_ch..in_ch+in_ch*out_ch)`: Weight-Matrix (flattened), nur Spatial-Position 0

**MIL-Programm (generiert von `ane_mil_linear_dynamic()`):**
```
1. Cast FP32 → FP16
2. slice_by_size → Aktivierungen [1, in_ch, 1, seq]
3. slice_by_size → Weights [1, in_ch*out_ch, 1, 1]
4. reshape → Weights [1, 1, out_ch, in_ch]
5. reshape + transpose → Aktivierungen [1, 1, seq, in_ch]
6. matmul(activations, weights^T) → [1, 1, seq, out_ch]
7. transpose + reshape → [1, out_ch, 1, seq]
8. Cast FP16 → FP32
```

**Neue libane API:**
```c
// MIL mit Weights-als-Input generieren
char *mil = ane_mil_linear_dynamic(in_ch, out_ch, seq);

// EINMAL kompilieren — keine Weights nötig
ANEKernel *k = ane_compile(mil, strlen(mil), NULL, 0,
                           1, &in_bytes, 1, &out_bytes, ANE_QOS_BACKGROUND);

// Training-Loop: nur IOSurface updaten, nie recompile
for (int step = 0; step < 10000; step++) {
    ane_write_dynamic_weights(k, 0, W, in_ch, out_ch, seq);
    ane_lock_input(k, 0);
    float *ptr = (float *)ane_input_ptr(k, 0);
    // Aktivierungen in Channels 0..in_ch schreiben
    ane_unlock_input(k, 0);
    ane_eval(k, ANE_QOS_BACKGROUND);
    // ... backward + SGD ...
}
```

</details>

<details>
<summary><b>RE-Ergebnisse: Was NICHT funktioniert hat</b></summary>

&nbsp;

Vor Dynamic Spatial Packing haben wir 5 andere Ansätze getestet:

| Ansatz | Ergebnis | Grund |
|:---|:---|:---|
| Disk-Patch + Unload/Reload | Weights nicht aktualisiert | ANE bakt Weights beim Compile ins HWX |
| In-Memory VM-Patch | Keine Wirkung | ANE nutzt SRAM-Kopie, ignoriert RAM-Patches |
| `_ANEWeight` Klasse | Request erstellt, Weights nicht angewandt | Unbekanntes Binding |
| `weightsBuffer` in `_ANERequest` | Akzeptiert, keine Wirkung | ANE ignoriert Runtime-Weights |
| `ane_reload_weights()` | MIL wird gelöscht | Unload löscht tmpDir |

**Nur Dynamic Spatial Packing funktioniert** — Weights als Input-Daten, nicht als Compile-Time-Konstanten.

</details>

---

## Eigenen Code mit libane

<details>
<summary><b>Statische Weights (Inference)</b> — <code>my_test.c</code></summary>

&nbsp;

```c
#include <stdio.h>
#include <string.h>
#include "ane.h"

int main() {
    ane_init();
    ANEDeviceInfo info = ane_device_info();
    printf("ANE: %s, %d cores\n", info.arch, info.num_cores);

    // Gewichte vorbereiten (2x2 Einheitsmatrix)
    float weights[] = {1.0f, 0.0f, 0.0f, 1.0f};
    ANEWeight w = ane_weight_fp16("@model_path/weights/w.bin", weights, 2, 2);

    // MIL generieren + kompilieren
    char *mil = ane_mil_linear(2, 2, 1, "@model_path/weights/w.bin");
    size_t in_sz = 2 * sizeof(float), out_sz = 2 * sizeof(float);
    ANEKernel *k = ane_compile(mil, strlen(mil), &w, 1,
                               1, &in_sz, 1, &out_sz, ANE_QOS_BACKGROUND);

    // Ausführen
    float input[] = {3.0f, 7.0f}, output[2];
    ane_write(k, 0, input, in_sz);
    ane_eval(k, ANE_QOS_BACKGROUND);
    ane_read(k, 0, output, out_sz);

    printf("Input:  [%.1f, %.1f]\n", input[0], input[1]);
    printf("Output: [%.1f, %.1f]\n", output[0], output[1]);

    ane_free(k); free(mil); ane_weight_free(&w);
}
```

```bash
xcrun clang -O2 -fobjc-arc -I libane -o my_test my_test.c libane/ane.m \
    -framework Foundation -framework IOSurface -ldl
```

</details>

<details>
<summary><b>Dynamische Weights (Training)</b> — Compile-Once Pattern</summary>

&nbsp;

```c
#include <string.h>
#include "ane.h"

int in_ch = 8, out_ch = 8, seq = 64;

// 1. MIL mit dynamischen Weights generieren
char *mil = ane_mil_linear_dynamic(in_ch, out_ch, seq);

// 2. EINMAL kompilieren — keine Weights nötig
size_t in_sz = (in_ch + in_ch * out_ch) * seq * sizeof(float);
size_t out_sz = out_ch * seq * sizeof(float);
ANEKernel *k = ane_compile(mil, strlen(mil), NULL, 0,
                           1, &in_sz, 1, &out_sz, ANE_QOS_BACKGROUND);

// 3. Training-Loop — Weights per IOSurface-Write, nie recompile
float W[8 * 8];
for (int step = 0; step < 10000; step++) {
    // Weights + Aktivierungen packen
    ane_write_dynamic_weights(k, 0, W, in_ch, out_ch, seq);
    ane_lock_input(k, 0);
    float *ptr = (float *)ane_input_ptr(k, 0);
    // ptr[0..in_ch*seq] = Aktivierungen
    ane_unlock_input(k, 0);

    ane_eval(k, ANE_QOS_BACKGROUND);
    // ... backward + SGD auf W ...
}

ane_free(k); free(mil);
```

</details>

Ausführliche API-Dokumentation: **[libane/README.md](libane/README.md)**

---

## Benchmark-Ergebnisse (M3 Pro)

| Metric | Wert | |
|:---|---:|:---|
| Peak FP16 | **9.36 TFLOPS** | Maximale Rechenleistung (kleine Tensoren) |
| Peak FP16 (large spatial) | **18.23 TOPS** | Große Tensor-Dimensionen |
| INT8 Speedup | **1.0–1.14x** | Lohnt nicht auf M3 Pro (M4: 1.88x) |
| Training Stories110M | **91–183 ms/step** | Je nach Vocab-Größe |
| Kernel-Compilation | **520 ms** | Einmalig beim Start (10 Kernel) |
| QoS Background vs Default | **42% schneller** | Weniger Scheduling-Overhead |

<details>
<summary><i>TFLOPS vs TOPS — was ist der Unterschied?</i></summary>

&nbsp;

**TFLOPS** = Tera Floating-Point Operations Per Second — zählt multiply+add als 2 Ops.<br>
**TOPS** = Tera Operations Per Second — zählt jede Operation einzeln.<br>
Deswegen kann TOPS höher sein als TFLOPS bei gleicher Hardware.

</details>

---

## Chip-Vergleich — Alle Apple Silicon Generationen

> [!NOTE]
> **Der ANE ist bei base/Pro/Max identisch.** Innerhalb einer Generation: selber 16-Core Neural Engine. Pro/Max bringen nur mehr GPU + Bandwidth. **Nur Ultra verdoppelt den ANE** (32 Cores). Memory-Bandwidth hilft dem ANE **nicht**, solange Tensoren im ~32MB SRAM bleiben.

| Chip | Arch | ANE TOPS | Mem BW | TFLOPS\* | INT8 | SRAM |
|:---|:---|---:|---:|---:|:---|---:|
| **M1** | H13 | 11 | 68 GB/s | ~5.5 | nur Weights | ~32 MB |
| M1 Pro | H13 | 11 | 200 GB/s | ~5.5 | nur Weights | ~32 MB |
| M1 Max | H13 | 11 | 400 GB/s | ~5.5 | nur Weights | ~32 MB |
| M1 Ultra | H13 x2 | **22** | 800 GB/s | ~11 | nur Weights | ~64 MB |
| | | | | | | |
| **M2** | H14 | 15.8 | 100 GB/s | ~8 | nur Weights | ~32 MB |
| M2 Pro | H14 | 15.8 | 200 GB/s | ~9.0 | nur Weights | ~32 MB |
| M2 Max | H14 | 15.8 | 400 GB/s | ~9.2 | nur Weights | ~32 MB |
| M2 Ultra | H14 x2 | **31.6** | 800 GB/s | ~18 | nur Weights | ~64 MB |
| | | | | | | |
| **M3** | H15 | 18 | 100 GB/s | ~9.4 | nur Weights | ~32 MB |
| **M3 Pro** | H15 (h15g) | 18 | 150 GB/s | **9.4** | 1.0–1.14x | ~32 MB |
| M3 Max | H15 (h15p) | 18 | 300–400 GB/s | ~9.5 | nur Weights | ~32 MB |
| M3 Ultra | H15 x2 | **36** | 819 GB/s | ~19 | nur Weights | ~64 MB |
| | | | | | | |
| **M4** | H16 (h16g) | **38** | 120 GB/s | ~11 | **1.88x (W8A8)** | ~32 MB |
| M4 Pro | H16 (h16p) | **38** | 273 GB/s | ~12 | **1.88x (W8A8)** | ~32 MB |
| M4 Max | H16 | **38** | 546 GB/s | ~11 | **1.88x (W8A8)** | ~32 MB |
| | | | | | | |
| **M5** | — | n/a | 153 GB/s | ~12–14† | TBD | ~32 MB |
| M5 Pro | — | n/a | 307 GB/s | ~12–14† | TBD | ~32 MB |
| M5 Max | — | n/a | 614 GB/s | ~12–14† | TBD | ~32 MB |

<sub>\* Gemessene FP16 TFLOPS auf ANE (Conv 1x1). M2/M4-Werte aus maderix/ANE Benchmarks.</sub><br>
<sub>† M5-ANE-Schätzung basierend auf M3→M4 Trend. Apple hat keine separaten ANE-TOPS veröffentlicht.</sub>

<details>
<summary><b>Was bedeutet "INT8 nur Weights" vs "W8A8"?</b></summary>

&nbsp;

**M1/M2/M3 (INT8 nur Weights):** Der ANE lädt INT8-Weights, aber **rechnet in FP16**. Das spart Speicher und Bandwidth, gibt aber **keinen Compute-Speedup**. Deswegen bringt INT8 auf M3 Pro nur 1.0–1.14x.

**M4+ (W8A8 = Weights AND Activations INT8):** Erst ab M4 (H16) können **beide** — Weights und Aktivierungen — in INT8 verarbeitet werden. Das verdoppelt den effektiven Durchsatz: **1.88x Speedup**. INT8-Training wird damit realistisch.

</details>

<details>
<summary><b>Geschätzte Training-Performance pro Chip</b></summary>

&nbsp;

Basierend auf Stories110M (124 Vocab, compacted), gemessen auf M3 Pro = 91ms/step:

| Chip | Geschätzt ms/step | INT8 ms/step | Faktor vs M3 Pro |
|:---|---:|---:|:---|
| M1 | ~160 ms | — | 0.6x |
| M2 | ~105 ms | — | 0.9x |
| **M3 Pro** | **91 ms** | — | _Baseline_ |
| M3 Ultra | ~50 ms | — | 1.8x |
| **M4** | **~45–55 ms** | **~25–30 ms** | **2x (4x mit INT8)** |
| M4 Pro | ~45–55 ms | ~25–30 ms | 2x |
| M4 Max | ~40–50 ms | ~22–28 ms | 2–2.5x |
| M5 | ~35–45 ms | TBD | ~2.5x |
| M5 Pro | ~35–45 ms | TBD | ~2.5x |
| M5 Max | ~30–40 ms | TBD | ~3x |

</details>

<details>
<summary><b>Optimierungstipps pro Generation</b></summary>

&nbsp;

**M1 (11 TOPS)** — Kleinster Durchsatz. Tensor-Tiling ist kritisch — halte Working Sets unter 24MB. FP16 nutzen, INT8 bringt nichts. QoS Background (9) bringt hier den größten relativen Gewinn.

**M2 (15.8 TOPS)** — 44% schneller als M1, gleiche Constraints. Kein INT8-Compute-Vorteil. Gleiche SRAM-Limits (~32MB).

**M3 / M3 Pro (18 TOPS)** — 14% über M2. Kein W8A8. Besonders flexible SRAM-Verwaltung (kein harter Cliff, gradueller Drop bis 73.5MB). Optimiere auf FP16, Conv 1x1 statt matmul.

**M4 (38 TOPS)** — **Der große Sprung**: 2x TOPS, echtes W8A8 INT8. Training-Steps halbieren sich. INT8-Quantisierung lohnt sich erstmals. Gleicher ~32MB SRAM.

**M5 (Fusion Architecture)** — ANE selbst leicht schneller, aber die GPU Neural Accelerators (10–40 pro Chip) sind der Game-Changer. Via Metal/MLX nutzbar, nicht via libane. Für maximale Performance: Hybrid ANE+GPU Ansatz.

**Ultra-Varianten (M1–M3)** — Doppelter ANE (32 Cores, 2x TOPS). Einzige Variante wo Pro/Max → Ultra einen echten ANE-Unterschied macht. Beide Dies arbeiten parallel — ideal für größere Modelle die auf 16 Cores nicht passen.

</details>

> [!IMPORTANT]
> Die GPU Neural Accelerators im M5 sind über **Metal/MLX** erreichbar, **nicht** über die privaten ANE-APIs die `libane` nutzt. Für `libane`-User zählt primär der schnellere Neural Engine + höhere Memory-Bandwidth. Für die vollen ~133 TOPS braucht man Apples MLX-Framework — das ist offiziell und stabil.

---

## Hardware-Constraints & Bottlenecks

### Bekannte ANE-Limits

| Constraint | Wert | Auswirkung |
|:---|:---|:---|
| **SRAM On-Chip** | ~32 MB (alle Generationen) | Tensoren >32MB spillen in DRAM → 30% Throughput-Drop |
| **Compilation-Limit** | ~119 pro Prozess | Danach stille Fehler. Gelöst durch Dynamic Spatial Packing |
| **IOSurface Minimum** | ~49 KB | Kleinere Tensoren müssen gepaddet werden |
| **IOSurface Sortierung** | Alphabetisch nach MIL-Name | Falsche Reihenfolge = stille Fehler |
| **`concat` Op** | Wird abgelehnt | Muss in separate Programme aufgeteilt werden |
| **`gelu` Op** | Nicht unterstützt | tanh-Approximation nutzen |
| **Conv 1x1 vs matmul** | Conv ist 3x schneller | Alle matmuls als 1x1 Conv ausdrücken |
| **FP16 Overflow** | max ±65504 | Aktivierungen clampen vor Softmax/RMSNorm |
| **Causal Masking** | Nicht nativ | `where()` MIL-Op als Workaround möglich |

<sub>Quellen: Eigene Forschung + [Orion Paper](https://arxiv.org/abs/2603.06728) (20 dokumentierte Constraints)</sub>

### Bottleneck-Analyse (M3 Pro, Stories110M)

```
Training Step = 91ms:

  ANE Forward          ██████░░░░░░░░░░░░░░  22ms  (24%)
  ANE Backward (dx)    ████████░░░░░░░░░░░░  15ms  (16%)
  CPU dW Gradients     ██████████░░░░░░░░░░  20ms  (22%)  ← Bottleneck
  CPU Attention/RoPE   ██████░░░░░░░░░░░░░░   8ms   (9%)
  CPU RMSNorm          ███░░░░░░░░░░░░░░░░░   5ms   (5%)
  CPU Adam Update      ██░░░░░░░░░░░░░░░░░░   3ms   (3%)
  Overhead             █████████░░░░░░░░░░░  18ms  (20%)

  ANE: 41%  ·  CPU: 59%
```

### Implementierte Optimierungen

| | Optimierung | Ergebnis |
|:---|:---|:---|
| **done** | **Dynamic Spatial Packing** — Weights als IOSurface-Input | 60 Compiles → **1 Compile**, ~119 Limit umgangen |
| **done** | **FP16 Overflow Protection** — Output/Gradient-Sanitierung | Verhindert NaN/Inf-Divergenz |
| **done** | **SRAM Budget Tracking** — Warnung bei >32MB | Diagnostik in `ane_compile()` |
| **done** | **Compile Budget Warning** — Warnung bei 110/119 | Safety-Net für Legacy-Code |

### Offene Optimierungen

| Optimierung | Gain | Aufwand |
|:---|:---|:---|
| **Pipeline-Parallelismus** — CPU Backward ‖ ANE Forward | ~40% Latenz | Hoch |
| **Attention auf ANE** — via `where()` MIL-Op | ~5ms CPU-Ersparnis | Hoch |
| **RMSNorm auf ANE** — als MIL-Programm | ~5ms CPU-Ersparnis | Mittel |
| **LoRA Adapter-as-Input** — Hot-Swap Fine-Tuning | Zero Recompile | Mittel |

> Vollständiger Optimierungsplan: **[ROADMAP.md](ROADMAP.md)**

---

## Projektstruktur

```
ANE-Training/
│
├── ane ······························· CLI Entry-Point (./ane)
├── README.md ·························· Dieses Dokument
├── ARCHITECTURE.md ···················· 4-Schichten Platform-Architektur
├── ROADMAP.md ························· Optimierungsplan (P0 erledigt)
├── RESEARCH_ANE_COMPLETE.md ··········· Vollständige Forschungsdoku
├── SUMMARY_TECHNICAL.md ·············· Technische Zusammenfassung
├── SUMMARY_SIMPLE.md ················· Nicht-technische Zusammenfassung
├── LICENSE ···························· MIT
├── install.sh ························· One-Liner Installer
│
├── examples/ ·························· Lauffähige Demos
│   ├── demo_train.c                     Training Demo (Dynamic Spatial Packing)
│   ├── bench.c                          Auto-Benchmark
│   ├── generate.c                       Text Generation (Dynamic Spatial Packing)
│   ├── explore.m                        ANE Explorer
│   └── Makefile
│
└── libane/ ···························· Unsere C-API
    ├── ane.h                            Stabile API (ändert sich nie)
    ├── ane.m                            Implementierung + Version-Detection
    ├── test_ane.c                       Test-Suite (3/3 bestanden)
    ├── README.md                        API-Dokumentation
    └── Makefile
```

---

<details>
<summary><h2>Glossar</h2></summary>

&nbsp;

| Begriff | Erklärung |
|:---|:---|
| **ANE** | Apple Neural Engine — KI-Beschleuniger in Apple Silicon |
| **MIL** | Model Intermediate Language — Apples Textformat für Modelle |
| **MLIR** | Multi-Level Intermediate Representation — Compiler-Zwischenformat |
| **LLIR** | Low-Level IR — maschinennahe Darstellung vor Kompilierung |
| **HWX** | Hardware Executable — finales Binary für den ANE |
| **IOSurface** | macOS Zero-Copy Shared Memory zwischen CPU und ANE |
| **QoS** | Quality of Service — Prioritätsstufe für ANE-Berechnungen |
| **TFLOPS** | Tera Floating-Point Ops/Sek (10¹² FP-Ops) |
| **TOPS** | Tera Operations/Sek (zählt jede einzelne Op) |
| **FP16** | 16-bit Float — natives Rechenformat des ANE |
| **Conv 1x1** | 1x1 Convolution — 3x schneller als matmul auf ANE |
| **Dynamic Spatial Packing** | Weights als IOSurface-Input statt BLOBFILE → 1x Compile, ∞ Training-Steps |

</details>

<details>
<summary><h2>Troubleshooting</h2></summary>

&nbsp;

### `ane_init()` gibt -1 zurück — Framework nicht gefunden

```bash
ls /System/Library/PrivateFrameworks/AppleNeuralEngine.framework/
```

Wenn leer: macOS zu alt oder Apple hat den Pfad geändert.

### `ane_init()` gibt -2 zurück — Klassen nicht gefunden

Apple hat private Klassen umbenannt. `ane_init()` listet alle gefundenen ANE-Klassen auf stderr. Neue Namen in `libane/ane.m` eintragen, neu kompilieren.

### Compile-Fehler: `framework not found`

```bash
xcode-select --install
# oder:
sudo xcode-select --reset
```

### `uname -m` zeigt `x86_64`

Intel Mac oder Rosetta. ANE gibt es nur auf Apple Silicon:

```bash
sysctl sysctl.proc_translated 2>/dev/null
# 1 = Rosetta, 0 oder Fehler = nativ
```

### Niedrigere TFLOPS als erwartet

- Keine anderen rechenintensiven Prozesse laufen lassen
- QoS Background (9) nutzen — 42% schneller als Default
- Erste Ausführung langsamer wegen Kernel-Compilation (520ms einmalig)

</details>

---

## Verwandte Projekte

| Projekt | Beschreibung |
|:---|:---|
| [maderix/ANE](https://github.com/maderix/ANE) | Erstes Training auf ANE — Inspiration für dieses Projekt |
| [Orion Paper](https://arxiv.org/abs/2603.06728) | Akademisches Paper: Delta Compilation, LoRA, 20 ANE Constraints |
| [NeuralForge](https://github.com/Khaeldur/NeuralForge) | On-Device LLM Fine-Tuning, Process-Restart, GGUF-Export |
| [ANEMLL](https://github.com/Anemll/Anemll) | ANE Machine Learning Library |
| [hollance/neural-engine](https://github.com/hollance/neural-engine) | Community-Dokumentation (Supported Devices, Internals) |
| [eiln/ane](https://github.com/eiln/ane) | Linux-Kernel-Driver — E5 Binary Format Analyse |
| [SqueezeBits Yetter](https://blog.squeezebits.com/) | Disaggregated Inference: ANE Prefill + GPU Decode |

---

> [!CAUTION]
> Dieses Projekt nutzt Apples **private, undokumentierte APIs**. Diese können sich mit jedem macOS-Update ändern. `libane` hat Version-Detection als Schutz — wenn Apple Klassen umbenennt, muss nur `ane.m` aktualisiert werden. Dein Code gegen `ane.h` bleibt unverändert.

<div align="center">

MIT License

</div>
