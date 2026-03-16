# ANE-Training — Apple Neural Engine Research & API

Reverse-Engineering von Apples privatem Neural Engine Framework. Enthält eine eigenständige C-API (`libane`), vollständige Hardware-Forschung, und Benchmark-Ergebnisse für M3 Pro.

## Was ist das?

Apple Silicon Chips (M1-M4) haben einen **Neural Engine (ANE)** — einen 16-Core KI-Beschleuniger mit bis zu 18 TOPS (Tera Operations Per Second). Apple beschränkt ihn offiziell auf Inference via CoreML. Dieses Projekt knackt diese Beschränkung auf und ermöglicht **Training** direkt auf dem ANE.

### Was wir entdeckt haben

- **35 private API-Klassen** (bekannte Projekte nutzen nur 4)
- **6 QoS-Level** — Background (9) ist 42% schneller als Default (21)
- **Hardware-Identität**: M3 Pro = `h15g`, 16 Cores, Board 192
- **Compilation-Pipeline**: MIL → MLIR → LLIR → HWX (siehe [Glossar](#glossar))
- **Performance**: 9.36 TFLOPS (FP16), 18.23 TOPS (large spatial)
- **INT8 lohnt nicht auf M3 Pro** (nur 1.0-1.14x, auf M4: 1.88x)
- **Conv 1x1 ist 3x schneller als matmul** auf ANE

### Was wir gebaut haben

**`libane`** — eine eigene C-API mit automatischer Version-Detection:
- Überlebt Apple API-Änderungen (probiert alternative Klassen-/Methoden-Namen)
- Device-Erkennung, QoS-Support, Zero-Copy I/O
- MIL-Code-Generierung, Weight-Blob-Builder
- 73KB Shared Library, alle Tests bestanden

---

## Voraussetzungen

Bevor du loslegst, stelle sicher dass du Folgendes hast:

| Was | Minimum | Geprüft mit |
|-----|---------|-------------|
| Mac | Apple Silicon (M1, M2, M3, M4) | M3 Pro |
| macOS | 15+ | 26.3.1 (Build 25D2128) |
| Xcode CLI Tools | Erforderlich | `xcode-select --install` |

**Prüfen ob alles da ist:**

```bash
# Apple Silicon?
uname -m          # muss "arm64" ausgeben

# Xcode CLI Tools installiert?
xcode-select -p   # muss einen Pfad zeigen, z.B. /Library/Developer/CommandLineTools

# Falls nicht installiert:
xcode-select --install
```

> **Intel Mac?** Dieses Projekt funktioniert **nur** auf Apple Silicon. Der Neural Engine existiert nur in M-Serie Chips.

---

## Installation

### Option A: Schnell (One-Liner)

```bash
curl -sSL https://raw.githubusercontent.com/slavko-at-klincov-it/ANE-Training/main/install.sh | bash
```

Das Skript prüft automatisch alle Voraussetzungen, klont das Repo, baut alles, und führt einen Benchmark aus.

### Option B: Manuell (Schritt für Schritt)

```bash
# 1. Repo klonen
git clone https://github.com/slavko-at-klincov-it/ANE-Training.git
cd ANE-Training

# 2. Erste Demo starten (libane wird automatisch mit-kompiliert)
cd examples
make demo
```

> **Hinweis:** Du musst libane **nicht** separat bauen. Das Examples-Makefile kompiliert `ane.m` direkt mit. Wenn du libane einzeln testen willst: `cd libane && make test`

---

## Schnellstart

Alle Demos liegen im `examples/`-Ordner:

```bash
cd examples
make demo       # ANE Training Demo (Y=2X, 60 Steps)
make bench      # Auto-Benchmark mit TFLOPS + ASCII Chart
make generate   # Shakespeare Text-Generation auf ANE
make explore    # ANE Framework Explorer (35 Klassen, interaktiv)
```

### 1. Training Demo — `make demo`

Trainiert einen Linear-Layer direkt auf dem ANE. Forward-Pass auf dem Neural Engine, Backward-Pass + SGD auf der CPU.

```
Hardware: h15g, 16 ANE cores
Goal: Train W so that Y = W @ X approximates Y = 2*X

step   loss       W[0,0]   W[1,1]   ms/step
0        1.4700    0.289    0.251   0.4
10       0.2149    1.314    1.313   0.3
59       0.0011    1.948    1.976   0.3

Diagonal average: 1.955 (converged!)
```

**Was passiert hier?** Die Gewichte (Weights) starten zufällig und werden Schritt für Schritt angepasst, bis `W ≈ 2.0`. Der Loss sinkt von 1.47 auf 0.001 — das Modell hat gelernt.

### 2. Auto-Benchmark — `make bench`

Erkennt deinen Chip, misst TFLOPS über verschiedene Konfigurationen, zeigt ein ASCII-Barchart mit Vergleich zu bekannten Chips.

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

### 3. Text Generation — `make generate`

Trainiert ein Bigram-Modell auf Shakespeare-Text, generiert dann Zeichen-für-Zeichen mit Typewriter-Effekt.

```
  Training bigram model on Shakespeare...
  step   loss      perplexity
  0       4.1589   64.00
  29      3.1245   22.76

  Generating text (200 chars, temperature=0.8)...
  To be or not to be, that is the question...
```

### 4. ANE Explorer — `make explore`

Zeigt alle 35 ANE-Klassen kategorisiert, markiert welche libane nutzt, interaktiver Modus.

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

---

## Eigenen Code schreiben mit libane

Hier ein vollständiges Beispiel — speichere es als `my_test.c`:

```c
#include <stdio.h>
#include <string.h>
#include "ane.h"

int main() {
    // 1. ANE initialisieren
    int rc = ane_init();
    if (rc != 0) {
        printf("ANE init fehlgeschlagen: %d\n", rc);
        return 1;
    }

    // 2. Hardware-Info abfragen
    ANEDeviceInfo info = ane_device_info();
    printf("ANE: %s, %d cores\n", info.arch, info.num_cores);

    // 3. Gewichte vorbereiten (2x2 Matrix, alles 1.0)
    float weights[] = {1.0f, 0.0f, 0.0f, 1.0f};  // Einheitsmatrix
    ANEWeight w = ane_weight_fp16("@model_path/weights/w.bin", weights, 2, 2);

    // 4. MIL-Programm generieren (Linear Layer: 2→2, Sequenzlänge 1)
    char *mil = ane_mil_linear(2, 2, 1, "@model_path/weights/w.bin");

    // 5. Auf ANE kompilieren
    size_t in_sz = 2 * 1 * sizeof(float);
    size_t out_sz = 2 * 1 * sizeof(float);
    ANEKernel *k = ane_compile(mil, strlen(mil), &w, 1,
                               1, &in_sz, 1, &out_sz, ANE_QOS_BACKGROUND);

    // 6. Input schreiben, ausführen, Output lesen
    float input[] = {3.0f, 7.0f};
    float output[2];
    ane_write(k, 0, input, in_sz);
    ane_eval(k, ANE_QOS_BACKGROUND);
    ane_read(k, 0, output, out_sz);

    printf("Input:  [%.1f, %.1f]\n", input[0], input[1]);
    printf("Output: [%.1f, %.1f]\n", output[0], output[1]);

    // 7. Aufräumen
    ane_free(k);
    free(mil);
    ane_weight_free(&w);
    return 0;
}
```

**Kompilieren und ausführen:**

```bash
xcrun clang -O2 -fobjc-arc -I libane -o my_test my_test.c libane/ane.m \
    -framework Foundation -framework IOSurface -ldl
./my_test
```

Ausführliche API-Dokumentation: [libane/README.md](libane/README.md)

---

## Benchmark-Ergebnisse (M3 Pro)

| Metric | Wert | Erklärung |
|--------|------|-----------|
| Peak FP16 | 9.36 TFLOPS | Maximale Rechenleistung bei kleinen Tensoren |
| Peak FP16 (large spatial) | 18.23 TOPS | Bei großen Tensor-Dimensionen (mehr parallele Ops) |
| INT8 Speedup | 1.0-1.14x | Lohnt sich nicht auf M3 Pro (auf M4: 1.88x) |
| Training Stories110M | 91-183 ms/step | Je nach Vocab-Größe |
| Kernel-Compilation | 520ms | Einmalig beim Start (für 10 Kernel) |
| QoS Background vs Default | 42% schneller | Weniger Scheduling-Overhead |

> **TFLOPS vs TOPS:** TFLOPS = Tera Floating-Point Operations Per Second (zählt multiply+add als 2 Ops). TOPS = Tera Operations Per Second (zählt jede Operation einzeln). Deswegen kann TOPS höher sein als TFLOPS bei gleicher Hardware.

---

## Projektstruktur

```
ANE-Training/
├── README.md                    ← Dieses Dokument
├── ARCHITECTURE.md              ← 4-Schichten Platform-Architektur
├── RESEARCH_ANE_COMPLETE.md     ← Vollständige Forschungsdoku
├── SUMMARY_TECHNICAL.md         ← Technische Zusammenfassung
├── SUMMARY_SIMPLE.md            ← Nicht-technische Zusammenfassung
├── LICENSE                      ← MIT
├── install.sh                   ← One-Liner Installer
│
├── examples/                    ← Lauffähige Demos
│   ├── demo_train.c             ← ANE Training Demo (make demo)
│   ├── bench.c                  ← Auto-Benchmark (make bench)
│   ├── generate.c               ← Text Generation (make generate)
│   ├── explore.m                ← ANE Explorer (make explore)
│   └── Makefile
│
├── libane/                      ← Unsere C-API
│   ├── ane.h                    ← Stabile API (ändert sich nie)
│   ├── ane.m                    ← Implementierung mit Version-Detection
│   ├── libane.dylib             ← Kompilierte Shared Library (73KB)
│   ├── test_ane.c               ← Test-Suite
│   ├── README.md                ← API-Dokumentation
│   └── Makefile
│
├── repo/                        ← maderix/ANE (Referenz-Implementierung)
│   ├── bridge/                  ← Original ANE-Bridge (4 Klassen)
│   ├── training/                ← Training-Pipeline
│   │   └── training_dynamic/    ← Dynamic Spatial Packing (was wir nutzen)
│   ├── test_advanced.m          ← Unsere API-Probe-Tests
│   └── *.m                      ← Diverse Benchmarks (SRAM, INT8, etc.)
│
└── personal-ai/                 ← Personal AI Assistent (auf libane aufgebaut)
    ├── README.md                ← Lösungs-Dokumentation
    └── ...                      ← Collect → Tokenize → Train → Query Pipeline
```

---

## Glossar

| Begriff | Erklärung |
|---------|-----------|
| **ANE** | Apple Neural Engine — der KI-Beschleuniger in Apple Silicon Chips |
| **MIL** | Model Intermediate Language — Apples Textformat für Modell-Beschreibungen |
| **MLIR** | Multi-Level Intermediate Representation — Compiler-Zwischenformat |
| **LLIR** | Low-Level IR — maschinennahe Darstellung vor der finalen Kompilierung |
| **HWX** | Hardware Executable — das finale Binary das auf dem ANE läuft |
| **IOSurface** | macOS-Mechanismus für geteilten Speicher zwischen CPU und ANE (Zero-Copy) |
| **QoS** | Quality of Service — Prioritätsstufe für ANE-Berechnungen |
| **TFLOPS** | Tera Floating-Point Operations Per Second (1 TFLOP = 10¹² FP-Ops/Sek) |
| **TOPS** | Tera Operations Per Second (zählt jede einzelne Operation) |
| **FP16** | 16-bit Floating Point — das native Rechenformat des ANE |
| **Conv 1x1** | 1x1 Convolution — wird auf ANE als schneller matmul-Ersatz genutzt |
| **Dynamic Spatial Packing** | Trick: Weights neben Aktivierungen im Tensor packen, um Re-Compilation zu vermeiden |

---

## Troubleshooting

### `ane_init()` gibt -1 zurück: Framework nicht gefunden

Das AppleNeuralEngine-Framework liegt normalerweise unter:
```
/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine
```
Wenn es fehlt, ist dein macOS zu alt oder Apple hat den Pfad geändert. Prüfe mit:
```bash
ls /System/Library/PrivateFrameworks/AppleNeuralEngine.framework/
```

### `ane_init()` gibt -2 zurück: Klassen nicht gefunden

Apple hat möglicherweise die privaten Klassen umbenannt. `ane_init()` listet automatisch alle gefundenen ANE-Klassen auf stderr. Die neuen Namen in `libane/ane.m` eintragen und neu kompilieren.

### Compile-Fehler: `framework not found`

Xcode Command Line Tools fehlen oder sind veraltet:
```bash
xcode-select --install
# oder bei Problemen:
sudo xcode-select --reset
```

### `uname -m` zeigt `x86_64`

Du bist auf einem Intel Mac oder läufst unter Rosetta. Der ANE existiert nur in Apple Silicon:
```bash
# Prüfe ob du unter Rosetta läufst:
sysctl sysctl.proc_translated 2>/dev/null
# 1 = Rosetta, 0 oder Fehler = nativ
```

### Benchmark zeigt niedrigere TFLOPS als erwartet

- Stelle sicher dass keine anderen rechenintensiven Prozesse laufen
- Nutze QoS Background (9) — ist 42% schneller als Default
- Erste Ausführung ist langsamer wegen Kernel-Compilation (520ms einmalig)

---

## Verwandte Projekte

- [maderix/ANE](https://github.com/maderix/ANE) — Erstes Training auf ANE (Inspiration für dieses Projekt)
- [Orion Paper (arxiv:2603.06728)](https://arxiv.org/abs/2603.06728) — Akademisches Paper zu ANE-Programmierung
- [hollance/neural-engine](https://github.com/hollance/neural-engine) — Community-Dokumentation
- [eiln/ane](https://github.com/eiln/ane) — Linux-Kernel-Driver für ANE

## Hinweis

Dieses Projekt nutzt Apples **private, undokumentierte APIs**. Diese können sich mit jedem macOS-Update ändern. `libane` hat Version-Detection als Schutz — wenn Apple Klassen umbenennt, muss nur `ane.m` aktualisiert werden. Dein Code gegen `ane.h` bleibt unverändert.

## Lizenz

MIT
