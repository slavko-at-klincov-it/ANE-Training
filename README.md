# ANE-Training — Apple Neural Engine Research & API

Reverse-Engineering von Apples privatem Neural Engine Framework. Enthält eine eigenständige C-API (`libane`), vollständige Hardware-Forschung, und Benchmark-Ergebnisse für M3 Pro.

## Was ist das?

Apple Silicon Chips (M1-M4) haben einen **Neural Engine (ANE)** — einen 16-Core KI-Beschleuniger mit bis zu 18 TOPS. Apple beschränkt ihn offiziell auf Inference via CoreML. Dieses Projekt knackt diese Beschränkung auf und ermöglicht **Training** direkt auf dem ANE.

### Was wir entdeckt haben

- **35 private API-Klassen** (bekannte Projekte nutzen nur 4)
- **6 QoS-Level** — Background (9) ist 42% schneller als Default (21)
- **Hardware-Identität**: M3 Pro = `h15g`, 16 Cores, Board 192
- **Compilation-Pipeline**: MIL → MLIR → LLIR → HWX
- **Performance**: 9.36 TFLOPS (FP16), 18.23 TOPS (large spatial)
- **INT8 lohnt nicht auf M3 Pro** (nur 1.0-1.14x, auf M4: 1.88x)
- **Conv 1x1 ist 3x schneller als matmul** auf ANE

### Was wir gebaut haben

**`libane`** — eine eigene C-API mit automatischer Version-Detection:
- Überlebt Apple API-Änderungen (probiert alternative Klassen-/Methoden-Namen)
- Device-Erkennung, QoS-Support, Zero-Copy I/O
- MIL-Code-Generierung, Weight-Blob-Builder
- 73KB Shared Library, alle Tests bestanden

## Schnellstart: Training auf ANE in 30 Sekunden

```bash
cd examples
make demo
```

```
Hardware: h15g, 16 ANE cores
Goal: Train W so that Y = W @ X approximates Y = 2*X

step   loss       W[0,0]   W[1,1]   ms/step
0        1.4700    0.289    0.251   0.4
10       0.2149    1.314    1.313   0.3
30       0.0173    1.800    1.844   0.3
59       0.0011    1.948    1.976   0.3

Diagonal average: 1.955 (converged!)
```

Die Demo trainiert einen Linear-Layer direkt auf dem ANE: Forward-Pass auf dem Neural Engine, Backward-Pass + SGD auf der CPU. In 60 Steps konvergiert die Weight-Matrix zum korrekten Ergebnis.

## Struktur

```
├── README.md                    ← Dieses Dokument
├── ARCHITECTURE.md              ← 4-Schichten Platform-Architektur
├── RESEARCH_ANE_COMPLETE.md     ← Vollständige Forschung (Benchmarks, APIs, Constraints)
├── SUMMARY_TECHNICAL.md         ← Technische Zusammenfassung
├── SUMMARY_SIMPLE.md            ← Nicht-technische Zusammenfassung
├── LICENSE                      ← MIT
│
├── examples/                    ← Lauffähige Demos
│   ├── demo_train.c             ← ANE Training Demo (make demo)
│   └── Makefile
│
└── libane/                      ← Unsere C-API
    ├── ane.h                    ← Stabile API (ändert sich nie)
    ├── ane.m                    ← Implementation mit Version-Detection
    ├── test_ane.c               ← Tests (3/3 bestanden)
    ├── README.md                ← API-Dokumentation
    └── Makefile
```

## libane — Quickstart

```bash
cd libane
make test
```

```c
#include "ane.h"

ane_init();
ANEDeviceInfo info = ane_device_info();
printf("ANE: %s, %d cores\n", info.arch, info.num_cores);
// → "ANE: h15g, 16 cores"

ANEWeight w = ane_weight_fp16("@model_path/weights/w.bin", data, 256, 256);
char *mil = ane_mil_linear(256, 256, 64, "@model_path/weights/w.bin");

ANEKernel *k = ane_compile(mil, strlen(mil), &w, 1,
                           1, &in_sz, 1, &out_sz, ANE_QOS_BACKGROUND);
ane_write(k, 0, input, bytes);
ane_eval(k, ANE_QOS_BACKGROUND);
ane_read(k, 0, output, bytes);
ane_free(k);
```

Ausführliche API-Dokumentation: [libane/README.md](libane/README.md)

## Benchmark-Ergebnisse (M3 Pro)

| Metric | Wert |
|--------|------|
| Peak FP16 (small spatial) | 9.36 TFLOPS |
| Peak FP16 (large spatial) | 18.23 TOPS |
| INT8 Speedup | 1.0-1.14x (lohnt nicht) |
| Training Stories110M | 91-183 ms/step |
| Kernel-Compilation | 520ms (einmalig, 10 Kernel) |
| QoS Background vs Default | 42% schneller |

## Verwandte Projekte

- [maderix/ANE](https://github.com/maderix/ANE) — Erstes Training auf ANE (Inspiration für dieses Projekt)
- [Orion Paper (arxiv:2603.06728)](https://arxiv.org/abs/2603.06728) — Akademisches Paper zu ANE-Programmierung
- [ANE-PersonalAI](https://github.com/TODO) — Personal AI Assistent der auf dieser Platform aufbaut
- [hollance/neural-engine](https://github.com/hollance/neural-engine) — Community-Dokumentation
- [eiln/ane](https://github.com/eiln/ane) — Linux-Kernel-Driver für ANE

## Voraussetzungen

- macOS 15+ auf Apple Silicon (getestet: M3 Pro, macOS 26.3.1)
- Xcode Command Line Tools

## Hinweis

Dieses Projekt nutzt Apples **private, undokumentierte APIs**. Diese können sich mit jedem macOS-Update ändern. `libane` hat Version-Detection als Schutz — wenn Apple Klassen umbenennt, muss nur `ane.m` aktualisiert werden.

## Lizenz

MIT
