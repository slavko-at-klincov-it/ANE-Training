# Architektur — ANE Training Platform

## Was ist das hier?

Dieses Projekt ist in zwei Teile aufgeteilt:

1. **Die Platform** (dieses Dokument) — alles was wir über Apples Neural Engine herausgefunden und gebaut haben. Das ist die Basis, auf der beliebige Lösungen aufgebaut werden können.
2. **Die Lösung** (`personal-ai/`) — eine konkrete Anwendung, die auf der Platform aufbaut: ein persönlicher KI-Assistent der lokal auf deinem Mac lernt.

Die Platform ist wiederverwendbar. Man könnte darauf auch einen Code-Assistenten, einen Log-Analyzer, einen lokalen Chatbot oder einen Bild-Classifier bauen.

---

## Schicht 1: Forschung & Erkenntnis

### Was wir gemacht haben
Reverse-Engineering von Apples privatem Neural Engine Framework durch:
- Runtime-Introspection aller ObjC-Klassen via `objc_copyClassList`
- Systematisches Proben von Methoden via `objc_msgSend` mit Crash-Recovery
- Benchmark-Suite auf eigenem M3 Pro Hardware ausgeführt
- Web-Research zu allen bekannten RE-Projekten und dem Orion-Paper

### Was wir herausgefunden haben

**35 private Klassen** entdeckt (das Original-Repo nutzt nur 4):

| Klasse | Was sie tut | Genutzt? |
|--------|------------|----------|
| `_ANEInMemoryModelDescriptor` | MIL-Text + Weights → Modell-Descriptor | Ja (libane) |
| `_ANEInMemoryModel` | Compile → Load → Evaluate → Unload | Ja (libane) |
| `_ANERequest` | Bindet IOSurface I/O an Evaluation | Ja (libane) |
| `_ANEIOSurfaceObject` | Wrapper für IOSurface (Zero-Copy) | Ja (libane) |
| `_ANEClient` | Direkte Hardware-Verbindung zum ANE-Daemon | Ja (Probing) |
| `_ANEDeviceInfo` | Hardware-Erkennung (Architektur, Cores, Board) | Ja (libane) |
| `_ANEQoSMapper` | 6 QoS-Level für Prioritätssteuerung | Ja (Probing) |
| `_ANEChainingRequest` | Kernel-Pipeline ohne CPU-Roundtrip | Teilweise (Objekte erstellt, validate noch nicht) |
| `_ANEBuffer` | IOSurface + Symbol-Index Bindung | Ja (Probing) |
| `_ANEPerformanceStats` | Hardware-Performance-Counter | Nein (braucht Entitlements?) |
| 25 weitere | Daemon-Connection, Virtualization, Weight-Management, etc. | Katalogisiert |

**Hardware-Identität des M3 Pro** (via `_ANEDeviceInfo`):
```
Architektur:  h15g (M4 = h16g)
Cores:        16
Einheiten:    1
Board-Typ:    192
Power-Gating: 0mW wenn idle
```

**6 QoS-Level** entdeckt und benchmarked:
```
Background(9):      0.143 ms  ← SCHNELLSTER! Für Training nutzen.
Utility(17):        0.227 ms
Default(21):        0.248 ms  ← Was das Original-Repo nutzt.
UserInitiated(25):  0.249 ms
UserInteractive(33):0.247 ms
```
**Erkenntnis:** Background-QoS ist 42% schneller als Default weil weniger Scheduling-Overhead.

**Compilation-Pipeline** (vorher unbekannt):
```
model.mil → model.bc.mlir → model.llir.bundle → model.hwx
(MIL Text)   (MLIR Bitcode)  (Low-Level IR)     (HW Binary)
```

**Performance-Benchmarks** (M3 Pro):
- Peak FP16 (kleine Spatial): 9.36 TFLOPS
- Peak FP16 (große Spatial): 18.23 TOPS
- INT8 bringt nur 1.0-1.14x (auf M4: 1.88x) → lohnt sich nicht auf M3
- Training: 183ms/step (32K Vocab), 91ms/step (nach Vocab-Compaction)

**20 ANE-Constraints** (aus dem Orion-Paper):
- `concat` wird vom ANE-Compiler abgelehnt
- `gelu` nicht unterstützt (tanh-Approximation nutzen)
- Conv 1x1 ist 3x schneller als matmul
- ~119 Compilations pro Prozess, dann exec()-Restart nötig
- Letzter Tensor-Axis muss 64-Byte-aligned sein

### Wo ist das dokumentiert?
- `RESEARCH_ANE_COMPLETE.md` — Vollständige Forschungsdoku (Benchmarks, API-Surface, Constraints, Quellen)
- `SUMMARY_TECHNICAL.md` — Technische Zusammenfassung des maderix/ANE Repos
- `SUMMARY_SIMPLE.md` — Nicht-technische Zusammenfassung mit Use-Cases
- `repo/test_advanced.m` — Unsere API-Probe-Tests (Chaining, QoS, Buffer, PerfStats)

---

## Schicht 2: libane — Unsere eigene C-API

### Warum?
Das maderix/ANE Repo hat seinen eigenen Bridge-Code (`bridge/ane_bridge.m`), aber:
- Keine Version-Detection (bricht wenn Apple Klassen umbenennt)
- Keine Device-Erkennung
- Kein QoS-Support (alles auf Default=21)
- Keine saubere API-Trennung

### Was wir gebaut haben

```
libane/
├── ane.h          ← Stabile C-API (ändert sich NIE)
├── ane.m          ← Implementierung (ändert sich wenn Apple API ändert)
├── libane.dylib   ← Shared Library (73KB, arm64)
├── test_ane.c     ← Test-Suite (3/3 bestanden)
└── Makefile
```

### Wie die Version-Detection funktioniert

```c
// ane.m versucht bekannte Klassen-Namen in Reihenfolge:
static const char *MODEL_CLASS_NAMES[] = {
    "_ANEInMemoryModel",    // aktuell (macOS 15-26)
    "_ANEModel",            // falls Apple umbenennt
    "ANEInMemoryModel",     // ohne Unterstrich
    "ANEModel",             // komplett neu
    NULL
};
```

Bei jedem `ane_init()`:
1. Framework laden (3 bekannte Pfade werden probiert)
2. Klassen auflösen (Alternativen durchprobieren)
3. Selektoren auflösen (Methoden-Namen durchprobieren)
4. API-Version bestimmen (1 = bekannt, 0 = unbekannt)
5. Wenn unbekannt: alle ANE-Klassen auflisten für Debugging

**Wenn Apple die API ändert:**
- `ane.h` bleibt gleich (dein Code ändert sich nicht)
- Neue Klassen-/Methoden-Namen in `ane.m` eintragen
- Neu kompilieren, fertig

### API-Übersicht

```c
// Initialisierung + Hardware-Erkennung
ane_init();
ANEDeviceInfo info = ane_device_info();   // h15g, 16 cores, etc.
ANEAPIInfo api = ane_api_info();          // version, classes found, selectors resolved
ane_print_diagnostics();                  // Druckt alles auf stderr

// Kompilierung
ANEWeight w = ane_weight_fp16("@model_path/weights/w.bin", data, rows, cols);
char *mil = ane_mil_linear(in_ch, out_ch, seq, weight_name);
ANEKernel *k = ane_compile(mil, len, &w, 1, 1, &in_sz, 1, &out_sz, ANE_QOS_BACKGROUND);

// Evaluation
ane_write(k, 0, input_data, bytes);       // Daten → IOSurface
ane_eval(k, ANE_QOS_BACKGROUND);          // ANE ausführen
ane_read(k, 0, output_data, bytes);       // IOSurface → Daten

// Zero-Copy (schneller, kein memcpy)
ane_lock_input(k, 0);
float *ptr = (float *)ane_input_ptr(k, 0);
ptr[0] = 42.0f;                           // Direkt in IOSurface schreiben
ane_unlock_input(k, 0);

// Aufräumen
ane_free(k);
ane_weight_free(&w);
```

### Architektur-Entscheidungen

| Entscheidung | Grund |
|-------------|-------|
| C-API (nicht ObjC/Swift) | Maximale Portabilität, von jedem Sprache nutzbar |
| Cached Selectors | Null Overhead zur Laufzeit bei Methoden-Aufrufen |
| IOSurface für I/O | Zero-Copy zwischen CPU und ANE (kein Staging-Buffer) |
| Conv 1x1 statt matmul | 3x schneller auf ANE (Orion-Paper bestätigt) |
| QoS=9 (Background) | 42% schneller als Default, niedrigste Systembelastung |
| FP32 Ein/Ausgang, FP16 intern | ANE rechnet in FP16, aber CPU braucht FP32 für Gradienten |

---

## Schicht 3: Training Pipeline

### Was wir nutzen
Das maderix/ANE Repo (`repo/training/training_dynamic/`) — ein funktionierender Transformer-Trainer der direkt auf dem ANE läuft.

### Was wir angepasst haben
- Dashboard-TFLOPS: 15.8 (M4) → 9.36 (M3 Pro)
- Synthetische Testdaten erstellt (500K Tokens)
- Tokenizer via git-lfs gepullt
- Verifiziert: 50 Steps stabil, kein NaN, keine Crashes

### Wie Training funktioniert (vereinfacht)
```
1. MIL-Programme generieren (10 Kernel-Typen pro Layer)
2. Einmal auf ANE kompilieren (520ms für alle 10)
3. Pro Training-Step:
   a. Weights in IOSurface-Spatial-Dimension packen (CPU)
   b. Forward-Pass: ANE evaluiert Kernel (22ms)
   c. Attention + Softmax + RoPE auf CPU (weil ANE kein Causal-Masking kann)
   d. Backward-Pass: ANE für dx-Gradienten (30ms), CPU für dW via CBLAS
   e. Adam-Optimizer-Update auf CPU
   f. Weights werden direkt im IOSurface aktualisiert (kein Recompile!)
```

### Schlüssel-Innovation: Dynamic Spatial Packing
Weights werden **neben** den Aktivierungen im Spatial-Dimension des Input-Tensors verpackt:
```
IOSurface: [1, DIM, 1, SEQ + WEIGHT_COLS]
                        ↑Daten  ↑Weights

Inside ANE-Kernel:
  daten   = slice(input, [0,0,0,0], [1,DIM,1,SEQ])
  weights = slice(input, [0,0,0,SEQ], [1,DIM,1,OC])
  output  = matmul(daten, weights)
```
→ **Ein Compile für alle Weight-Updates.** Ohne diesen Trick müsste man nach jedem Adam-Step alle Kernel neu kompilieren (119-Compile-Limit!).

### Performance auf M3 Pro
| Modell | Parameter | ms/step | Kompilier-Zeit |
|--------|-----------|---------|----------------|
| Stories110M (32K Vocab) | 109.5M | 183ms | 520ms |
| Stories110M (124 Vocab, compacted) | 109.5M | 91ms | 520ms |
| Qwen3-0.6B | 596M | ~412ms (geschätzt) | ~800ms |

---

## Schicht 4: Lösung (Personal AI)

→ Siehe `personal-ai/README.md`

Die Lösung baut auf ALLEN drei darunterliegenden Schichten auf:
- **Schicht 1** (Forschung) lieferte das Wissen über QoS, Vocab-Compaction, Constraints
- **Schicht 2** (libane) liefert die stabile API für zukünftige ANE-Nutzung
- **Schicht 3** (Training) liefert den funktionierenden Trainer

---

## Dateistruktur

```
ANE-Training/
│
├── ARCHITECTURE.md              ← Dieses Dokument (Platform-Architektur)
├── RESEARCH_ANE_COMPLETE.md     ← Volle Forschungs-Doku
├── SUMMARY_TECHNICAL.md         ← Technische Zusammenfassung
├── SUMMARY_SIMPLE.md            ← Einfache Zusammenfassung
│
├── libane/                      ← SCHICHT 2: Unsere C-API
│   ├── ane.h                    ← Stabile API-Schnittstelle
│   ├── ane.m                    ← Implementierung mit Version-Detection
│   ├── libane.dylib             ← Kompilierte Shared Library
│   ├── test_ane.c               ← Tests (3/3 bestanden)
│   └── Makefile
│
├── repo/                        ← SCHICHT 3: maderix/ANE (Referenz + Training)
│   ├── bridge/                  ← Original ANE-Bridge (4 Klassen)
│   ├── training/                ← Training-Code
│   │   └── training_dynamic/    ← Dynamic-Pipeline (was wir nutzen)
│   ├── test_advanced.m          ← Unsere API-Tests
│   ├── inmem_bench.m            ← Performance-Benchmarks
│   ├── sram_bench.m             ← SRAM-Probing
│   └── ane_int8_bench.m         ← INT8-Benchmarks
│
└── personal-ai/                 ← SCHICHT 4: Lösung (Personal AI)
    ├── README.md                ← Lösungs-Dokumentation
    ├── pai                      ← CLI Entry-Point
    ├── collector/               ← Daten-Sammlung
    ├── tokenizer/               ← Tokenisierung
    ├── trainer/                 ← Training-Scheduler
    └── inference/               ← Query-Interface
```

---

## Chronologie

| Wann | Was | Warum | Ergebnis |
|------|-----|-------|----------|
| Phase A | API-Tests & Probing | Wissen was die ANE-Hardware kann | 35 Klassen, QoS-Levels, h15g-Identität |
| Phase B | libane gebaut | Stabile, version-sichere API | ane.h/ane.m, 3/3 Tests bestanden |
| Phase C | Training verifiziert | Beweis dass es auf M3 Pro funktioniert | 91-183ms/step, 50 Steps stabil |
| Phase D | Personal AI gebaut | Konkrete Anwendung | Collect→Tokenize→Train→Query Pipeline |

## Was kann man noch darauf bauen?

Die Platform (Schicht 1-3) ist die Basis. Darauf kann man bauen:

- **Code-Assistent**: Lernt deine Code-Patterns, schlägt Completions vor
- **Log-Analyzer**: Trainiert auf System-/App-Logs, erkennt Anomalien
- **Dokument-Suche**: Semantische Suche über alle deine Dateien
- **Meeting-Notes**: Lernt dein Vokabular, erstellt bessere Zusammenfassungen
- **Lokaler Chatbot**: Fine-tuned auf deinen Schreibstil
- **Federated Learning**: Mehrere Geräte trainieren lokal, teilen nur Gradienten

Alles 100% lokal, 100% privat, auf dem ANE deines Macs.
