# libane — C-API für Apple Neural Engine

Eigenständige C-Bibliothek die Apples privates Neural Engine Framework in eine saubere, stabile API verpackt. Mit automatischer Version-Detection — überlebt API-Änderungen in zukünftigen macOS-Versionen.

## Build

```bash
make test        # Kompilieren + Tests laufen lassen
make libane.dylib  # Shared Library bauen
```

## API

### Initialisierung

```c
#include "ane.h"

// Framework laden, Klassen auflösen, Version erkennen
int rc = ane_init();  // 0=OK, -1=Framework nicht gefunden, -2=Klassen fehlen

// Hardware-Info
ANEDeviceInfo info = ane_device_info();
// info.arch = "h15g", info.num_cores = 16, info.has_ane = true, ...

// API-Diagnose (druckt auf stderr)
ane_print_diagnostics();

// Was wurde erkannt?
ANEAPIInfo api = ane_api_info();
// api.api_version = 1, api.classes_found = 35, api.has_chaining = true, ...
```

### Kompilierung & Evaluation

**Hinweis:** `@model_path/weights/...` ist ein symbolischer Name, kein Dateipfad. Er verbindet Weight-Blobs im MIL-Programm mit den tatsächlichen Daten im Speicher. Der Name muss im MIL-Code und beim `ane_weight_*`-Aufruf übereinstimmen — der ANE-Compiler löst die Zuordnung auf.

```c
// Weight-Blob bauen (float32 → ANE FP16 Format)
ANEWeight w = ane_weight_fp16("@model_path/weights/w.bin", float_data, rows, cols);

// MIL-Code generieren (Linear Layer als 1x1 Conv)
char *mil = ane_mil_linear(in_ch, out_ch, seq, "@model_path/weights/w.bin");

// Kompilieren
size_t in_sz = in_ch * seq * 4;   // fp32
size_t out_sz = out_ch * seq * 4;
ANEKernel *k = ane_compile(mil, strlen(mil), &w, 1,
                           1, &in_sz, 1, &out_sz,
                           ANE_QOS_BACKGROUND);

// Daten schreiben → ANE ausführen → Ergebnis lesen
ane_write(k, 0, input_data, in_sz);
ane_eval(k, ANE_QOS_BACKGROUND);
ane_read(k, 0, output_data, out_sz);

// Aufräumen
ane_free(k);
free(mil);
ane_weight_free(&w);
```

### Dynamic Weights (Training)

Für Training braucht man Weights die sich **ohne Rekompilierung** ändern lassen. Der ANE hat ein hartes Kompilierungs-Limit (siehe [Compile-Budget](#compile-budget)) — jeder `ane_compile()`-Aufruf zählt und nach ~119 Kompilierungen pro Prozess versagt der ANE still.

**Dynamic Spatial Packing** löst das Problem: Weights werden als Eingabe-Channels kodiert statt zur Compile-Zeit eingebacken. Einmal kompilieren, Weights beliebig oft per IOSurface-Write ändern.

#### `ane_mil_linear_dynamic(in_ch, out_ch, seq)`

Generiert MIL für einen **dynamischen** Linear Layer. Im Gegensatz zu `ane_mil_linear()` werden die Weights nicht zur Compile-Zeit eingebacken, sondern als zusätzliche Input-Channels übergeben.

- **Input-Tensor:** `[1, in_ch + in_ch*out_ch, 1, seq]` fp32
  - Channels `[0..in_ch)` = Aktivierungen (Eingabedaten)
  - Channels `[in_ch..in_ch+in_ch*out_ch)` = Weight-Matrix, flach kodiert
  - Weight-Layout: `W[i][j]` liegt in Channel `(in_ch + i*in_ch + j)`, Spatial-Position 0
- **Output-Tensor:** `[1, out_ch, 1, seq]` fp32
- **Kein Weight-Name nötig** — es gibt keinen `@model_path/weights/...` Parameter
- **Wann benutzen:** Immer wenn sich Weights ändern (Training, Fine-Tuning, Online-Learning). `ane_mil_linear()` ist nur für statische Inferenz mit festen Weights gedacht.

#### `ane_write_dynamic_weights(k, idx, W, in_ch, out_ch, seq)`

Packt eine Weight-Matrix `W[out_ch][in_ch]` in das korrekte Channel/Spatial-Layout des Input-IOSurface. Muss nach jedem Weight-Update aufgerufen werden, **vor** `ane_eval()`.

- `k` — Kompilierter Kernel (von `ane_compile()`)
- `idx` — Input-Tensor-Index (normalerweise 0)
- `W` — Weight-Daten als `float[out_ch * in_ch]` (row-major)
- `in_ch, out_ch, seq` — Dieselben Dimensionen wie bei `ane_mil_linear_dynamic()`

#### Vollständiges Beispiel: Compile-Once Pattern

```c
#include "ane.h"
#include <string.h>
#include <stdlib.h>

int in_ch = 8, out_ch = 4, seq = 1;

// 1) MIL generieren — keine Weights zur Compile-Zeit
char *mil = ane_mil_linear_dynamic(in_ch, out_ch, seq);

// 2) Input = Aktivierungen + Weights zusammen, kein ANEWeight nötig
size_t in_sz = (in_ch + in_ch * out_ch) * seq * sizeof(float);
size_t out_sz = out_ch * seq * sizeof(float);

// 3) EINMAL kompilieren — dieser Kernel lebt für die gesamte Training-Session
ANEKernel *k = ane_compile(mil, strlen(mil), NULL, 0,
                           1, &in_sz, 1, &out_sz,
                           ANE_QOS_BACKGROUND);
free(mil);

// 4) Training-Loop: Weights ändern ohne Rekompilierung
float W[4 * 8];  // out_ch × in_ch
float input[8];   // Aktivierungen
float output[4];  // Ergebnis

for (int step = 0; step < 10000; step++) {
    // ... W und input aktualisieren (Gradient Descent etc.) ...

    // Weights in IOSurface packen (korrektes Layout)
    ane_write_dynamic_weights(k, 0, W, in_ch, out_ch, seq);

    // Aktivierungen in die ersten in_ch Channels schreiben
    ane_lock_input(k, 0);
    float *ptr = (float *)ane_input_ptr(k, 0);
    memcpy(ptr, input, in_ch * sizeof(float));
    ane_unlock_input(k, 0);

    // ANE ausführen
    ane_eval(k, ANE_QOS_BACKGROUND);

    // Ergebnis lesen
    ane_read(k, 0, output, out_sz);
}

ane_free(k);
```

### Zero-Copy I/O

```c
// Direkt in IOSurface schreiben (kein memcpy)
ane_lock_input(k, 0);
float *ptr = (float *)ane_input_ptr(k, 0);
for (int i = 0; i < n; i++) ptr[i] = data[i];
ane_unlock_input(k, 0);

ane_eval(k, ANE_QOS_BACKGROUND);

// Direkt aus IOSurface lesen
ane_lock_output(k, 0);
float *out = (float *)ane_output_ptr(k, 0);
// ... out[i] nutzen ...
ane_unlock_output(k, 0);
```

### Weight-Builder

```c
// FP16 (Standard)
ANEWeight w = ane_weight_fp16(name, float_data, rows, cols);

// FP16 transponiert
ANEWeight wt = ane_weight_fp16_transposed(name, float_data, rows, cols);

// INT8 quantisiert (symmetrisch, scale = max(|w|)/127)
float scale;
ANEWeight wq = ane_weight_int8(name, float_data, rows, cols, &scale);
```

### QoS-Level

```c
ANE_QOS_BACKGROUND        // 9  — Schnellster! Für Training.
ANE_QOS_UTILITY           // 17
ANE_QOS_DEFAULT           // 21 — Standard
ANE_QOS_USER_INITIATED    // 25
ANE_QOS_USER_INTERACTIVE  // 33
ANE_QOS_REALTIME          // 0  — Spezialmodus
```

### Compile-Budget

Der ANE hat ein **hartes Kompilierungs-Limit pro Prozess**. Nach ~119 `ane_compile()`-Aufrufen versagt der ANE still — keine Fehlermeldung, einfach Absturz oder falsche Ergebnisse.

```c
#define ANE_COMPILE_BUDGET     119  // Absolutes Limit
#define ANE_COMPILE_SAFE_LIMIT 110  // Sicherheitsmarge

// Aktuellen Zähler abfragen
int count = ane_compile_count();

if (count >= ANE_COMPILE_SAFE_LIMIT) {
    // Prozess neu starten (exec()) bevor das Budget aufgebraucht ist
    // Der Zähler resettet sich nur bei Prozess-Neustart
}
```

**Konsequenz für Training:** Deshalb ist [Dynamic Spatial Packing](#dynamic-weights-training) der empfohlene Ansatz — einmal kompilieren, Weights beliebig oft ändern, Budget bleibt bei 1.

### Weight-Reload (EXPERIMENTAL)

> **EXPERIMENTAL — Nicht empfohlen.** `ane_reload_weights()` nutzt Delta-Kompilierung (Modell entladen, Weight-Dateien auf Disk patchen, Modell neu laden). Das funktioniert, ist aber **fragil und langsamer** als Dynamic Spatial Packing. Für Training wird der [Dynamic Weights](#dynamic-weights-training) Ansatz empfohlen.

```c
// Neue Weights bauen
ANEWeight w_new = ane_weight_fp16("@model_path/weights/w.bin", new_data, rows, cols);

// Weights hot-swap ohne ane_compile() — zählt NICHT zum Compile-Budget
bool ok = ane_reload_weights(k, &w_new, 1, ANE_QOS_BACKGROUND);
if (!ok) {
    // Fallback: neu kompilieren (verbraucht Compile-Budget!)
    k = ane_compile(...);
}

ane_weight_free(&w_new);
```

## Version-Detection

Wenn Apple in einem zukünftigen macOS private Klassen umbenennt:

1. `ane_init()` probiert automatisch bekannte Alternativen
2. Wenn nichts passt: gibt `-2` zurück und listet ALLE gefundenen ANE-Klassen auf stderr
3. Neue Namen in `ane.m` eintragen, neu kompilieren — fertig
4. `ane.h` bleibt unverändert — dein Code bricht nie

```
# Diagnose-Output wenn alles funktioniert:
=== libane diagnostics ===
API version:     1 (current)
macOS build:     25D2128
Classes found:   35
Descriptor:      OK (_ANEInMemoryModelDescriptor)
Model:           OK (_ANEInMemoryModel)
Request:         OK
IOSurface:       OK
...
```

## Getestet auf

- MacBook Pro M3 Pro, 18GB RAM, macOS 26.3.1 (Build 25D2128)
- ANE Architektur: h15g, 16 Cores
