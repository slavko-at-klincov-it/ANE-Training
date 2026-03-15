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
