// tiny_ane.h — Tiny ANE model, fast iteration (15M params)
#pragma once

#define MODEL_NAME "Tiny-ANE-15M"

#define DIM 256
#define HIDDEN 768
#define HEADS 4
#define KV_HEADS 4
#define HD (DIM/HEADS)       // = 64
#define GQA_RATIO 1          // MHA: no GQA
#define Q_DIM (HEADS * HD)   // = 256 = DIM
#define KV_DIM (KV_HEADS * HD) // = 256 = DIM
#define SEQ 256
#define NLAYERS 6
#define VOCAB 32000

#define CKPT_PATH "ane_tiny_ckpt.bin"
#define DEFAULT_DATA_PATH "../tinystories_data00.bin"
