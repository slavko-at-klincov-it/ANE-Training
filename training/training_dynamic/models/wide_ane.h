// wide_ane.h — Wide but shallow ANE model (640ch, 45M params)
#pragma once

#define MODEL_NAME "Wide-ANE-45M"

#define DIM 640
#define HIDDEN 1920
#define HEADS 10
#define KV_HEADS 10
#define HD (DIM/HEADS)       // = 64
#define GQA_RATIO 1          // MHA: no GQA
#define Q_DIM (HEADS * HD)   // = 640 = DIM
#define KV_DIM (KV_HEADS * HD) // = 640 = DIM
#define SEQ 256
#define NLAYERS 6
#define VOCAB 32000

#define CKPT_PATH "ane_wide_ckpt.bin"
#define DEFAULT_DATA_PATH "../tinystories_data00.bin"
