// small_ane.h — Small ANE sweet spot (384ch, 30M params)
#pragma once

#define MODEL_NAME "Small-ANE-30M"

#define DIM 384
#define HIDDEN 1152
#define HEADS 6
#define KV_HEADS 6
#define HD (DIM/HEADS)       // = 64
#define GQA_RATIO 1          // MHA: no GQA
#define Q_DIM (HEADS * HD)   // = 384 = DIM
#define KV_DIM (KV_HEADS * HD) // = 384 = DIM
#define SEQ 256
#define NLAYERS 8
#define VOCAB 32000

#define CKPT_PATH "ane_small_ckpt.bin"
#define DEFAULT_DATA_PATH "../tinystories_data00.bin"
