// medium_ane.h — Medium ANE sweet spot (512ch, 42M params)
#pragma once

#define MODEL_NAME "Medium-ANE-42M"

#define DIM 512
#define HIDDEN 1536
#define HEADS 8
#define KV_HEADS 8
#define HD (DIM/HEADS)       // = 64
#define GQA_RATIO 1          // MHA: no GQA
#define Q_DIM (HEADS * HD)   // = 512 = DIM
#define KV_DIM (KV_HEADS * HD) // = 512 = DIM
#define SEQ 256
#define NLAYERS 8
#define VOCAB 32000

#define CKPT_PATH "ane_medium_ckpt.bin"
#define DEFAULT_DATA_PATH "../tinystories_data00.bin"
