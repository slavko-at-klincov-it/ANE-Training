// qwen3_06b_s128.h — Qwen3-0.6B with SEQ=128 (reduced memory for 18GB Macs)
#pragma once

#define MODEL_NAME "Qwen3-0.6B-S128"

#define DIM 1024
#define HIDDEN 3072
#define HEADS 16
#define KV_HEADS 8
#define HD 128               // explicit head_dim (NOT DIM/HEADS)
#define GQA_RATIO (HEADS / KV_HEADS)  // = 2
#define Q_DIM (HEADS * HD)            // = 2048
#define KV_DIM (KV_HEADS * HD)        // = 1024 (= DIM for this model)
#define SEQ 128
#define NLAYERS 28
#define VOCAB 151936

#define CKPT_PATH "ane_qwen3_06b_s128_ckpt.bin"
#define DEFAULT_DATA_PATH "../tinystories_all.bin"
