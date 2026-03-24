#!/bin/bash
# run_inference_benchmark.sh — ANE vs GPU inference benchmark
# Usage: ./run_inference_benchmark.sh [tokens] [model]
set -e

TOKENS=${1:-100}
MODEL=${2:-both}
DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$DIR/results"
CKPT_DIR="$DIR/../training/training_dynamic"
TOK_PATH="$DIR/../assets/models/tokenizer.bin"

mkdir -p "$RESULTS_DIR"

run_gpu_inference() {
    local model=$1
    local label=$2
    local ckpt=""
    if [ "$model" = "tiny_ane" ] && [ -f "$CKPT_DIR/ane_tiny_ckpt.bin" ]; then
        ckpt="--checkpoint $CKPT_DIR/ane_tiny_ckpt.bin"
    elif [ "$model" = "stories110m" ] && [ -f "$CKPT_DIR/ane_stories110M_dyn_ckpt.bin" ]; then
        ckpt="--checkpoint $CKPT_DIR/ane_stories110M_dyn_ckpt.bin"
    fi

    echo ">>> GPU (MPS) Inference: $label"
    echo "------------------------------------------------------------"
    cd "$DIR"
    python3 gpu_inference.py \
        --model "$model" \
        $ckpt \
        --tokens "$TOKENS" \
        --output "$RESULTS_DIR/${model}_gpu_inference.json" \
        2>&1 | tee "$RESULTS_DIR/${model}_gpu_inference_log.txt"
    echo ""
}

run_ane_inference() {
    local model=$1
    local label=$2
    local binary="$CKPT_DIR/generate_${model}"

    if [ ! -f "$binary" ]; then
        echo "WARNING: ANE generate binary not found: $binary"
        echo "  Build with: cd training/training_dynamic && make generate_${model}"
        return 1
    fi

    echo ">>> ANE Inference: $label"
    echo "------------------------------------------------------------"
    cd "$CKPT_DIR"
    "$binary" --temp 0 --max_tokens "$TOKENS" --tokenizer "$TOK_PATH" \
        2>&1 | tee "$RESULTS_DIR/${model}_ane_inference_log.txt"
    echo ""
}

run_model() {
    local model=$1
    local label=$2

    echo ""
    echo "============================================================"
    echo "  Inference Benchmark: $label"
    echo "  Tokens: $TOKENS"
    echo "============================================================"
    echo ""

    run_gpu_inference "$model" "$label"

    echo ">>> Cooling down 5s..."
    sleep 5
    echo ""

    run_ane_inference "$model" "$label"
}

echo "============================================================"
echo "  ANE vs GPU Inference Benchmark"
echo "  Date: $(date '+%Y-%m-%d %H:%M')"
echo "  Tokens: $TOKENS"
echo "============================================================"

if [ "$MODEL" = "both" ] || [ "$MODEL" = "tiny_ane" ]; then
    run_model "tiny_ane" "Tiny-ANE-15M"
fi

if [ "$MODEL" = "both" ] || [ "$MODEL" = "stories110m" ]; then
    run_model "stories110m" "Stories-110M"
fi

echo ""
echo "============================================================"
echo "  Generating Comparison"
echo "============================================================"
cd "$DIR"
python3 compare_inference.py "$RESULTS_DIR"
