#!/bin/bash
# run_benchmark.sh — ANE vs GPU training benchmark
# Usage: ./run_benchmark.sh [steps] [model]
#   steps: Adam updates per run (default: 100)
#   model: tiny_ane | stories110m (default: both)
set -e

STEPS=${1:-100}
MODEL=${2:-both}
ACCUM=10
DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$DIR/results"
DATA="$DIR/../training/tinystories_data00.bin"

mkdir -p "$RESULTS_DIR"

if [ ! -f "$DATA" ]; then
    echo "ERROR: Training data not found at $DATA"
    exit 1
fi

run_gpu() {
    local model=$1
    local label=$2
    echo ">>> GPU (MPS): $label"
    echo "------------------------------------------------------------"
    cd "$DIR"
    python3 gpu_train.py \
        --data "$DATA" \
        --model "$model" \
        --steps "$STEPS" \
        --accum "$ACCUM" \
        --log-every 10 \
        --output "$RESULTS_DIR/${model}_gpu_results.json" \
        2>&1 | tee "$RESULTS_DIR/${model}_gpu_log.txt"
    echo ""
}

run_ane() {
    local model=$1
    local label=$2
    local binary="$DIR/../training/training_dynamic/train_${model}"

    if [ ! -f "$binary" ]; then
        echo "WARNING: ANE binary not found: $binary"
        echo "  Build with: cd training/training_dynamic && make train_${model}"
        return 1
    fi

    echo ">>> ANE: $label"
    echo "------------------------------------------------------------"
    cd "$DIR/../training/training_dynamic"
    "$binary" --scratch --steps $((STEPS * ACCUM)) --accum $ACCUM \
        --data "$DATA" \
        2>&1 | tee "$RESULTS_DIR/${model}_ane_log.txt"
    echo ""
}

run_model() {
    local model=$1
    local label=$2

    echo ""
    echo "============================================================"
    echo "  Benchmark: $label"
    echo "  Steps: $STEPS Adam updates (${ACCUM}x accumulation)"
    echo "============================================================"
    echo ""

    run_gpu "$model" "$label"

    echo ">>> Cooling down 10s..."
    sleep 10
    echo ""

    run_ane "$model" "$label"
}

# Header
echo "============================================================"
echo "  ANE vs GPU Training Benchmark"
echo "  Date: $(date '+%Y-%m-%d %H:%M')"
echo "  Hardware: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'Apple Silicon')"
echo "  Steps: $STEPS Adam updates, accum=$ACCUM"
echo "============================================================"

if [ "$MODEL" = "both" ] || [ "$MODEL" = "tiny_ane" ]; then
    run_model "tiny_ane" "Tiny-ANE-15M"
fi

if [ "$MODEL" = "both" ] || [ "$MODEL" = "stories110m" ]; then
    run_model "stories110m" "Stories-110M"
fi

# Comparison
echo "============================================================"
echo "  Generating Comparison"
echo "============================================================"
cd "$DIR"
python3 compare.py "$RESULTS_DIR"
