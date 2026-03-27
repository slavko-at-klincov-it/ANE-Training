#!/bin/bash
# run_power_benchmark.sh — ANE vs GPU power & energy efficiency benchmark
# Usage: sudo ./run_power_benchmark.sh [tokens]
#
# Measures CPU/GPU/ANE power rails during inference workloads.
# Requires root for powermetrics.
set -e

TOKENS=${1:-100}
DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$DIR/results"
CKPT_DIR="$DIR/../training/training_dynamic"
TOK_PATH="$DIR/../assets/models/tokenizer.bin"

mkdir -p "$RESULTS_DIR"

if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: Power measurement requires root."
    echo "  Run: sudo $0 $TOKENS"
    exit 1
fi

# Resolve the real user for running python/ANE (not as root)
REAL_USER="${SUDO_USER:-$(whoami)}"

echo "============================================================"
echo "  ANE vs GPU Power & Efficiency Benchmark"
echo "  Date: $(date '+%Y-%m-%d %H:%M')"
echo "  Hardware: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'Apple Silicon')"
echo "  Tokens: $TOKENS per run"
echo "  User: $REAL_USER (workloads run as user, powermetrics as root)"
echo "============================================================"

# --- Measure idle baseline ---
echo ""
echo ">>> Phase 0: Idle baseline (5s)"
echo "------------------------------------------------------------"
powermetrics -s cpu_power,gpu_power,ane_power -i 1000 -n 5 \
    --output-file "$RESULTS_DIR/power_idle.txt" 2>/dev/null
echo "  Idle baseline captured"

# --- Helper: start/stop powermetrics ---
start_power() {
    local label=$1
    powermetrics -s cpu_power,gpu_power,ane_power -i 500 \
        --output-file "$RESULTS_DIR/power_${label}.txt" 2>/dev/null &
    POWER_PID=$!
    sleep 1  # let it start
}

stop_power() {
    if [ -n "$POWER_PID" ]; then
        kill "$POWER_PID" 2>/dev/null || true
        wait "$POWER_PID" 2>/dev/null || true
        POWER_PID=""
    fi
}

# --- GPU Inference with power ---
echo ""
echo ">>> Phase 1: GPU (MPS) Inference — Stories-110M"
echo "------------------------------------------------------------"
start_power "gpu_inference"
su "$REAL_USER" -c "cd '$DIR' && python3 gpu_inference.py \
    --model stories110m \
    --checkpoint '$CKPT_DIR/ane_stories110M_dyn_ckpt.bin' \
    --tokens $TOKENS \
    --output '$RESULTS_DIR/power_gpu_inference.json'" 2>&1 | tee "$RESULTS_DIR/power_gpu_inference_log.txt"
stop_power
echo ""

# Cool down
echo ">>> Cooling down 10s..."
sleep 10

# --- ANE Inference with power ---
echo ""
echo ">>> Phase 2: ANE Inference — Stories-110M"
echo "------------------------------------------------------------"
ANE_BIN="$CKPT_DIR/generate_stories110m"
if [ ! -f "$ANE_BIN" ]; then
    echo "ERROR: $ANE_BIN not found. Build with: cd training/training_dynamic && make generate_stories110m"
    exit 1
fi
start_power "ane_inference"
su "$REAL_USER" -c "cd '$CKPT_DIR' && ./generate_stories110m \
    --temp 0 --max_tokens $TOKENS \
    --tokenizer '$TOK_PATH'" 2>&1 | tee "$RESULTS_DIR/power_ane_inference_log.txt"
stop_power
echo ""

# Cool down
echo ">>> Cooling down 10s..."
sleep 10

# --- GPU Training with power ---
echo ""
echo ">>> Phase 3: GPU (MPS) Training — Stories-110M (20 steps)"
echo "------------------------------------------------------------"
start_power "gpu_training"
su "$REAL_USER" -c "cd '$DIR' && python3 gpu_train.py \
    --model stories110m \
    --data '$DIR/../training/tinystories_data00.bin' \
    --steps 20 --accum 10 --log-every 5 \
    --output '$RESULTS_DIR/power_gpu_training.json'" 2>&1 | tee "$RESULTS_DIR/power_gpu_training_log.txt"
stop_power
echo ""

# Cool down
echo ">>> Cooling down 10s..."
sleep 10

# --- ANE Training with power ---
echo ""
echo ">>> Phase 4: ANE Training — Stories-110M (200 steps)"
echo "------------------------------------------------------------"
ANE_TRAIN="$CKPT_DIR/train_stories110m"
if [ ! -f "$ANE_TRAIN" ]; then
    echo "ERROR: $ANE_TRAIN not found."
    exit 1
fi
start_power "ane_training"
su "$REAL_USER" -c "cd '$CKPT_DIR' && ./train_stories110m \
    --scratch --steps 200 --accum 10 \
    --data '$DIR/../training/tinystories_data00.bin'" 2>&1 | tee "$RESULTS_DIR/power_ane_training_log.txt"
stop_power
echo ""

# --- Parse & Compare ---
echo "============================================================"
echo "  Generating Power Comparison"
echo "============================================================"
cd "$DIR"
python3 compare_power.py "$RESULTS_DIR"
