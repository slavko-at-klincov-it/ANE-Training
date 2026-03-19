#!/bin/bash
# sweep_models.sh — ANE Architecture Sweep
# Builds and trains multiple model configs, records speed + loss convergence.
# Each model trains for 200 steps from scratch, then results are compared.
#
# Usage: bash sweep_models.sh [steps]
#   steps: number of training steps per model (default: 200)

set -euo pipefail
cd "$(dirname "$0")"

STEPS=${1:-200}
MODELS="tiny_ane small_ane medium_ane wide_ane"
RESULTS_FILE="sweep_results.txt"
TMPDIR_SWEEP="/tmp/ane_sweep_$$"
mkdir -p "$TMPDIR_SWEEP"

echo "=============================================="
echo "  ANE Architecture Sweep"
echo "  Steps per model: $STEPS"
echo "=============================================="
echo ""

# Helper: extract loss from a step line like "step 0    loss=10.3790  lr=..."
extract_loss() {
    sed -n 's/.*loss=\([0-9.]*\).*/\1/p' | head -1
}

# Helper: extract ms/step from "Train time: 1438ms (71.9ms/step)"
extract_ms_step() {
    sed -n 's/.*(\([0-9.]*\)ms\/step).*/\1/p' | head -1
}

# Build all models first
echo "--- Building models ---"
for model in $MODELS; do
    echo -n "  Building $model... "
    if make -s "train_${model}" MODEL="$model" 2>"$TMPDIR_SWEEP/${model}_build.log"; then
        echo "OK"
    else
        echo "FAILED (see $TMPDIR_SWEEP/${model}_build.log)"
    fi
done
echo ""

# Train each model
echo "--- Training models ($STEPS steps each, from scratch) ---"
echo ""

for model in $MODELS; do
    binary="./train_${model}"
    log="$TMPDIR_SWEEP/${model}_train.log"

    if [ ! -f "$binary" ]; then
        echo "SKIP $model (binary not found)"
        continue
    fi

    echo ">>> Training $model ..."
    # Remove any existing checkpoint so we start fresh
    rm -f "ane_${model}_ckpt.bin" "ane_tiny_ckpt.bin" "ane_small_ckpt.bin" "ane_medium_ckpt.bin" "ane_wide_ckpt.bin" 2>/dev/null || true

    # Run training, capture output
    # Use gtimeout if available, otherwise just run directly
    if command -v gtimeout >/dev/null 2>&1; then
        TIMEOUT_CMD="gtimeout 900"
    elif command -v timeout >/dev/null 2>&1; then
        TIMEOUT_CMD="timeout 900"
    else
        TIMEOUT_CMD=""
    fi

    if $TIMEOUT_CMD "$binary" --scratch --steps "$STEPS" --accum 1 --warmup 20 --lr 3e-4 2>&1 | tee "$log"; then
        echo "  $model completed."
    else
        echo "  $model failed or timed out."
    fi
    echo ""
done

# Parse results
echo "=============================================="
echo "  RESULTS"
echo "=============================================="
echo ""

# Header
FMT="%-20s %8s %10s %10s %10s %10s %12s\n"
HDR_LINE="--------------------"
printf "$FMT" "Model" "Params" "ms/step" "Loss@0" "Loss@100" "Loss@last" "Compile(ms)"
printf "$FMT" "$HDR_LINE" "--------" "----------" "----------" "----------" "----------" "------------"

# Write to file
{
printf "ANE Architecture Sweep Results\n"
printf "Steps per model: %d\n" "$STEPS"
printf "Date: %s\n\n" "$(date)"
printf "$FMT" "Model" "Params" "ms/step" "Loss@0" "Loss@100" "Loss@last" "Compile(ms)"
printf "$FMT" "$HDR_LINE" "--------" "----------" "----------" "----------" "----------" "------------"
} > "$RESULTS_FILE"

for model in $MODELS; do
    log="$TMPDIR_SWEEP/${model}_train.log"
    if [ ! -f "$log" ]; then
        continue
    fi

    # Extract params from "Params: XXM" line
    params=$(grep -o 'Params: [0-9.]*M' "$log" 2>/dev/null | head -1 | sed 's/Params: //' || echo "N/A")

    # Extract compile time from "Compiled 10 kernels in 459ms"
    compile_ms=$(grep 'Compiled 10 kernels in' "$log" 2>/dev/null | head -1 | sed 's/.*in \([0-9]*\)ms.*/\1/' || echo "N/A")

    # Extract first loss (step 0)
    loss_first=$(grep -E '^step 0 ' "$log" 2>/dev/null | extract_loss)
    [ -z "$loss_first" ] && loss_first="N/A"

    # Extract loss at step ~100
    loss_100=$(grep -E '^step (9[0-9]|100) ' "$log" 2>/dev/null | tail -1 | extract_loss)
    [ -z "$loss_100" ] && loss_100="N/A"

    # Extract loss at last reported step
    loss_last=$(grep -E '^step [0-9]+ ' "$log" 2>/dev/null | tail -1 | extract_loss)
    [ -z "$loss_last" ] && loss_last="N/A"

    # Extract ms/step from efficiency report
    ms_step=$(grep 'Train time:' "$log" 2>/dev/null | extract_ms_step)
    [ -z "$ms_step" ] && ms_step="N/A"

    printf "$FMT" "$model" "$params" "$ms_step" "$loss_first" "$loss_100" "$loss_last" "$compile_ms"
    printf "$FMT" "$model" "$params" "$ms_step" "$loss_first" "$loss_100" "$loss_last" "$compile_ms" >> "$RESULTS_FILE"
done

echo ""
echo "Results saved to: $(pwd)/$RESULTS_FILE"
echo "Logs saved to: $TMPDIR_SWEEP/"
