#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAINING_DIR="$SCRIPT_DIR/../training/training_dynamic"

# Build libane_train if needed
if [ ! -f "$TRAINING_DIR/libane_train_stories110m.a" ]; then
    echo "==> Building libane_train_stories110m.a ..."
    cd "$TRAINING_DIR"
    make libane_train MODEL=stories110m
fi

echo "==> Compiling ANE Training menu bar app ..."
cd "$SCRIPT_DIR"

swiftc -O \
  -parse-as-library \
  -import-objc-header BridgingHeader.h \
  -I "$TRAINING_DIR" \
  -L "$TRAINING_DIR" \
  -lane_train_stories110m \
  -framework Foundation \
  -framework IOSurface \
  -framework Accelerate \
  -framework Metal \
  -framework IOKit \
  -framework SwiftUI \
  -framework AppKit \
  -o ANETraining \
  ANETraining.swift

echo "==> Built: $SCRIPT_DIR/ANETraining"
echo "   Run with: ./ANETraining"
