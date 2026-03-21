#!/bin/bash
# Download pretokenized TinyStories data for ANE training
# Format: flat uint16 token IDs (Llama2 BPE, 32K vocab)
# Source: enio/TinyStories on HuggingFace (pretokenized with karpathy/llama2.c)
#
# The tar.gz contains data00.bin..data49.bin (50 shards, ~40MB each = ~2GB total tokens).
#
# Usage:
#   bash download_data.sh              # Download all 50 shards, concatenate to tinystories_all.bin
#   bash download_data.sh --shard 0    # Download only shard 0 (tinystories_data00.bin)
#   bash download_data.sh --shards 10  # Download first 10 shards, concatenate

set -e
export LC_ALL=C  # Force C locale to avoid decimal comma issues

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TAR_URL="https://huggingface.co/datasets/enio/TinyStories/resolve/main/tok32000/TinyStories_tok32000.tar.gz?download=true"
TAR_FILE="$SCRIPT_DIR/TinyStories_tok32000.tar.gz"

# Parse args
MODE="all"
NUM_SHARDS=50
SINGLE_SHARD=-1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --shard)
            MODE="single"
            SINGLE_SHARD="$2"
            shift 2
            ;;
        --shards)
            MODE="multi"
            NUM_SHARDS="$2"
            shift 2
            ;;
        *)
            echo "Usage: $0 [--shard N | --shards N]"
            exit 1
            ;;
    esac
done

# Determine output file
if [ "$MODE" = "single" ]; then
    OUTPUT="$SCRIPT_DIR/tinystories_data$(printf '%02d' $SINGLE_SHARD).bin"
else
    OUTPUT="$SCRIPT_DIR/tinystories_all.bin"
fi

# Check if output already exists
if [ -f "$OUTPUT" ]; then
    SIZE=$(stat -f%z "$OUTPUT" 2>/dev/null || stat -c%s "$OUTPUT" 2>/dev/null)
    TOKENS=$((SIZE / 2))
    MB=$((SIZE / 1000000))
    echo "$OUTPUT already exists ($TOKENS tokens, ${MB} MB)"
    echo "Delete it first if you want to re-download."
    exit 0
fi

echo "=== TinyStories Data Download ==="
echo "Downloading pretokenized TinyStories (32K vocab, ~993 MB compressed)..."
echo "  Source: enio/TinyStories on HuggingFace"
echo "  This will take a few minutes depending on your connection."
echo ""

# Download the tar.gz
if [ ! -f "$TAR_FILE" ]; then
    if command -v curl &>/dev/null; then
        curl -L --progress-bar -o "$TAR_FILE" "$TAR_URL"
    elif command -v wget &>/dev/null; then
        wget --show-progress -O "$TAR_FILE" "$TAR_URL"
    else
        echo "Error: need curl or wget"
        exit 1
    fi
else
    echo "Tar file already downloaded, skipping..."
fi

# Verify it's actually a gzip file (not an error page)
if ! file "$TAR_FILE" | grep -q "gzip"; then
    echo "Error: Downloaded file is not a valid gzip archive."
    echo "Content: $(head -c 100 "$TAR_FILE")"
    rm -f "$TAR_FILE"
    exit 1
fi

echo ""

# Create temp directory for extraction
TMPDIR="$SCRIPT_DIR/.tmp_extract_$$"
mkdir -p "$TMPDIR"
trap "rm -rf '$TMPDIR'" EXIT

if [ "$MODE" = "single" ]; then
    # Extract single shard
    SHARD_NAME="data$(printf '%02d' $SINGLE_SHARD).bin"
    echo "Extracting $SHARD_NAME from archive..."

    DATA_FILE=$(tar tzf "$TAR_FILE" 2>/dev/null | grep "$SHARD_NAME" | head -1)
    if [ -z "$DATA_FILE" ]; then
        echo "Error: $SHARD_NAME not found in archive."
        exit 1
    fi

    tar xzf "$TAR_FILE" -C "$TMPDIR" "$DATA_FILE"
    FOUND=$(find "$TMPDIR" -name "$SHARD_NAME" -type f 2>/dev/null | head -1)
    mv "$FOUND" "$OUTPUT"
else
    # Extract ALL shards in a single tar pass, then concatenate selected ones
    if [ "$MODE" = "all" ]; then
        echo "Extracting all shards (single pass, ~2 GB uncompressed)..."
    else
        echo "Extracting shards (single pass, will use first $NUM_SHARDS)..."
    fi

    # Single tar extraction pass — extract everything
    tar xzf "$TAR_FILE" -C "$TMPDIR"

    # Find how many shards we got
    AVAILABLE=$(find "$TMPDIR" -name 'data[0-9][0-9].bin' -type f 2>/dev/null | wc -l | tr -d ' ')
    echo "  Extracted $AVAILABLE shards"

    if [ "$NUM_SHARDS" -gt "$AVAILABLE" ]; then
        NUM_SHARDS=$AVAILABLE
    fi

    # Concatenate in order
    echo "  Concatenating $NUM_SHARDS shards into $(basename $OUTPUT)..."
    > "$OUTPUT"  # truncate/create
    for i in $(seq 0 $((NUM_SHARDS - 1))); do
        SHARD_NAME="data$(printf '%02d' $i).bin"
        FOUND=$(find "$TMPDIR" -name "$SHARD_NAME" -type f 2>/dev/null | head -1)
        if [ -n "$FOUND" ]; then
            cat "$FOUND" >> "$OUTPUT"
            SHARD_SIZE=$(stat -f%z "$FOUND" 2>/dev/null || stat -c%s "$FOUND" 2>/dev/null)
            SHARD_TOKENS=$((SHARD_SIZE / 2))
            SHARD_MB=$((SHARD_SIZE / 1000000))
            echo "    [$(printf '%02d' $((i+1)))/$NUM_SHARDS] $SHARD_NAME: $SHARD_TOKENS tokens (${SHARD_MB} MB)"
        else
            echo "    WARNING: $SHARD_NAME not found"
        fi
    done
fi

# Clean up temp dir (trap handles this, but be explicit)
rm -rf "$TMPDIR"

# Clean up tar.gz to save disk space (it's ~993 MB)
echo ""
echo "Cleaning up archive (~993 MB)..."
rm -f "$TAR_FILE"

SIZE=$(stat -f%z "$OUTPUT" 2>/dev/null || stat -c%s "$OUTPUT" 2>/dev/null)
TOKENS=$((SIZE / 2))
MB=$((SIZE / 1000000))
echo ""
echo "=== Done ==="
echo "  File: $OUTPUT"
echo "  Tokens: $TOKENS (${MB} MB)"

# Sanity check
python3 -c "
import struct
with open('$OUTPUT', 'rb') as f:
    tokens = struct.unpack('<10H', f.read(20))
    print(f'  First 10 tokens: {tokens}')
    f.seek(0, 2)
    total = f.tell() // 2
    f.seek(-20, 2)
    last = struct.unpack('<10H', f.read(20))
    print(f'  Last 10 tokens:  {last}')
    f.seek(0)
    chunk = f.read(200000)
    vals = struct.unpack(f'<{len(chunk)//2}H', chunk)
    mx = max(vals)
    print(f'  Max token ID (sample): {mx}')
    if mx >= 32000:
        print('  WARNING: tokens exceed 32K vocab!')
" 2>/dev/null || true

echo ""
echo "Usage in training:"
echo "  ./train_large_ane --data $OUTPUT"
