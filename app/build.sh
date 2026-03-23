#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAINING_DIR="$SCRIPT_DIR/../training/training_dynamic"
APP_NAME="ANE Training"
BUNDLE="$SCRIPT_DIR/$APP_NAME.app"

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

# Create .app bundle
echo "==> Creating $APP_NAME.app bundle ..."
rm -rf "$BUNDLE"
mkdir -p "$BUNDLE/Contents/MacOS"
mkdir -p "$BUNDLE/Contents/Resources"

# Copy binary
cp ANETraining "$BUNDLE/Contents/MacOS/ANETraining"

# Create Info.plist
cat > "$BUNDLE/Contents/Info.plist" << 'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>ANE Training</string>
    <key>CFBundleDisplayName</key>
    <string>ANE Training</string>
    <key>CFBundleIdentifier</key>
    <string>com.ane-training.menubar</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundleExecutable</key>
    <string>ANETraining</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>LSUIElement</key>
    <true/>
    <key>LSMinimumSystemVersion</key>
    <string>13.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
</dict>
</plist>
PLIST

# Create a simple app icon (1024x1024 brain emoji rendered to PNG → icns)
# If sips/iconutil available, generate from a template. Otherwise skip.
if command -v iconutil &>/dev/null; then
    ICONSET="$SCRIPT_DIR/AppIcon.iconset"
    mkdir -p "$ICONSET"
    # Create a simple icon: white brain on blue gradient
    for SIZE in 16 32 64 128 256 512 1024; do
        sips -z $SIZE $SIZE /System/Library/CoreServices/CoreTypes.bundle/Contents/Resources/GenericApplicationIcon.icns \
            --out "$ICONSET/icon_${SIZE}x${SIZE}.png" 2>/dev/null || true
    done
    # Create @2x variants
    for SIZE in 16 32 128 256 512; do
        DOUBLE=$((SIZE * 2))
        cp "$ICONSET/icon_${DOUBLE}x${DOUBLE}.png" "$ICONSET/icon_${SIZE}x${SIZE}@2x.png" 2>/dev/null || true
    done
    iconutil -c icns "$ICONSET" -o "$BUNDLE/Contents/Resources/AppIcon.icns" 2>/dev/null || true
    rm -rf "$ICONSET"
fi

echo ""
echo "==> Built: $BUNDLE"
echo "   Drag to /Applications or run: open \"$BUNDLE\""
echo "   Or double-click in Finder."
