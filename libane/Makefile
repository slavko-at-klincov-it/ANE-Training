CC = xcrun clang
CFLAGS = -O2 -Wall -Wno-deprecated-declarations
OBJCFLAGS = -fobjc-arc
FRAMEWORKS = -framework Foundation -framework IOSurface
LDFLAGS = -ldl

.PHONY: all clean test test-gpu

all: libane.dylib libane_gpu.dylib test_ane test_gpu

# --- ANE library ---

libane.dylib: ane.m ane.h
	$(CC) $(CFLAGS) $(OBJCFLAGS) -fPIC -dynamiclib -o $@ ane.m $(FRAMEWORKS) $(LDFLAGS)

test_ane: test_ane.c ane.m ane.h
	$(CC) $(CFLAGS) $(OBJCFLAGS) -o $@ test_ane.c ane.m $(FRAMEWORKS) $(LDFLAGS)

# --- GPU library (optional, no Metal link-time dependency) ---

libane_gpu.dylib: ane_gpu.m ane_gpu.h
	$(CC) $(CFLAGS) $(OBJCFLAGS) -fPIC -dynamiclib -o $@ ane_gpu.m $(FRAMEWORKS) $(LDFLAGS)

test_gpu: test_gpu.c ane_gpu.m ane_gpu.h
	$(CC) $(CFLAGS) $(OBJCFLAGS) -o $@ test_gpu.c ane_gpu.m $(FRAMEWORKS) $(LDFLAGS) -lm

# --- Test targets ---

test: test_ane
	./test_ane

test-gpu: test_gpu
	./test_gpu

clean:
	rm -f libane.dylib libane_gpu.dylib test_ane test_gpu
