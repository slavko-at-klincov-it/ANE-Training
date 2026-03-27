CC = xcrun clang
CFLAGS = -O2 -Wall -Wno-deprecated-declarations
OBJCFLAGS = -fobjc-arc
FRAMEWORKS = -framework Foundation -framework IOSurface
LDFLAGS = -ldl
LIBANE = ../libane/ane.m

.PHONY: demo bench bench_inference generate explore clean all

all: demo_train bench_bin bench_inference_bin generate_bin explore_bin

# ===== Training Demo =====
demo: demo_train
	@echo ""
	@echo "=== Running ANE Training Demo ==="
	@echo ""
	@./demo_train

demo_train: demo_train.c $(LIBANE) ../libane/ane.h
	$(CC) $(CFLAGS) $(OBJCFLAGS) -I../libane -o $@ $< $(LIBANE) $(FRAMEWORKS) $(LDFLAGS)

# ===== Auto-Benchmark =====
bench: bench_bin
	@./bench

bench_bin: bench.c $(LIBANE) ../libane/ane.h
	$(CC) $(CFLAGS) $(OBJCFLAGS) -I../libane -o bench $< $(LIBANE) $(FRAMEWORKS) $(LDFLAGS)

# ===== CPU vs GPU vs ANE Inference Benchmark =====
bench_inference: bench_inference_bin
	@./bench_inference

bench_inference_bin: bench_inference.m $(LIBANE) ../libane/ane.h
	$(CC) $(CFLAGS) $(OBJCFLAGS) -I../libane -o bench_inference $< $(LIBANE) $(FRAMEWORKS) $(LDFLAGS) -framework Metal -framework MetalPerformanceShaders -framework Accelerate

# ===== Text Generation =====
generate: generate_bin
	@./generate

generate_bin: generate.c $(LIBANE) ../libane/ane.h
	$(CC) $(CFLAGS) $(OBJCFLAGS) -I../libane -o generate $< $(LIBANE) $(FRAMEWORKS) $(LDFLAGS) -lm

# ===== ANE Explorer =====
explore: explore_bin
	@./explore

explore_bin: explore.m ../libane/ane.h
	$(CC) $(CFLAGS) $(OBJCFLAGS) -I../libane -o explore $< $(FRAMEWORKS) $(LDFLAGS)

clean:
	rm -f demo_train bench bench_inference generate explore
