CC = xcrun clang
CFLAGS = -O2 -DACCELERATE_NEW_LAPACK -framework Foundation -framework IOSurface -framework Accelerate \
         -isysroot $(shell xcrun --show-sdk-path) -fobjc-arc

# Model selection: make MODEL=stories110m
#   Available: tiny_ane, small_ane, medium_ane, wide_ane, stories110m, qwen3_06b
MODEL ?= stories110m
MODEL_HDR = models/$(MODEL).h

# ANE-optimized models for sweep (excludes large qwen3_06b and stories110m)
SWEEP_MODELS = tiny_ane small_ane medium_ane wide_ane

train: train.m config.h io.h cpu_ops.h mil_dynamic.h $(MODEL_HDR)
	@echo "Building for model: $(MODEL)"
	$(CC) $(CFLAGS) -include $(MODEL_HDR) -o train train.m

# Build a specific model into train_<model>
train_%: train.m config.h io.h cpu_ops.h mil_dynamic.h models/%.h
	@echo "Building for model: $*"
	$(CC) $(CFLAGS) -include models/$*.h -o train_$* train.m

sweep: $(addprefix train_,$(SWEEP_MODELS))
	@echo ""
	@echo "=== ANE Architecture Sweep (200 steps each) ==="
	@echo ""
	@bash sweep_models.sh

# Generate binary for a specific model
generate_%: generate.m config.h io.h cpu_ops.h mil_dynamic.h models/%.h
	@echo "Building generate for model: $*"
	$(CC) $(CFLAGS) -include models/$*.h -o generate_$* generate.m

clean:
	rm -f train train_tiny_ane train_small_ane train_medium_ane train_wide_ane
	rm -f generate_tiny_ane generate_small_ane generate_medium_ane generate_wide_ane generate_stories110m

.PHONY: clean sweep
