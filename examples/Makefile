CC = xcrun clang
CFLAGS = -O2 -Wall -Wno-deprecated-declarations
OBJCFLAGS = -fobjc-arc
FRAMEWORKS = -framework Foundation -framework IOSurface
LDFLAGS = -ldl
LIBANE = ../libane/ane.m

.PHONY: demo clean

demo: demo_train
	@echo ""
	@echo "=== Running ANE Training Demo ==="
	@echo ""
	@./demo_train

demo_train: demo_train.c $(LIBANE) ../libane/ane.h
	$(CC) $(CFLAGS) $(OBJCFLAGS) -I../libane -o $@ $< $(LIBANE) $(FRAMEWORKS) $(LDFLAGS)

clean:
	rm -f demo_train
