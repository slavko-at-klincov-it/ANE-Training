CC = xcrun clang
CFLAGS = -O2 -Wall -Wno-deprecated-declarations
OBJCFLAGS = -fobjc-arc
FRAMEWORKS = -framework Foundation -framework IOSurface
LDFLAGS = -ldl

.PHONY: all clean test

all: libane.dylib test_ane

libane.dylib: ane.m ane.h
	$(CC) $(CFLAGS) $(OBJCFLAGS) -fPIC -dynamiclib -o $@ ane.m $(FRAMEWORKS) $(LDFLAGS)

test_ane: test_ane.c ane.m ane.h
	$(CC) $(CFLAGS) $(OBJCFLAGS) -o $@ test_ane.c ane.m $(FRAMEWORKS) $(LDFLAGS)

test: test_ane
	./test_ane

clean:
	rm -f libane.dylib test_ane
