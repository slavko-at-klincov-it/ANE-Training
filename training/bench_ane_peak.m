// bench_ane_peak.m — Measure peak ANE throughput without compile overhead
// Compile once, then blast ane_eval() in a tight loop to find max TFLOPS.
// Build: xcrun clang -O2 -fobjc-arc -o bench_ane_peak bench_ane_peak.m \
//        -framework Foundation -framework CoreML -framework IOSurface -ldl
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>

static Class g_D, g_I, g_AR, g_AIO;
static mach_timebase_info_data_t g_tb;
static double tb_ms(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }

static IOSurfaceRef make_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0});
}

typedef struct { id model; IOSurfaceRef ioIn, ioOut; id request; NSString *tmpDir; } Kern;

static Kern compile_conv(int CH_IN, int CH_OUT, int SP) {
    int ws = CH_OUT * CH_IN * 2;
    int tot = 128 + ws;
    uint8_t *blob = (uint8_t*)calloc(tot, 1);
    blob[0]=1; blob[4]=2; blob[64]=0xEF; blob[65]=0xBE; blob[66]=0xAD; blob[67]=0xDE; blob[68]=1;
    *(uint32_t*)(blob+72) = ws;
    *(uint32_t*)(blob+80) = 128;
    _Float16 *wp = (_Float16*)(blob+128);
    srand48(42);
    for (int i = 0; i < CH_OUT*CH_IN; i++) wp[i] = (_Float16)(0.01*(2*drand48()-1));
    NSData *wdata = [NSData dataWithBytesNoCopy:blob length:tot freeWhenDone:YES];

    NSString *mil = [NSString stringWithFormat:
        @"program(1.3)\n"
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n"
        "{\n"
        "    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
        "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
        "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
        "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
        "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
        "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
        "        tensor<fp16, [%d,%d,1,1]> W = const()[name=string(\"W\"), "
        "val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(64)))];\n"
        "        tensor<fp16, [1,%d,1,%d]> y = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W,x=x)"
        "[name=string(\"out\")];\n"
        "    } -> (y);\n"
        "}\n", CH_IN, SP, CH_OUT, CH_IN, CH_OUT, CH_IN, CH_OUT, SP];

    NSData *md = [mil dataUsingEncoding:NSUTF8StringEncoding];
    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D, @selector(modelWithMILText:weights:optionsPlist:),
        md, @{@"@model_path/weights/weight.bin": @{@"offset":@0, @"data":wdata}}, nil);
    id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(g_I, @selector(inMemoryModelWithDescriptor:), desc);
    id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    [[NSFileManager defaultManager] createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
        withIntermediateDirectories:YES attributes:nil error:nil];
    [md writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    [wdata writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

    NSError *e = nil;
    ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e);
    ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);

    IOSurfaceRef ioIn = make_surface(CH_IN * SP * 2);
    IOSurfaceRef ioOut = make_surface(CH_OUT * SP * 2);
    id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioIn);
    id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioOut);
    id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        @[wI], @[@0], @[wO], @[@0], nil, nil, @0);

    return (Kern){mdl, ioIn, ioOut, req, td};
}

static void bench(const char *label, Kern *k, int iters, double flops_per_eval) {
    // Warmup
    NSError *e = nil;
    for (int i = 0; i < 5; i++)
        ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            k->model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, k->request, &e);

    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < iters; i++) {
        ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            k->model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, k->request, &e);
    }
    double ms = tb_ms(mach_absolute_time() - t0);
    double ms_per = ms / iters;
    double tflops = flops_per_eval / (ms_per * 1e9);
    printf("  %-30s  %5d iters  %8.2f ms total  %6.3f ms/eval  %6.2f TFLOPS\n",
           label, iters, ms, ms_per, tflops);
}

int main() {
    @autoreleasepool {
        setbuf(stdout, NULL);
        mach_timebase_info(&g_tb);
        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
        g_D  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        g_I  = NSClassFromString(@"_ANEInMemoryModel");
        g_AR = NSClassFromString(@"_ANERequest");
        g_AIO= NSClassFromString(@"_ANEIOSurfaceObject");

        printf("=== ANE Peak Throughput Benchmark ===\n");
        printf("Compile once, then eval N times in tight loop.\n");
        printf("M3 Pro theoretical peak: 15.8 TFLOPS (fp16)\n\n");

        // Test various kernel sizes to find peak utilization
        struct { int ch_in, ch_out, sp; const char *name; } tests[] = {
            {  64,   64,  32, "64x64 sp=32 (tiny)"},
            { 256,  256,  64, "256x256 sp=64 (small)"},
            { 768,  768, 256, "768x768 sp=256 (DIM)"},
            {2048,  768, 256, "2048x768 sp=256 (FFN)"},
            { 768, 2048, 256, "768x2048 sp=256 (FFN)"},
            {1024, 1024, 256, "1024x1024 sp=256"},
            {2048, 2048, 128, "2048x2048 sp=128 (big)"},
        };
        int n_tests = sizeof(tests)/sizeof(tests[0]);

        printf("Compiling %d kernels...\n", n_tests);
        Kern kerns[n_tests];
        for (int i = 0; i < n_tests; i++) {
            printf("  [%d/%d] %s\n", i+1, n_tests, tests[i].name);
            kerns[i] = compile_conv(tests[i].ch_in, tests[i].ch_out, tests[i].sp);
        }

        printf("\nBenchmarking (pure ane_eval, no IO copy):\n");
        printf("  %-30s  %5s  %12s  %10s  %10s\n", "Kernel", "Iters", "Total ms", "ms/eval", "TFLOPS");
        printf("  %-30s  %5s  %12s  %10s  %10s\n", "------", "-----", "--------", "-------", "------");

        for (int i = 0; i < n_tests; i++) {
            double flops = 2.0 * tests[i].ch_in * tests[i].ch_out * tests[i].sp; // 2*C_in*C_out*SP for matmul
            int iters = (tests[i].ch_in >= 1024) ? 200 : 500;
            bench(tests[i].name, &kerns[i], iters, flops);
        }

        // Sustained 5-second test on the biggest kernel
        printf("\n--- 5-second sustained test (768x768 sp=256) ---\n");
        {
            Kern *k = &kerns[2]; // 768x768
            double flops = 2.0 * 768 * 768 * 256;
            NSError *e = nil;
            int count = 0;
            uint64_t t0 = mach_absolute_time();
            while (tb_ms(mach_absolute_time() - t0) < 5000.0) {
                ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                    k->model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, k->request, &e);
                count++;
            }
            double ms = tb_ms(mach_absolute_time() - t0);
            double ms_per = ms / count;
            double tflops = flops / (ms_per * 1e9);
            printf("  %d evals in %.1fs  (%.3f ms/eval)  %.2f TFLOPS sustained\n",
                   count, ms/1000, ms_per, tflops);
        }

        // Sustained 5-second test on the largest kernel
        printf("\n--- 5-second sustained test (2048x2048 sp=128) ---\n");
        {
            Kern *k = &kerns[6]; // 2048x2048
            double flops = 2.0 * 2048 * 2048 * 128;
            NSError *e = nil;
            int count = 0;
            uint64_t t0 = mach_absolute_time();
            while (tb_ms(mach_absolute_time() - t0) < 5000.0) {
                ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                    k->model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, k->request, &e);
                count++;
            }
            double ms = tb_ms(mach_absolute_time() - t0);
            double ms_per = ms / count;
            double tflops = flops / (ms_per * 1e9);
            printf("  %d evals in %.1fs  (%.3f ms/eval)  %.2f TFLOPS sustained\n",
                   count, ms/1000, ms_per, tflops);
        }

        printf("\nDone. Compare sustained TFLOPS to 15.8T peak.\n");

        // Cleanup
        NSError *e = nil;
        for (int i = 0; i < n_tests; i++) {
            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(kerns[i].model, @selector(unloadWithQoS:error:), 21, &e);
            CFRelease(kerns[i].ioIn); CFRelease(kerns[i].ioOut);
            [[NSFileManager defaultManager] removeItemAtPath:kerns[i].tmpDir error:nil];
        }
    }
    return 0;
}
