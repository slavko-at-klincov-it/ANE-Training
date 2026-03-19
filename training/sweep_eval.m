// sweep_eval.m — Exhaustive eval method + QoS sweep for maximum ANE TFLOPS
// Tests: standard eval, evaluateRealTimeWithModel, doEvaluateDirectWithModel,
//        options dicts, tight vs paced loops, thread priorities
// Kernel: 768x768 sp256 single conv (best single from prior benchmarks)
// Build: xcrun clang -O2 -Wall -fobjc-arc -o sweep_eval sweep_eval.m -framework Foundation -framework IOSurface -ldl
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>
#import <dispatch/dispatch.h>
#import <pthread.h>

// ── Timing ──
static mach_timebase_info_data_t g_tb;
static inline double ns_from_ticks(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom; }
static inline double us_from_ticks(uint64_t t) { return ns_from_ticks(t) / 1e3; }
__attribute__((unused))
static inline double ms_from_ticks(uint64_t t) { return ns_from_ticks(t) / 1e6; }

// ── IOSurface ──
static IOSurfaceRef make_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0});
}

// ── Stats ──
typedef struct {
    double mean_us, min_us, p50_us, p99_us, max_us, tflops;
} BenchResult;

static int cmp_double(const void *a, const void *b) {
    double da = *(const double*)a, db = *(const double*)b;
    return (da > db) - (da < db);
}

static BenchResult compute_stats(double *latencies_us, int n, double gflop_per_eval) {
    BenchResult r = {0};
    if (n == 0) return r;
    qsort(latencies_us, n, sizeof(double), cmp_double);
    double sum = 0;
    for (int i = 0; i < n; i++) sum += latencies_us[i];
    r.mean_us = sum / n;
    r.min_us  = latencies_us[0];
    r.max_us  = latencies_us[n-1];
    r.p50_us  = latencies_us[n/2];
    r.p99_us  = latencies_us[(int)(n * 0.99)];
    r.tflops  = gflop_per_eval / (r.mean_us / 1e6) / 1e3;  // GFLOP/s / 1000
    return r;
}

static void print_result(const char *label, BenchResult r) {
    printf("  %-50s  %7.1f  %7.1f  %7.1f  %7.1f  %7.1f  %6.3f\n",
           label, r.mean_us, r.min_us, r.p50_us, r.p99_us, r.max_us, r.tflops);
}

// ── Globals ──
static const int CH = 768;
static const int SP = 256;
static const int WARMUP = 10;
static const int ITERS  = 500;

int main() {
    @autoreleasepool {
        setbuf(stdout, NULL);
        mach_timebase_info(&g_tb);

        printf("\n");
        printf("  ====================================================\n");
        printf("  ANE EVAL METHOD + QoS SWEEP\n");
        printf("  Kernel: %dx%d sp%d | Warmup: %d | Iters: %d\n", CH, CH, SP, WARMUP, ITERS);
        printf("  ====================================================\n\n");

        // ── Load framework ──
        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
        Class g_D   = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class g_I   = NSClassFromString(@"_ANEInMemoryModel");
        Class g_AR  = NSClassFromString(@"_ANERequest");
        Class g_AIO = NSClassFromString(@"_ANEIOSurfaceObject");
        Class g_Cli = NSClassFromString(@"_ANEClient");
        if (!g_D || !g_I || !g_AR || !g_AIO) {
            printf("  FATAL: Missing ANE classes\n");
            return 1;
        }

        // ── Device info ──
        Class devInfo = NSClassFromString(@"_ANEDeviceInfo");
        if (devInfo) {
            id arch = ((id(*)(Class,SEL))objc_msgSend)(devInfo, @selector(aneArchitectureType));
            id build = ((id(*)(Class,SEL))objc_msgSend)(devInfo, @selector(buildVersion));
            printf("  ANE: %s | Build: %s\n", arch ? [arch UTF8String] : "?", build ? [build UTF8String] : "?");
        }
        NSProcessInfoThermalState ts = [[NSProcessInfo processInfo] thermalState];
        printf("  Thermal: %s\n\n", ts == 0 ? "Nominal" : ts == 1 ? "Fair" : ts == 2 ? "Serious" : "Critical");

        // ── Compile kernel ──
        printf("  Compiling %dx%d sp%d kernel...\n", CH, CH, SP);

        _Float16 *wf = (_Float16*)calloc(CH*CH, sizeof(_Float16));
        for (int i = 0; i < CH; i++) wf[i*CH+i] = (_Float16)1.0f;
        int ws = CH*CH*2, tot = 128+ws;
        uint8_t *blob = (uint8_t*)calloc(tot,1);
        blob[0]=1; blob[4]=2; blob[64]=0xEF; blob[65]=0xBE; blob[66]=0xAD; blob[67]=0xDE; blob[68]=1;
        *(uint32_t*)(blob+72)=ws; *(uint32_t*)(blob+80)=128;
        memcpy(blob+128, wf, ws);
        NSData *wdata = [NSData dataWithBytesNoCopy:blob length:tot freeWhenDone:YES];
        free(wf);

        NSString *mil = [NSString stringWithFormat:
            @"program(1.3)\n"
            "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
            "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
            "{\"coremltools-version\", \"9.0\"}})]\n"
            "{\n"
            "    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
            "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
            "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
            "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
            "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
            "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
            "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];\n"
            "        tensor<fp16, [1,%d,1,%d]> x16 = cast(dtype=to16,x=x)[name=string(\"cin\")];\n"
            "        tensor<fp16, [%d,%d,1,1]> W = const()[name=string(\"W\"), "
            "val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(64)))];\n"
            "        tensor<fp16, [1,%d,1,%d]> y16 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W,x=x16)"
            "[name=string(\"conv\")];\n"
            "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];\n"
            "        tensor<fp32, [1,%d,1,%d]> y = cast(dtype=to32,x=y16)[name=string(\"cout\")];\n"
            "    } -> (y);\n"
            "}\n", CH, SP, CH, SP, CH, CH, CH, CH, CH, SP, CH, SP];

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
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e);
        if (!ok) { printf("  COMPILE FAILED: %s\n", e ? [[e description] UTF8String] : "?"); return 1; }
        ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
        if (!ok) {
            usleep(100000);
            e = nil;
            ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
        }
        if (!ok) { printf("  LOAD FAILED: %s\n", e ? [[e description] UTF8String] : "?"); return 1; }
        printf("  Compiled + loaded OK\n");

        // ── Create I/O surfaces and request ──
        int ioBytes = CH * SP * 4;
        IOSurfaceRef ioIn  = make_surface(ioBytes);
        IOSurfaceRef ioOut = make_surface(ioBytes);
        id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioIn);
        id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioOut);

        IOSurfaceLock(ioIn, 0, NULL);
        float *inp = (float*)IOSurfaceGetBaseAddress(ioIn);
        for (int i = 0; i < CH*SP; i++) inp[i] = 1.0f;
        IOSurfaceUnlock(ioIn, 0, NULL);

        id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wI], @[@0], @[wO], @[@0], nil, nil, @0);

        double GFLOP = 2.0 * CH * CH * SP / 1e9;
        printf("  GFLOP/eval: %.3f\n\n", GFLOP);

        // Latency buffer
        double *lats = (double*)malloc(ITERS * sizeof(double));

        // ── Get _ANEClient ──
        id client = nil;
        if (g_Cli) {
            client = ((id(*)(Class,SEL))objc_msgSend)(g_Cli, @selector(sharedConnection));
        }

        // ════════════════════════════════════════════════════════════════════
        // SECTION A: Standard evaluateWithQoS with different QoS values
        // ════════════════════════════════════════════════════════════════════
        printf("  ========================================================================\n");
        printf("  SECTION A: evaluateWithQoS:options:request:error: — QoS sweep\n");
        printf("  ========================================================================\n");
        printf("  %-50s  %7s  %7s  %7s  %7s  %7s  %6s\n",
               "Method", "Mean", "Min", "P50", "P99", "Max", "TFLOPS");
        printf("  %-50s  %7s  %7s  %7s  %7s  %7s  %6s\n",
               "--------------------------------------------------", "-------", "-------", "-------", "-------", "-------", "------");

        unsigned int qos_vals[] = {0, 5, 9, 13, 17, 21, 25, 33};
        const char *qos_names[] = {"RealTime(0)", "Unknown(5)", "Background(9)", "Unknown(13)",
                                    "Utility(17)", "Default(21)", "UserInit(25)", "UserInter(33)"};

        for (int q = 0; q < 8; q++) {
            // Warmup
            for (int i = 0; i < WARMUP; i++) {
                ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                    mdl, @selector(evaluateWithQoS:options:request:error:),
                    qos_vals[q], @{}, req, &e);
            }
            // Timed
            for (int i = 0; i < ITERS; i++) {
                uint64_t t0 = mach_absolute_time();
                ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                    mdl, @selector(evaluateWithQoS:options:request:error:),
                    qos_vals[q], @{}, req, &e);
                lats[i] = us_from_ticks(mach_absolute_time() - t0);
            }
            char label[64];
            snprintf(label, sizeof(label), "evalWithQoS %s", qos_names[q]);
            print_result(label, compute_stats(lats, ITERS, GFLOP));
        }

        // ════════════════════════════════════════════════════════════════════
        // SECTION B: evaluateRealTimeWithModel on _ANEClient
        // ════════════════════════════════════════════════════════════════════
        printf("\n  ========================================================================\n");
        printf("  SECTION B: evaluateRealTimeWithModel:options:request:error:\n");
        printf("  ========================================================================\n");
        printf("  %-50s  %7s  %7s  %7s  %7s  %7s  %6s\n",
               "Method", "Mean", "Min", "P50", "P99", "Max", "TFLOPS");
        printf("  %-50s  %7s  %7s  %7s  %7s  %7s  %6s\n",
               "--------------------------------------------------", "-------", "-------", "-------", "-------", "-------", "------");

        if (client) {
            SEL rtSel = @selector(evaluateRealTimeWithModel:options:request:error:);
            if ([client respondsToSelector:rtSel]) {
                // Warmup
                for (int i = 0; i < WARMUP; i++) {
                    ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
                        client, rtSel, mdl, @{}, req, &e);
                }
                // Timed
                for (int i = 0; i < ITERS; i++) {
                    uint64_t t0 = mach_absolute_time();
                    ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
                        client, rtSel, mdl, @{}, req, &e);
                    lats[i] = us_from_ticks(mach_absolute_time() - t0);
                }
                print_result("evaluateRealTimeWithModel (client)", compute_stats(lats, ITERS, GFLOP));
            } else {
                printf("  evaluateRealTimeWithModel: NOT AVAILABLE on client\n");
            }
        } else {
            printf("  _ANEClient: NOT AVAILABLE\n");
        }

        // ════════════════════════════════════════════════════════════════════
        // SECTION C: doEvaluateDirectWithModel on _ANEClient
        // ════════════════════════════════════════════════════════════════════
        printf("\n  ========================================================================\n");
        printf("  SECTION C: doEvaluateDirectWithModel:options:request:qos:error:\n");
        printf("  ========================================================================\n");
        printf("  %-50s  %7s  %7s  %7s  %7s  %7s  %6s\n",
               "Method", "Mean", "Min", "P50", "P99", "Max", "TFLOPS");
        printf("  %-50s  %7s  %7s  %7s  %7s  %7s  %6s\n",
               "--------------------------------------------------", "-------", "-------", "-------", "-------", "-------", "------");

        if (client) {
            SEL directSel = @selector(doEvaluateDirectWithModel:options:request:qos:error:);
            if ([client respondsToSelector:directSel]) {
                unsigned int direct_qos[] = {0, 9, 21};
                const char *direct_names[] = {"RT(0)", "BG(9)", "Default(21)"};
                for (int q = 0; q < 3; q++) {
                    // Warmup
                    for (int i = 0; i < WARMUP; i++) {
                        ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                            client, directSel, mdl, @{}, req, direct_qos[q], &e);
                    }
                    // Timed
                    for (int i = 0; i < ITERS; i++) {
                        uint64_t t0 = mach_absolute_time();
                        ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                            client, directSel, mdl, @{}, req, direct_qos[q], &e);
                        lats[i] = us_from_ticks(mach_absolute_time() - t0);
                    }
                    char label[64];
                    snprintf(label, sizeof(label), "doEvaluateDirect QoS=%s", direct_names[q]);
                    print_result(label, compute_stats(lats, ITERS, GFLOP));
                }
            } else {
                printf("  doEvaluateDirectWithModel: NOT AVAILABLE on client\n");
            }
        } else {
            printf("  _ANEClient: NOT AVAILABLE\n");
        }

        // ════════════════════════════════════════════════════════════════════
        // SECTION D: Standard eval with different options dicts
        // ════════════════════════════════════════════════════════════════════
        printf("\n  ========================================================================\n");
        printf("  SECTION D: evaluateWithQoS QoS=9 — different options dicts\n");
        printf("  ========================================================================\n");
        printf("  %-50s  %7s  %7s  %7s  %7s  %7s  %6s\n",
               "Method", "Mean", "Min", "P50", "P99", "Max", "TFLOPS");
        printf("  %-50s  %7s  %7s  %7s  %7s  %7s  %6s\n",
               "--------------------------------------------------", "-------", "-------", "-------", "-------", "-------", "------");

        NSDictionary *opts_empty = @{};
        NSDictionary *opts_nofence = @{@"kANEFDisableIOFencesUseSharedEventsKey": @YES};
        NSDictionary *opts_perf = @{@"kANEFHighPerformanceKey": @YES};
        NSDictionary *opts_nofence_perf = @{
            @"kANEFDisableIOFencesUseSharedEventsKey": @YES,
            @"kANEFHighPerformanceKey": @YES
        };

        NSDictionary *all_opts[] = {opts_empty, opts_nofence, opts_perf, opts_nofence_perf};
        const char *opt_names[] = {
            "opts={}",
            "opts={DisableIOFences=YES}",
            "opts={HighPerformance=YES}",
            "opts={DisableIOFences+HighPerf}"
        };

        for (int o = 0; o < 4; o++) {
            // Warmup
            for (int i = 0; i < WARMUP; i++) {
                ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                    mdl, @selector(evaluateWithQoS:options:request:error:),
                    (unsigned int)9, all_opts[o], req, &e);
            }
            // Timed
            for (int i = 0; i < ITERS; i++) {
                uint64_t t0 = mach_absolute_time();
                ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                    mdl, @selector(evaluateWithQoS:options:request:error:),
                    (unsigned int)9, all_opts[o], req, &e);
                lats[i] = us_from_ticks(mach_absolute_time() - t0);
            }
            char label[64];
            snprintf(label, sizeof(label), "QoS=9 %s", opt_names[o]);
            print_result(label, compute_stats(lats, ITERS, GFLOP));
        }

        // ════════════════════════════════════════════════════════════════════
        // SECTION E: Tight loop vs paced loop (QoS=9)
        // ════════════════════════════════════════════════════════════════════
        printf("\n  ========================================================================\n");
        printf("  SECTION E: Tight loop vs paced loop (QoS=9, %d iters)\n", ITERS);
        printf("  ========================================================================\n");
        printf("  %-50s  %7s  %7s  %7s  %7s  %7s  %6s\n",
               "Method", "Mean", "Min", "P50", "P99", "Max", "TFLOPS");
        printf("  %-50s  %7s  %7s  %7s  %7s  %7s  %6s\n",
               "--------------------------------------------------", "-------", "-------", "-------", "-------", "-------", "------");

        // Tight loop (already measured above, re-measure for consistency)
        for (int i = 0; i < WARMUP; i++) {
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                mdl, @selector(evaluateWithQoS:options:request:error:), (unsigned int)9, @{}, req, &e);
        }
        for (int i = 0; i < ITERS; i++) {
            uint64_t t0 = mach_absolute_time();
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                mdl, @selector(evaluateWithQoS:options:request:error:), (unsigned int)9, @{}, req, &e);
            lats[i] = us_from_ticks(mach_absolute_time() - t0);
        }
        print_result("Tight loop (no delay)", compute_stats(lats, ITERS, GFLOP));

        // 10us delay between evals
        for (int i = 0; i < WARMUP; i++) {
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                mdl, @selector(evaluateWithQoS:options:request:error:), (unsigned int)9, @{}, req, &e);
        }
        for (int i = 0; i < ITERS; i++) {
            uint64_t t0 = mach_absolute_time();
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                mdl, @selector(evaluateWithQoS:options:request:error:), (unsigned int)9, @{}, req, &e);
            lats[i] = us_from_ticks(mach_absolute_time() - t0);
            usleep(10);  // 10us gap
        }
        print_result("10us delay between evals", compute_stats(lats, ITERS, GFLOP));

        // 50us delay
        for (int i = 0; i < WARMUP; i++) {
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                mdl, @selector(evaluateWithQoS:options:request:error:), (unsigned int)9, @{}, req, &e);
        }
        for (int i = 0; i < ITERS; i++) {
            uint64_t t0 = mach_absolute_time();
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                mdl, @selector(evaluateWithQoS:options:request:error:), (unsigned int)9, @{}, req, &e);
            lats[i] = us_from_ticks(mach_absolute_time() - t0);
            usleep(50);  // 50us gap
        }
        print_result("50us delay between evals", compute_stats(lats, ITERS, GFLOP));

        // 200us delay
        for (int i = 0; i < WARMUP; i++) {
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                mdl, @selector(evaluateWithQoS:options:request:error:), (unsigned int)9, @{}, req, &e);
        }
        for (int i = 0; i < ITERS; i++) {
            uint64_t t0 = mach_absolute_time();
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                mdl, @selector(evaluateWithQoS:options:request:error:), (unsigned int)9, @{}, req, &e);
            lats[i] = us_from_ticks(mach_absolute_time() - t0);
            usleep(200);  // 200us gap
        }
        print_result("200us delay between evals", compute_stats(lats, ITERS, GFLOP));

        // ════════════════════════════════════════════════════════════════════
        // SECTION F: Thread priority (dispatch_qos_class_t) effect
        // ════════════════════════════════════════════════════════════════════
        printf("\n  ========================================================================\n");
        printf("  SECTION F: Thread priority (pthread QoS) effect — QoS=9\n");
        printf("  ========================================================================\n");
        printf("  %-50s  %7s  %7s  %7s  %7s  %7s  %6s\n",
               "Method", "Mean", "Min", "P50", "P99", "Max", "TFLOPS");
        printf("  %-50s  %7s  %7s  %7s  %7s  %7s  %6s\n",
               "--------------------------------------------------", "-------", "-------", "-------", "-------", "-------", "------");

        // We use dispatch queues at different QoS to run the benchmark
        typedef struct {
            dispatch_qos_class_t qos_class;
            const char *name;
        } ThreadQoS;

        ThreadQoS thread_qos[] = {
            {QOS_CLASS_BACKGROUND,         "thread=Background"},
            {QOS_CLASS_UTILITY,            "thread=Utility"},
            {QOS_CLASS_DEFAULT,            "thread=Default"},
            {QOS_CLASS_USER_INITIATED,     "thread=UserInitiated"},
            {QOS_CLASS_USER_INTERACTIVE,   "thread=UserInteractive"},
        };

        for (int tq = 0; tq < 5; tq++) {
            __block BenchResult result;
            dispatch_queue_attr_t attr = dispatch_queue_attr_make_with_qos_class(
                DISPATCH_QUEUE_SERIAL, thread_qos[tq].qos_class, 0);
            dispatch_queue_t queue = dispatch_queue_create("sweep.bench", attr);

            dispatch_semaphore_t done = dispatch_semaphore_create(0);
            dispatch_async(queue, ^{
                @autoreleasepool {
                    double *tlats = (double*)malloc(ITERS * sizeof(double));
                    NSError *te = nil;
                    // Warmup
                    for (int i = 0; i < WARMUP; i++) {
                        ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                            mdl, @selector(evaluateWithQoS:options:request:error:),
                            (unsigned int)9, @{}, req, &te);
                    }
                    // Timed
                    for (int i = 0; i < ITERS; i++) {
                        uint64_t t0 = mach_absolute_time();
                        ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                            mdl, @selector(evaluateWithQoS:options:request:error:),
                            (unsigned int)9, @{}, req, &te);
                        tlats[i] = us_from_ticks(mach_absolute_time() - t0);
                    }
                    result = compute_stats(tlats, ITERS, GFLOP);
                    free(tlats);
                    dispatch_semaphore_signal(done);
                }
            });
            dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
            print_result(thread_qos[tq].name, result);
        }

        // ════════════════════════════════════════════════════════════════════
        // SECTION G: Combined best — top 3 from above re-tested with 1000 iters
        // ════════════════════════════════════════════════════════════════════
        printf("\n  ========================================================================\n");
        printf("  SECTION G: Extended run (1000 iters) — top candidates\n");
        printf("  ========================================================================\n");
        printf("  %-50s  %7s  %7s  %7s  %7s  %7s  %6s\n",
               "Method", "Mean", "Min", "P50", "P99", "Max", "TFLOPS");
        printf("  %-50s  %7s  %7s  %7s  %7s  %7s  %6s\n",
               "--------------------------------------------------", "-------", "-------", "-------", "-------", "-------", "------");

        int EXT_ITERS = 1000;
        double *elats = (double*)malloc(EXT_ITERS * sizeof(double));

        // Candidate 1: evalWithQoS QoS=9 tight loop
        for (int i = 0; i < WARMUP*2; i++) {
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                mdl, @selector(evaluateWithQoS:options:request:error:), (unsigned int)9, @{}, req, &e);
        }
        for (int i = 0; i < EXT_ITERS; i++) {
            uint64_t t0 = mach_absolute_time();
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                mdl, @selector(evaluateWithQoS:options:request:error:), (unsigned int)9, @{}, req, &e);
            elats[i] = us_from_ticks(mach_absolute_time() - t0);
        }
        print_result("evalWithQoS QoS=9 (1000 iters)", compute_stats(elats, EXT_ITERS, GFLOP));

        // Candidate 2: evalWithQoS QoS=0 (RealTime)
        for (int i = 0; i < WARMUP*2; i++) {
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                mdl, @selector(evaluateWithQoS:options:request:error:), (unsigned int)0, @{}, req, &e);
        }
        for (int i = 0; i < EXT_ITERS; i++) {
            uint64_t t0 = mach_absolute_time();
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                mdl, @selector(evaluateWithQoS:options:request:error:), (unsigned int)0, @{}, req, &e);
            elats[i] = us_from_ticks(mach_absolute_time() - t0);
        }
        print_result("evalWithQoS QoS=0/RT (1000 iters)", compute_stats(elats, EXT_ITERS, GFLOP));

        // Candidate 3: evaluateRealTimeWithModel
        if (client) {
            SEL rtSel = @selector(evaluateRealTimeWithModel:options:request:error:);
            if ([client respondsToSelector:rtSel]) {
                for (int i = 0; i < WARMUP*2; i++) {
                    ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
                        client, rtSel, mdl, @{}, req, &e);
                }
                for (int i = 0; i < EXT_ITERS; i++) {
                    uint64_t t0 = mach_absolute_time();
                    ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
                        client, rtSel, mdl, @{}, req, &e);
                    elats[i] = us_from_ticks(mach_absolute_time() - t0);
                }
                print_result("evaluateRealTimeWithModel (1000 iters)", compute_stats(elats, EXT_ITERS, GFLOP));
            }
        }

        // Candidate 4: doEvaluateDirect QoS=9
        if (client) {
            SEL directSel = @selector(doEvaluateDirectWithModel:options:request:qos:error:);
            if ([client respondsToSelector:directSel]) {
                for (int i = 0; i < WARMUP*2; i++) {
                    ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                        client, directSel, mdl, @{}, req, (unsigned int)9, &e);
                }
                for (int i = 0; i < EXT_ITERS; i++) {
                    uint64_t t0 = mach_absolute_time();
                    ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                        client, directSel, mdl, @{}, req, (unsigned int)9, &e);
                    elats[i] = us_from_ticks(mach_absolute_time() - t0);
                }
                print_result("doEvaluateDirect QoS=9 (1000 iters)", compute_stats(elats, EXT_ITERS, GFLOP));
            }
        }

        // Candidate 5: QoS=9 + DisableIOFences
        for (int i = 0; i < WARMUP*2; i++) {
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                mdl, @selector(evaluateWithQoS:options:request:error:), (unsigned int)9, opts_nofence, req, &e);
        }
        for (int i = 0; i < EXT_ITERS; i++) {
            uint64_t t0 = mach_absolute_time();
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                mdl, @selector(evaluateWithQoS:options:request:error:), (unsigned int)9, opts_nofence, req, &e);
            elats[i] = us_from_ticks(mach_absolute_time() - t0);
        }
        print_result("QoS=9 + DisableIOFences (1000 iters)", compute_stats(elats, EXT_ITERS, GFLOP));

        free(elats);

        // ── Final summary ──
        printf("\n  ========================================================================\n");
        printf("  DONE — Check results above for the lowest-latency combo\n");
        printf("  Thermal: %s\n",
               [[NSProcessInfo processInfo] thermalState] == 0 ? "Nominal" :
               [[NSProcessInfo processInfo] thermalState] == 1 ? "Fair" :
               [[NSProcessInfo processInfo] thermalState] == 2 ? "Serious" : "Critical");
        printf("  ========================================================================\n\n");

        // ── Cleanup ──
        ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
            mdl, @selector(unloadWithQoS:error:), 21, &e);
        CFRelease(ioIn);
        CFRelease(ioOut);
        free(lats);
        [[NSFileManager defaultManager] removeItemAtPath:td error:nil];
    }
    return 0;
}
