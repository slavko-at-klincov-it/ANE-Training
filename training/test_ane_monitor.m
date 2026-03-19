// test_ane_monitor.m — Deep ANE monitoring probe
// Tests: perfStatsMask, _ANEDeviceInfo, thermal state, hw execution time
// Runs benchmark with all monitoring active
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>

static mach_timebase_info_data_t g_tb;
static double tb_ms(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }
static double tb_us(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e3; }

static IOSurfaceRef make_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0});
}

// ── Thermal state ──
static const char *thermal_state_str(NSProcessInfoThermalState s) {
    switch (s) {
        case NSProcessInfoThermalStateNominal:  return "Nominal (cool)";
        case NSProcessInfoThermalStateFair:     return "Fair (warm)";
        case NSProcessInfoThermalStateSerious:  return "Serious (hot, throttling likely)";
        case NSProcessInfoThermalStateCritical: return "Critical (throttling active)";
        default: return "Unknown";
    }
}

// ── Compile a kernel with given dimensions ──
static id compile_kernel(Class g_D, Class g_I, int CH, int SP, NSData **outWData) {
    _Float16 *w = (_Float16*)calloc(CH*CH, sizeof(_Float16));
    for (int i = 0; i < CH; i++) w[i*CH+i] = (_Float16)1.0f;
    int ws = CH*CH*2, tot = 128+ws;
    uint8_t *blob = (uint8_t*)calloc(tot,1);
    blob[0]=1; blob[4]=2; blob[64]=0xEF; blob[65]=0xBE; blob[66]=0xAD; blob[67]=0xDE; blob[68]=1;
    *(uint32_t*)(blob+72)=ws; *(uint32_t*)(blob+80)=128;
    memcpy(blob+128, w, ws);
    NSData *wdata = [NSData dataWithBytesNoCopy:blob length:tot freeWhenDone:YES];
    free(w);
    *outWData = wdata;

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
    ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e);
    ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
    return mdl;
}

int main() {
    @autoreleasepool {
        setbuf(stdout, NULL);
        mach_timebase_info(&g_tb);
        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);

        printf("\n");
        printf("  ██████ ANE DEEP MONITOR ██████\n\n");

        // ════════════════════════════════════════
        // Section 1: Device Info
        // ════════════════════════════════════════
        printf("  ── Device Info (_ANEDeviceInfo) ──\n\n");
        Class devInfo = NSClassFromString(@"_ANEDeviceInfo");
        if (devInfo) {
            BOOL hasANE = ((BOOL(*)(Class,SEL))objc_msgSend)(devInfo, @selector(hasANE));
            unsigned int numANEs = ((unsigned int(*)(Class,SEL))objc_msgSend)(devInfo, @selector(numANEs));
            unsigned int numCores = ((unsigned int(*)(Class,SEL))objc_msgSend)(devInfo, @selector(numANECores));
            id archType = ((id(*)(Class,SEL))objc_msgSend)(devInfo, @selector(aneArchitectureType));
            id subType = ((id(*)(Class,SEL))objc_msgSend)(devInfo, @selector(aneSubType));
            id subTypeVariant = ((id(*)(Class,SEL))objc_msgSend)(devInfo, @selector(aneSubTypeVariant));
            id subTypeProdVar = ((id(*)(Class,SEL))objc_msgSend)(devInfo, @selector(aneSubTypeProductVariant));
            long long boardType = ((long long(*)(Class,SEL))objc_msgSend)(devInfo, @selector(aneBoardType));
            id product = ((id(*)(Class,SEL))objc_msgSend)(devInfo, @selector(productName));
            id build = ((id(*)(Class,SEL))objc_msgSend)(devInfo, @selector(buildVersion));
            id bootArgs = ((id(*)(Class,SEL))objc_msgSend)(devInfo, @selector(bootArgs));
            BOOL excessPower = ((BOOL(*)(Class,SEL))objc_msgSend)(devInfo, @selector(isExcessivePowerDrainWhenIdle));
            BOOL isVM = ((BOOL(*)(Class,SEL))objc_msgSend)(devInfo, @selector(isVirtualMachine));
            BOOL isInternal = ((BOOL(*)(Class,SEL))objc_msgSend)(devInfo, @selector(isInternalBuild));

            printf("  hasANE:              %s\n", hasANE ? "YES" : "NO");
            printf("  numANEs:             %u\n", numANEs);
            printf("  numANECores:         %u\n", numCores);
            printf("  Architecture:        %s\n", archType ? [archType UTF8String] : "nil");
            printf("  SubType:             %s\n", subType ? [subType UTF8String] : "nil");
            printf("  SubType Variant:     %s\n", subTypeVariant ? [subTypeVariant UTF8String] : "nil");
            printf("  SubType ProdVariant: %s\n", subTypeProdVar ? [subTypeProdVar UTF8String] : "nil");
            printf("  Board Type:          %lld\n", boardType);
            printf("  Product:             %s\n", product ? [product UTF8String] : "nil");
            printf("  Build:               %s\n", build ? [build UTF8String] : "nil");
            printf("  Boot Args:           %s\n", bootArgs ? [bootArgs UTF8String] : "(none)");
            printf("  Excessive Idle Power:%s\n", excessPower ? "YES" : "NO");
            printf("  Virtual Machine:     %s\n", isVM ? "YES" : "NO");
            printf("  Internal Build:      %s\n", isInternal ? "YES" : "NO");
        } else {
            printf("  _ANEDeviceInfo: NOT FOUND\n");
        }

        // ════════════════════════════════════════
        // Section 2: Thermal State
        // ════════════════════════════════════════
        printf("\n  ── Thermal State ──\n\n");
        NSProcessInfoThermalState ts = [[NSProcessInfo processInfo] thermalState];
        printf("  Current: %s\n", thermal_state_str(ts));

        // ════════════════════════════════════════
        // Section 3: perfStatsMask probe
        // ════════════════════════════════════════
        printf("\n  ── Performance Stats (perfStatsMask probe) ──\n\n");

        Class g_D  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class g_I  = NSClassFromString(@"_ANEInMemoryModel");
        Class g_AR = NSClassFromString(@"_ANERequest");
        Class g_AIO= NSClassFromString(@"_ANEIOSurfaceObject");
        Class perfClass = NSClassFromString(@"_ANEPerformanceStats");

        int CH = 512, SP = 64;
        NSData *wdata;
        id mdl = compile_kernel(g_D, g_I, CH, SP, &wdata);

        int ioBytes = CH * SP * 4;
        IOSurfaceRef ioIn = make_surface(ioBytes);
        IOSurfaceRef ioOut = make_surface(ioBytes);
        id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioIn);
        id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioOut);

        // Fill input
        IOSurfaceLock(ioIn, 0, NULL);
        float *inp = (float*)IOSurfaceGetBaseAddress(ioIn);
        for (int i = 0; i < CH*SP; i++) inp[i] = 1.0f;
        IOSurfaceUnlock(ioIn, 0, NULL);

        // Test different perfStatsMask values
        unsigned int masks[] = {0, 1, 0xFF, 0xFFFF, 0xFFFFFFFF};
        const char *mask_names[] = {"0x0", "0x1", "0xFF", "0xFFFF", "0xFFFFFFFF"};

        for (int m = 0; m < 5; m++) {
            printf("  Testing perfStatsMask = %s\n", mask_names[m]);

            // Set perfStatsMask on model
            ((void(*)(id,SEL,unsigned int))objc_msgSend)(mdl, @selector(setPerfStatsMask:), masks[m]);
            unsigned int readBack = ((unsigned int(*)(id,SEL))objc_msgSend)(mdl, @selector(perfStatsMask));
            printf("    Mask readback: 0x%X\n", readBack);

            // Create request WITH perfStats slot
            // Try with nil perfStats first — see if driver populates it
            id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
                @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                @[wI], @[@0], @[wO], @[@0], nil, nil, @0);

            NSError *e = nil;
            uint64_t t0 = mach_absolute_time();
            BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                mdl, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
            double lat = tb_us(mach_absolute_time() - t0);

            // Check perfStats after eval
            id ps = ((id(*)(id,SEL))objc_msgSend)(req, @selector(perfStats));
            id psArr = ((id(*)(id,SEL))objc_msgSend)(req, @selector(perfStatsArray));

            printf("    Eval: %s (%.0f µs)\n", ok ? "OK" : "FAIL", lat);
            printf("    perfStats after eval: %s\n", ps ? [[ps description] UTF8String] : "nil");
            printf("    perfStatsArray: %s\n", psArr ? [[psArr description] UTF8String] : "nil");

            if (ps && perfClass) {
                uint64_t hwTime = ((uint64_t(*)(id,SEL))objc_msgSend)(ps, @selector(hwExecutionTime));
                id pcData = ((id(*)(id,SEL))objc_msgSend)(ps, @selector(perfCounterData));
                id rawData = ((id(*)(id,SEL))objc_msgSend)(ps, @selector(pStatsRawData));
                printf("    hwExecutionTime: %llu ns\n", hwTime);
                printf("    perfCounterData: %s (%lu bytes)\n",
                    pcData ? "present" : "nil", pcData ? [(NSData*)pcData length] : 0UL);
                printf("    pStatsRawData: %s (%lu bytes)\n",
                    rawData ? "present" : "nil", rawData ? [(NSData*)rawData length] : 0UL);

                // Dump raw perf counter bytes
                if (pcData && [(NSData*)pcData length] > 0) {
                    const uint8_t *bytes = [(NSData*)pcData bytes];
                    NSUInteger len = [(NSData*)pcData length];
                    printf("    perfCounter hex (%lu bytes): ", (unsigned long)len);
                    for (NSUInteger i = 0; i < len && i < 128; i++) printf("%02x", bytes[i]);
                    if (len > 128) printf("...");
                    printf("\n");

                    // Try interpreting as uint64 array
                    if (len >= 8) {
                        printf("    As uint64 values:\n");
                        const uint64_t *vals = (const uint64_t*)bytes;
                        for (NSUInteger i = 0; i < len/8 && i < 16; i++) {
                            printf("      [%lu] = %llu\n", (unsigned long)i, vals[i]);
                        }
                    }
                }
                if (rawData && [(NSData*)rawData length] > 0) {
                    const uint8_t *bytes = [(NSData*)rawData bytes];
                    NSUInteger len = [(NSData*)rawData length];
                    printf("    rawStats hex (%lu bytes): ", (unsigned long)len);
                    for (NSUInteger i = 0; i < len && i < 128; i++) printf("%02x", bytes[i]);
                    if (len > 128) printf("...");
                    printf("\n");
                }

                // Try stringForPerfCounter
                printf("    Performance counter names:\n");
                for (int i = 0; i < 32; i++) {
                    id str = ((id(*)(id,SEL,int))objc_msgSend)(ps, @selector(stringForPerfCounter:), i);
                    if (str && [(NSString*)str length] > 0) {
                        printf("      counter[%d] = %s\n", i, [(NSString*)str UTF8String]);
                    }
                }

                // performanceCounters method
                id counters = ((id(*)(id,SEL))objc_msgSend)(ps, @selector(performanceCounters));
                printf("    performanceCounters: %s\n", counters ? [[counters description] UTF8String] : "nil");
            }
            printf("\n");
        }

        // ════════════════════════════════════════
        // Section 4: Real-time task mode
        // ════════════════════════════════════════
        printf("  ── Real-Time Task Mode ──\n\n");
        Class clientClass = NSClassFromString(@"_ANEClient");
        if (clientClass) {
            id client = ((id(*)(Class,SEL))objc_msgSend)(clientClass, @selector(sharedConnection));
            if (client) {
                // Benchmark without RT
                uint64_t t0 = mach_absolute_time();
                int N = 200;
                NSError *e = nil;
                id reqBench = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
                    @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                    @[wI], @[@0], @[wO], @[@0], nil, nil, @0);
                for (int i = 0; i < N; i++) {
                    ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                        mdl, @selector(evaluateWithQoS:options:request:error:), 21, @{}, reqBench, &e);
                }
                double normal_ms = tb_ms(mach_absolute_time() - t0);
                printf("  Normal mode:    %d evals in %.1f ms (%.2f ms/eval)\n", N, normal_ms, normal_ms/N);

                // Try RT mode
                BOOL rtOk = ((BOOL(*)(id,SEL))objc_msgSend)(client, @selector(beginRealTimeTask));
                printf("  beginRealTimeTask: %s\n", rtOk ? "OK" : "FAILED");

                if (rtOk) {
                    t0 = mach_absolute_time();
                    for (int i = 0; i < N; i++) {
                        ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                            mdl, @selector(evaluateWithQoS:options:request:error:), 21, @{}, reqBench, &e);
                    }
                    double rt_ms = tb_ms(mach_absolute_time() - t0);
                    printf("  Real-time mode: %d evals in %.1f ms (%.2f ms/eval)\n", N, rt_ms, rt_ms/N);

                    double speedup = normal_ms / rt_ms;
                    printf("  Speedup: %.2fx %s\n", speedup, speedup > 1.05 ? "FASTER" : (speedup < 0.95 ? "SLOWER" : "~SAME"));

                    ((BOOL(*)(id,SEL))objc_msgSend)(client, @selector(endRealTimeTask));
                }
            }
        }

        // ════════════════════════════════════════
        // Section 5: Sustained benchmark with thermal monitoring
        // ════════════════════════════════════════
        printf("\n  ── Sustained Run + Thermal Monitor (10s) ──\n\n");

        // Use larger kernel for sustained test
        int CH2 = 768, SP2 = 256;
        NSData *wdata2;
        id mdl2 = compile_kernel(g_D, g_I, CH2, SP2, &wdata2);
        int ioBytes2 = CH2 * SP2 * 4;
        IOSurfaceRef ioIn2 = make_surface(ioBytes2);
        IOSurfaceRef ioOut2 = make_surface(ioBytes2);
        id wI2 = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioIn2);
        id wO2 = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioOut2);

        IOSurfaceLock(ioIn2, 0, NULL);
        float *inp2 = (float*)IOSurfaceGetBaseAddress(ioIn2);
        for (int i = 0; i < CH2*SP2; i++) inp2[i] = 1.0f;
        IOSurfaceUnlock(ioIn2, 0, NULL);

        id reqSus = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wI2], @[@0], @[wO2], @[@0], nil, nil, @0);

        double GFLOP = 2.0 * CH2 * CH2 * SP2 / 1e9;
        double duration_s = 10.0;
        int interval_evals = 500;  // check thermal every N evals
        int total_evals = 0;
        double elapsed = 0;

        printf("  Kernel: %dx%d sp%d (%.3f GFLOP/eval)\n", CH2, CH2, SP2, GFLOP);
        printf("  %-6s  %-10s  %-10s  %-8s  %s\n", "Time", "Evals", "TFLOPS", "Lat/eval", "Thermal");
        printf("  %-6s  %-10s  %-10s  %-8s  %s\n", "------", "----------", "----------", "--------", "----------");

        NSError *e = nil;
        uint64_t tStart = mach_absolute_time();
        uint64_t tInterval = tStart;

        while (elapsed < duration_s) {
            for (int i = 0; i < interval_evals; i++) {
                ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                    mdl2, @selector(evaluateWithQoS:options:request:error:), 21, @{}, reqSus, &e);
            }
            total_evals += interval_evals;

            uint64_t now = mach_absolute_time();
            elapsed = tb_ms(now - tStart) / 1000.0;
            double interval_ms = tb_ms(now - tInterval);
            double tflops = (interval_evals * GFLOP) / (interval_ms / 1000.0) / 1000.0;
            double lat = interval_ms / interval_evals;
            tInterval = now;

            NSProcessInfoThermalState ts = [[NSProcessInfo processInfo] thermalState];
            printf("  %5.1fs  %10d  %8.2f    %6.3fms  %s\n",
                elapsed, total_evals, tflops, lat, thermal_state_str(ts));
        }

        uint64_t tEnd = mach_absolute_time();
        double total_ms = tb_ms(tEnd - tStart);
        double avg_tflops = (total_evals * GFLOP) / (total_ms / 1000.0) / 1000.0;
        double avg_lat = total_ms / total_evals;

        printf("\n  ── Summary ──\n\n");
        printf("  Duration:     %.1f s\n", total_ms / 1000.0);
        printf("  Total evals:  %d\n", total_evals);
        printf("  Avg TFLOPS:   %.2f\n", avg_tflops);
        printf("  Avg latency:  %.3f ms/eval\n", avg_lat);
        printf("  Final thermal: %s\n", thermal_state_str([[NSProcessInfo processInfo] thermalState]));

        // Cleanup
        ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl, @selector(unloadWithQoS:error:), 21, &e);
        ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl2, @selector(unloadWithQoS:error:), 21, &e);
        CFRelease(ioIn); CFRelease(ioOut);
        CFRelease(ioIn2); CFRelease(ioOut2);
    }
    return 0;
}
