// test_compiler_opts.m — Discover ANE compiler options and their performance impact
// Probes: _ANEStrings, _ANEInMemoryModel compilation methods, kANEF* key constants,
//         ANECCreateCompilerOptionDictionary, optionsPlist on descriptor, compile/load/eval options
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>

static mach_timebase_info_data_t g_tb;
static double tb_ms(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }
static double tb_us(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e3; }

// Helper: resolve a kANEF* NSString constant from a dlsym handle
static NSString *resolve_key(void *handle, const char *sym) {
    CFStringRef *ptr = (CFStringRef *)dlsym(handle, sym);
    if (ptr && *ptr) return (__bridge NSString *)*ptr;
    return nil;
}

static IOSurfaceRef make_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0});
}

// ── Build MIL text for a CH×CH conv, spatial SP ──
static NSString *make_mil(int CH, int SP) {
    return [NSString stringWithFormat:
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
}

// ── Build weight blob ──
static NSData *make_weight_data(int CH) {
    _Float16 *w = (_Float16*)calloc(CH*CH, sizeof(_Float16));
    for (int i = 0; i < CH; i++) w[i*CH+i] = (_Float16)1.0f;
    int ws = CH*CH*2, tot = 128+ws;
    uint8_t *blob = (uint8_t*)calloc(tot,1);
    blob[0]=1; blob[4]=2; blob[64]=0xEF; blob[65]=0xBE; blob[66]=0xAD; blob[67]=0xDE; blob[68]=1;
    *(uint32_t*)(blob+72)=ws; *(uint32_t*)(blob+80)=128;
    memcpy(blob+128, w, ws);
    free(w);
    return [NSData dataWithBytesNoCopy:blob length:tot freeWhenDone:YES];
}

// ── Compile+load a model with given compile/load options ──
static id compile_model(Class g_D, Class g_I, int CH, int SP, NSData *wdata,
                        NSDictionary *compileOpts, NSDictionary *loadOpts,
                        NSDictionary *optionsPlist,
                        NSError **outErr) {
    NSString *mil = make_mil(CH, SP);
    NSData *md = [mil dataUsingEncoding:NSUTF8StringEncoding];

    // Create descriptor — try with optionsPlist
    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D,
        @selector(modelWithMILText:weights:optionsPlist:),
        md, @{@"@model_path/weights/weight.bin": @{@"offset":@0, @"data":wdata}},
        optionsPlist);
    if (!desc) { if (outErr) *outErr = [NSError errorWithDomain:@"ANE" code:1 userInfo:@{NSLocalizedDescriptionKey:@"descriptor nil"}]; return nil; }

    id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(g_I, @selector(inMemoryModelWithDescriptor:), desc);
    if (!mdl) { if (outErr) *outErr = [NSError errorWithDomain:@"ANE" code:2 userInfo:@{NSLocalizedDescriptionKey:@"model nil"}]; return nil; }

    // Write temp files (required by compiler)
    id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    [[NSFileManager defaultManager] createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
        withIntermediateDirectories:YES attributes:nil error:nil];
    [md writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    [wdata writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

    NSError *e = nil;
    BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
        mdl, @selector(compileWithQoS:options:error:), 21, compileOpts ?: @{}, &e);
    if (!ok) {
        if (outErr) *outErr = e;
        [[NSFileManager defaultManager] removeItemAtPath:td error:nil];
        return nil;
    }

    e = nil;
    ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
        mdl, @selector(loadWithQoS:options:error:), 21, loadOpts ?: @{}, &e);
    if (!ok) {
        if (outErr) *outErr = e;
        return nil;
    }

    return mdl;
}

// ── Benchmark: run N evals, return avg latency in µs ──
static double benchmark(id mdl, id req, int N) {
    NSError *e = nil;
    // warmup
    for (int i = 0; i < 10; i++)
        ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            mdl, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);

    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < N; i++) {
        ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            mdl, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
    }
    return tb_us(mach_absolute_time() - t0) / N;
}

// ── Benchmark with eval options ──
static double benchmark_with_eval_opts(id mdl, id req, int N, NSDictionary *evalOpts) {
    NSError *e = nil;
    for (int i = 0; i < 10; i++)
        ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            mdl, @selector(evaluateWithQoS:options:request:error:), 21, evalOpts, req, &e);

    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < N; i++) {
        ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            mdl, @selector(evaluateWithQoS:options:request:error:), 21, evalOpts, req, &e);
    }
    return tb_us(mach_absolute_time() - t0) / N;
}

// ── Dump methods of a class matching substring ──
static void dump_methods(Class cls, const char *substr, BOOL instance) {
    unsigned int count = 0;
    Method *methods = instance ? class_copyMethodList(cls, &count) : class_copyMethodList(object_getClass(cls), &count);
    for (unsigned int i = 0; i < count; i++) {
        const char *name = sel_getName(method_getName(methods[i]));
        if (!substr || strstr(name, substr)) {
            printf("    %c%s\n", instance ? '-' : '+', name);
        }
    }
    free(methods);
}

int main() {
    @autoreleasepool {
        setbuf(stdout, NULL);
        mach_timebase_info(&g_tb);

        // Load frameworks
        void *h1 = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
        void *h2 = dlopen("/System/Library/PrivateFrameworks/ANECompiler.framework/ANECompiler", RTLD_NOW);

        printf("\n");
        printf("  ██████ ANE COMPILER OPTIONS PROBE ██████\n\n");

        Class g_D  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class g_I  = NSClassFromString(@"_ANEInMemoryModel");
        Class g_AR = NSClassFromString(@"_ANERequest");
        Class g_AIO= NSClassFromString(@"_ANEIOSurfaceObject");
        Class g_Str= NSClassFromString(@"_ANEStrings");
        Class g_MIP= NSClassFromString(@"_ANEModelInstanceParameters");

        // ════════════════════════════════════════════════
        // Section 1: Method dumps — compilation-related
        // ════════════════════════════════════════════════
        printf("  ── _ANEInMemoryModel: compile/load/option methods ──\n\n");
        printf("  Instance methods containing 'compil':\n");
        dump_methods(g_I, "compil", YES);
        printf("\n  Instance methods containing 'option':\n");
        dump_methods(g_I, "option", YES);
        printf("\n  Instance methods containing 'Option':\n");
        dump_methods(g_I, "Option", YES);
        printf("\n  Instance methods containing 'load':\n");
        dump_methods(g_I, "load", YES);
        printf("\n  Instance methods containing 'cache':\n");
        dump_methods(g_I, "cache", YES);
        printf("\n  Instance methods containing 'Cache':\n");
        dump_methods(g_I, "Cache", YES);
        printf("\n  Instance methods containing 'power':\n");
        dump_methods(g_I, "power", YES);
        printf("\n  Instance methods containing 'Power':\n");
        dump_methods(g_I, "Power", YES);
        printf("\n  Class methods containing 'compil':\n");
        dump_methods(g_I, "compil", NO);
        printf("\n  Class methods containing 'option':\n");
        dump_methods(g_I, "option", NO);

        // ── ALL instance methods of _ANEInMemoryModel ──
        printf("\n  ── ALL _ANEInMemoryModel instance methods ──\n\n");
        dump_methods(g_I, NULL, YES);

        printf("\n  ── ALL _ANEInMemoryModel class methods ──\n\n");
        dump_methods(g_I, NULL, NO);

        // ════════════════════════════════════════════════
        // Section 2: _ANEInMemoryModelDescriptor methods
        // ════════════════════════════════════════════════
        printf("\n  ── _ANEInMemoryModelDescriptor: ALL methods ──\n\n");
        printf("  Instance methods:\n");
        dump_methods(g_D, NULL, YES);
        printf("\n  Class methods:\n");
        dump_methods(g_D, NULL, NO);

        // ════════════════════════════════════════════════
        // Section 3: _ANEStrings
        // ════════════════════════════════════════════════
        printf("\n  ── _ANEStrings: ALL methods ──\n\n");
        if (g_Str) {
            printf("  Class methods:\n");
            dump_methods(g_Str, NULL, NO);
            printf("\n  Instance methods:\n");
            dump_methods(g_Str, NULL, YES);

            // Call every class method that returns a string (no args)
            printf("\n  String constants from _ANEStrings:\n");
            unsigned int cnt = 0;
            Method *methods = class_copyMethodList(object_getClass(g_Str), &cnt);
            for (unsigned int i = 0; i < cnt; i++) {
                SEL s = method_getName(methods[i]);
                const char *name = sel_getName(s);
                // Only call zero-arg methods
                if (strchr(name, ':') == NULL) {
                    @try {
                        id val = ((id(*)(Class,SEL))objc_msgSend)(g_Str, s);
                        if (val && [val isKindOfClass:[NSString class]]) {
                            printf("    %-50s = \"%s\"\n", name, [(NSString*)val UTF8String]);
                        } else if (val) {
                            printf("    %-50s = %s\n", name, [[val description] UTF8String]);
                        }
                    } @catch (NSException *ex) {
                        printf("    %-50s = <exception: %s>\n", name, [[ex reason] UTF8String]);
                    }
                }
            }
            free(methods);
        } else {
            printf("  _ANEStrings: NOT FOUND\n");
        }

        // ════════════════════════════════════════════════
        // Section 4: _ANEModelInstanceParameters
        // ════════════════════════════════════════════════
        printf("\n  ── _ANEModelInstanceParameters: ALL methods ──\n\n");
        if (g_MIP) {
            printf("  Instance methods:\n");
            dump_methods(g_MIP, NULL, YES);
            printf("\n  Class methods:\n");
            dump_methods(g_MIP, NULL, NO);
        } else {
            printf("  _ANEModelInstanceParameters: NOT FOUND\n");
        }

        // ════════════════════════════════════════════════
        // Section 5: Read kANEF* string constants from framework
        // ════════════════════════════════════════════════
        printf("\n  ── kANEF* Key Constants (from framework exports) ──\n\n");

        // These are exported as C string pointers
        const char *key_syms[] = {
            "kANEFCompilerOptionsFilenameKey",
            "kANEFEnablePowerSavingKey",
            "kANEFEnableLateLatchKey",
            "kANEFKeepModelMemoryWiredKey",
            "kANEFSkipPreparePhaseKey",
            "kANEFPerformanceStatsMaskKey",
            "kANEFDisableIOFencesUseSharedEventsKey",
            "kANEFEnableFWToFWSignal",
            "kANEFModelTypeKey",
            "kANEFModelMILValue",
            "kANEFModelMLIRValue",
            "kANEFModelPreCompiledValue",
            "kANEFModelANECIRValue",
            "kANEFModelCoreMLValue",
            "kANEFModelLLIRBundleValue",
            "kANEFModelDescriptionKey",
            "kANEFModelIdentityStrKey",
            "kANEFNetPlistFilenameKey",
            "kANEFModelLoadPerformanceStatsKey",
            "kANEFConstantSurfaceIDKey",
            "kANEFConstantSurfaceAlignmentKey",
            "kANEFIntermediateBufferHandleKey",
            "kANEFMemoryPoolIDKey",
            "kANEFEspressoFileResourcesKey",
            "kANEFModelProceduresArrayKey",
            "kANEFModelProcedureIDKey",
            "kANEFModelInputSymbolIndexArrayKey",
            "kANEFModelOutputSymbolIndexArrayKey",
            "kANEFModelInputSymbolsArrayKey",
            "kANEFModelOutputSymbolsArrayKey",
            "kANEFModelProcedureNameToIDMapKey",
            "kANEFModelProcedureNameToStatsSizeMapKey",
            "kANEFModelInput16KAlignmentArrayKey",
            "kANEFModelOutput16KAlignmentArrayKey",
            "kANEFModelIsEncryptedKey",
            "kANEFAOTCacheUrlIdentifierKey",
            "kANEFInMemoryModelIdentifierKey",
            "kANEFInMemoryModelIsCachedKey",
            "kANEFModelInstanceParameters",
            "kANEFBaseModelIdentifierKey",
            "kANEFCompilationInitiatedByE5MLKey",
            "kANEFModelHasCacheURLIdentifierKey",
            "kANEFModelCacheIdentifierUsingSourceURLKey",
            "kANEFRetainModelsWithoutSourceURLKey",
            "kANEFHintEnergyEfficientWorkloadKey",
            "kANEFHintReportResidentPagesKey",
            "kANEFHintReportSessionStatusKey",
            "kANEFHintReportTotalPagesKey",
            "kANEFHintSessionAbort",
            "kANEFHintSessionInfo",
            "kANEFHintSessionStart",
            "kANEFHintSessionStop",
            "kANEModelKeyAllSegmentsValue",
            "kANEModelKeyEspressoTranslationOptions",
            "kANEModelKeyNoSegmentsValue",
            NULL
        };

        for (int i = 0; key_syms[i]; i++) {
            NSString *val = resolve_key(h1, key_syms[i]);
            if (val) {
                printf("    %-50s = \"%s\"\n", key_syms[i], [val UTF8String]);
            } else {
                printf("    %-50s = <not resolved>\n", key_syms[i]);
            }
        }

        // ════════════════════════════════════════════════
        // Section 6: ANECCreateCompilerOptionDictionary
        // ════════════════════════════════════════════════
        printf("\n  ── ANECCreateCompilerOptionDictionary / ANECCreateCompilerOptionsCFString ──\n\n");
        if (h2) {
            // These functions likely require arguments (model dict, etc.)
            // Just report if they're resolved; don't call with NULL args
            void *fn1 = dlsym(h2, "ANECCreateCompilerOptionDictionary");
            void *fn2 = dlsym(h2, "ANECCreateCompilerOptionsCFString");
            void *fn3 = dlsym(h2, "ANECCreateCompilerInputDictionary");
            void *fn4 = dlsym(h2, "ANECCompile");
            void *fn5 = dlsym(h2, "ANECCompileJIT");
            void *fn6 = dlsym(h2, "ANECCompileOnline");
            void *fn7 = dlsym(h2, "ANECCompileOffline");
            void *fn8 = dlsym(h2, "ANECCreateDeviceProperty");
            void *fn9 = dlsym(h2, "ANECGetDeviceProperty");
            printf("  ANECCreateCompilerOptionDictionary:  %s\n", fn1 ? "RESOLVED" : "not found");
            printf("  ANECCreateCompilerOptionsCFString:   %s\n", fn2 ? "RESOLVED" : "not found");
            printf("  ANECCreateCompilerInputDictionary:   %s\n", fn3 ? "RESOLVED" : "not found");
            printf("  ANECCompile:                         %s\n", fn4 ? "RESOLVED" : "not found");
            printf("  ANECCompileJIT:                      %s\n", fn5 ? "RESOLVED" : "not found");
            printf("  ANECCompileOnline:                   %s\n", fn6 ? "RESOLVED" : "not found");
            printf("  ANECCompileOffline:                  %s\n", fn7 ? "RESOLVED" : "not found");
            printf("  ANECCreateDeviceProperty:            %s\n", fn8 ? "RESOLVED" : "not found");
            printf("  ANECGetDeviceProperty:               %s\n", fn9 ? "RESOLVED" : "not found");

            // NOTE: These C functions require specific arguments (model dict, etc.)
            // Calling with wrong args causes SIGSEGV. Skipping direct calls.
            printf("  (Skipping direct calls — unknown arg signatures)\n");
        } else {
            printf("  ANECompiler framework not loaded\n");
        }

        // ════════════════════════════════════════════════
        // Section 7: Read compilerOptionsFileName from a model
        // ════════════════════════════════════════════════
        printf("\n  ── Model properties: compilerOptionsFileName, etc. ──\n\n");
        int CH = 512, SP = 64;
        NSData *wdata = make_weight_data(CH);
        NSError *err = nil;
        id mdl = compile_model(g_D, g_I, CH, SP, wdata, @{}, @{}, nil, &err);
        if (!mdl) {
            printf("  BASELINE compile failed: %s\n", err ? [[err description] UTF8String] : "unknown");
            return 1;
        }

        // Read various properties
        SEL propSels[] = {
            @selector(compilerOptionsFileName),
            @selector(compilerOptions),
            @selector(compilerOptionsWithOptions:isCompiledModelCached:),
            @selector(modelDescription),
            @selector(modelType),
            @selector(modelIdentityStr),
            @selector(isCompiledModelCached),
            @selector(compiledModelCacheIdentifier),
            NULL
        };
        const char *propNames[] = {
            "compilerOptionsFileName",
            "compilerOptions",
            "compilerOptionsWithOptions:isCompiledModelCached:",
            "modelDescription",
            "modelType",
            "modelIdentityStr",
            "isCompiledModelCached",
            "compiledModelCacheIdentifier",
            NULL
        };

        for (int i = 0; propSels[i]; i++) {
            if ([mdl respondsToSelector:propSels[i]]) {
                @try {
                    // Special: compilerOptionsWithOptions:isCompiledModelCached: takes args
                    if (propSels[i] == @selector(compilerOptionsWithOptions:isCompiledModelCached:)) {
                        // Try with empty dict, cached=NO
                        id result = ((id(*)(id,SEL,id,BOOL))objc_msgSend)(mdl, propSels[i], @{}, NO);
                        printf("    %-45s (opts=@{}, cached=NO): %s\n", propNames[i],
                            result ? [[result description] UTF8String] : "nil");
                        // Try cached=YES
                        result = ((id(*)(id,SEL,id,BOOL))objc_msgSend)(mdl, propSels[i], @{}, YES);
                        printf("    %-45s (opts=@{}, cached=YES): %s\n", propNames[i],
                            result ? [[result description] UTF8String] : "nil");
                    } else {
                        id result = ((id(*)(id,SEL))objc_msgSend)(mdl, propSels[i]);
                        NSString *desc = result ? [result description] : @"nil";
                        if ([desc length] > 500) desc = [[desc substringToIndex:500] stringByAppendingString:@"..."];
                        printf("    %-45s = %s\n", propNames[i], [desc UTF8String]);
                    }
                } @catch (NSException *ex) {
                    printf("    %-45s = <exception: %s>\n", propNames[i], [[ex reason] UTF8String]);
                }
            } else {
                printf("    %-45s = <not implemented>\n", propNames[i]);
            }
        }

        // Unload baseline
        ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl, @selector(unloadWithQoS:error:), 21, &err);

        // ════════════════════════════════════════════════
        // Section 8: Benchmark — compile options
        // ════════════════════════════════════════════════
        printf("\n  ── Compile Option Benchmarks (512x512 sp64, 200 evals each) ──\n\n");

        int ioBytes = CH * SP * 4;
        IOSurfaceRef ioIn = make_surface(ioBytes);
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

        // Read the actual kANEF key values
        NSString *kPowerSave = resolve_key(h1, "kANEFEnablePowerSavingKey");
        NSString *kLateLatch = resolve_key(h1, "kANEFEnableLateLatchKey");
        NSString *kWired = resolve_key(h1, "kANEFKeepModelMemoryWiredKey");
        NSString *kSkipPrep = resolve_key(h1, "kANEFSkipPreparePhaseKey");
        NSString *kPerfMask = resolve_key(h1, "kANEFPerformanceStatsMaskKey");
        NSString *kDisableFences = resolve_key(h1, "kANEFDisableIOFencesUseSharedEventsKey");
        NSString *kFW2FW = resolve_key(h1, "kANEFEnableFWToFWSignal");
        NSString *kModelType = resolve_key(h1, "kANEFModelTypeKey");
        NSString *kMILValue = resolve_key(h1, "kANEFModelMILValue");
        NSString *kEnergyEfficient = resolve_key(h1, "kANEFHintEnergyEfficientWorkloadKey");

        int N = 200;
        double baseline_us = 0;

        // Define test configurations
        typedef struct {
            const char *name;
            NSDictionary *compileOpts;
            NSDictionary *loadOpts;
            NSDictionary *optionsPlist;
            NSDictionary *evalOpts;
        } TestConfig;

        NSMutableArray *configs = [NSMutableArray array];

        // Config 0: Baseline (empty dicts everywhere)
        [configs addObject:@[@"Baseline @{}", @{}, @{}, [NSNull null], @{}]];

        // Configs using actual kANEF keys
        if (kPowerSave) {
            [configs addObject:@[@"compile: EnablePowerSaving=YES", @{kPowerSave: @YES}, @{}, [NSNull null], @{}]];
            [configs addObject:@[@"compile: EnablePowerSaving=NO", @{kPowerSave: @NO}, @{}, [NSNull null], @{}]];
            [configs addObject:@[@"load: EnablePowerSaving=YES", @{}, @{kPowerSave: @YES}, [NSNull null], @{}]];
            [configs addObject:@[@"load: EnablePowerSaving=NO", @{}, @{kPowerSave: @NO}, [NSNull null], @{}]];
        }

        if (kLateLatch) {
            [configs addObject:@[@"compile: EnableLateLatch=YES", @{kLateLatch: @YES}, @{}, [NSNull null], @{}]];
            [configs addObject:@[@"compile: EnableLateLatch=NO", @{kLateLatch: @NO}, @{}, [NSNull null], @{}]];
        }

        if (kWired) {
            [configs addObject:@[@"load: KeepMemoryWired=YES", @{}, @{kWired: @YES}, [NSNull null], @{}]];
            [configs addObject:@[@"load: KeepMemoryWired=NO", @{}, @{kWired: @NO}, [NSNull null], @{}]];
        }

        if (kSkipPrep) {
            [configs addObject:@[@"compile: SkipPreparePhase=YES", @{kSkipPrep: @YES}, @{}, [NSNull null], @{}]];
        }

        if (kPerfMask) {
            [configs addObject:@[@"compile: PerfStatsMask=0xFF", @{kPerfMask: @0xFF}, @{}, [NSNull null], @{}]];
        }

        if (kDisableFences) {
            [configs addObject:@[@"load: DisableIOFences=YES", @{}, @{kDisableFences: @YES}, [NSNull null], @{}]];
            [configs addObject:@[@"load: DisableIOFences=NO", @{}, @{kDisableFences: @NO}, [NSNull null], @{}]];
        }

        if (kFW2FW) {
            [configs addObject:@[@"compile: FWToFWSignal=YES", @{kFW2FW: @YES}, @{}, [NSNull null], @{}]];
        }

        if (kEnergyEfficient) {
            [configs addObject:@[@"load: EnergyEfficient=YES", @{}, @{kEnergyEfficient: @YES}, [NSNull null], @{}]];
            [configs addObject:@[@"load: EnergyEfficient=NO", @{}, @{kEnergyEfficient: @NO}, [NSNull null], @{}]];
        }

        // Try speculative / common option keys
        [configs addObject:@[@"compile: optimization_level=1", @{@"optimization_level": @1}, @{}, [NSNull null], @{}]];
        [configs addObject:@[@"compile: optimization_level=3", @{@"optimization_level": @3}, @{}, [NSNull null], @{}]];
        [configs addObject:@[@"compile: optimize=YES", @{@"optimize": @YES}, @{}, [NSNull null], @{}]];
        [configs addObject:@[@"compile: target=h15g", @{@"target": @"h15g"}, @{}, [NSNull null], @{}]];

        // Try optionsPlist on descriptor — expects NSData (plist bytes), not NSDictionary
        // Serialize plists as NSData for the descriptor
        {
            NSDictionary *emptyPlist = @{};
            NSData *emptyData = [NSPropertyListSerialization dataWithPropertyList:emptyPlist
                format:NSPropertyListBinaryFormat_v1_0 options:0 error:nil];
            [configs addObject:@[@"plist: empty dict (as data)", @{}, @{}, emptyData ?: [NSNull null], @{}]];

            if (kModelType && kMILValue) {
                NSDictionary *typePlist = @{kModelType: kMILValue};
                NSData *typeData = [NSPropertyListSerialization dataWithPropertyList:typePlist
                    format:NSPropertyListBinaryFormat_v1_0 options:0 error:nil];
                [configs addObject:@[@"plist: modelType=MIL (as data)", @{}, @{}, typeData ?: [NSNull null], @{}]];
            }

            // Try with known compiler-relevant keys in the plist
            if (kPowerSave) {
                NSDictionary *psPlist = @{kPowerSave: @NO};
                NSData *psData = [NSPropertyListSerialization dataWithPropertyList:psPlist
                    format:NSPropertyListBinaryFormat_v1_0 options:0 error:nil];
                [configs addObject:@[@"plist: PowerSave=NO (as data)", @{}, @{}, psData ?: [NSNull null], @{}]];
            }
        }

        // Eval options
        if (kPowerSave) {
            [configs addObject:@[@"eval: EnablePowerSaving=YES", @{}, @{}, [NSNull null], @{kPowerSave: @YES}]];
            [configs addObject:@[@"eval: EnablePowerSaving=NO", @{}, @{}, [NSNull null], @{kPowerSave: @NO}]];
        }
        if (kLateLatch) {
            [configs addObject:@[@"eval: EnableLateLatch=YES", @{}, @{}, [NSNull null], @{kLateLatch: @YES}]];
        }
        if (kDisableFences) {
            [configs addObject:@[@"eval: DisableIOFences=YES", @{}, @{}, [NSNull null], @{kDisableFences: @YES}]];
        }

        // Combined power tuning
        if (kPowerSave && kLateLatch) {
            NSDictionary *allPerf = @{kPowerSave: @NO, kLateLatch: @YES};
            [configs addObject:@[@"compile: PowerSave=NO+LateLatch=YES", allPerf, @{}, [NSNull null], @{}]];
        }
        if (kWired && kDisableFences) {
            NSDictionary *loadPerf = @{kWired: @YES, kDisableFences: @YES};
            [configs addObject:@[@"load: Wired=YES+NoFences=YES", @{}, loadPerf, [NSNull null], @{}]];
        }

        printf("  %-50s  %10s  %10s  %s\n", "Configuration", "Lat (us)", "vs Base", "Status");
        printf("  %-50s  %10s  %10s  %s\n",
            "--------------------------------------------------", "----------", "----------", "--------");

        for (NSArray *cfg in configs) {
            NSString *name = cfg[0];
            NSDictionary *compOpts = cfg[1];
            NSDictionary *loadOpts = cfg[2];
            NSDictionary *plist = ([cfg[3] isEqual:[NSNull null]]) ? nil : cfg[3];
            NSDictionary *evalOpts = cfg[4];

            err = nil;
            id testMdl = compile_model(g_D, g_I, CH, SP, wdata, compOpts, loadOpts, plist, &err);
            if (!testMdl) {
                printf("  %-50s  %10s  %10s  COMPILE FAIL: %s\n",
                    [name UTF8String], "-", "-",
                    err ? [[[err localizedDescription] substringToIndex:MIN(60, [[err localizedDescription] length])] UTF8String] : "unknown");
                continue;
            }

            double lat = 0;
            @try {
                if ([evalOpts count] > 0) {
                    lat = benchmark_with_eval_opts(testMdl, req, N, evalOpts);
                } else {
                    lat = benchmark(testMdl, req, N);
                }
            } @catch (NSException *ex) {
                printf("  %-50s  %10s  %10s  EVAL FAIL: %s\n",
                    [name UTF8String], "-", "-", [[ex reason] UTF8String]);
                ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(testMdl, @selector(unloadWithQoS:error:), 21, &err);
                continue;
            }

            if (baseline_us == 0) baseline_us = lat;
            double pct = ((lat - baseline_us) / baseline_us) * 100.0;
            const char *status = (pct < -2.0) ? "FASTER" : (pct > 2.0) ? "SLOWER" : "~SAME";

            printf("  %-50s  %10.1f  %+9.1f%%  %s\n",
                [name UTF8String], lat, pct, status);

            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(testMdl, @selector(unloadWithQoS:error:), 21, &err);
        }

        // ════════════════════════════════════════════════
        // Section 9: Probe compilerOptionsWithOptions for known option dicts
        // ════════════════════════════════════════════════
        printf("\n  ── compilerOptionsWithOptions: probing known keys ──\n\n");

        // Compile a fresh model for probing
        err = nil;
        id probeMdl = compile_model(g_D, g_I, CH, SP, wdata, @{}, @{}, nil, &err);
        if (probeMdl && [probeMdl respondsToSelector:@selector(compilerOptionsWithOptions:isCompiledModelCached:)]) {
            NSDictionary *testDicts[] = {
                @{},
                @{@"optimization_level": @0},
                @{@"optimization_level": @1},
                @{@"optimization_level": @2},
                @{@"optimization_level": @3},
                @{@"target": @"h15g"},
                NULL
            };
            const char *testNames[] = {
                "empty",
                "optimization_level=0",
                "optimization_level=1",
                "optimization_level=2",
                "optimization_level=3",
                "target=h15g",
            };

            // Add kANEF keys if resolved
            NSMutableArray *extraDicts = [NSMutableArray array];
            NSMutableArray *extraNames = [NSMutableArray array];

            if (kPowerSave) {
                [extraDicts addObject:@{kPowerSave: @YES}];
                [extraNames addObject:@"EnablePowerSaving=YES"];
                [extraDicts addObject:@{kPowerSave: @NO}];
                [extraNames addObject:@"EnablePowerSaving=NO"];
            }
            if (kLateLatch) {
                [extraDicts addObject:@{kLateLatch: @YES}];
                [extraNames addObject:@"EnableLateLatch=YES"];
            }
            if (kSkipPrep) {
                [extraDicts addObject:@{kSkipPrep: @YES}];
                [extraNames addObject:@"SkipPreparePhase=YES"];
            }

            for (int i = 0; testDicts[i]; i++) {
                @try {
                    id result = ((id(*)(id,SEL,id,BOOL))objc_msgSend)(
                        probeMdl, @selector(compilerOptionsWithOptions:isCompiledModelCached:), testDicts[i], NO);
                    NSString *desc = result ? [result description] : @"nil";
                    if ([desc length] > 300) desc = [[desc substringToIndex:300] stringByAppendingString:@"..."];
                    printf("    opts=%-30s cached=NO -> %s\n", testNames[i], [desc UTF8String]);
                } @catch (NSException *ex) {
                    printf("    opts=%-30s cached=NO -> EXCEPTION: %s\n", testNames[i], [[ex reason] UTF8String]);
                }
            }

            for (NSUInteger i = 0; i < extraDicts.count; i++) {
                @try {
                    id result = ((id(*)(id,SEL,id,BOOL))objc_msgSend)(
                        probeMdl, @selector(compilerOptionsWithOptions:isCompiledModelCached:), extraDicts[i], NO);
                    NSString *desc = result ? [result description] : @"nil";
                    if ([desc length] > 300) desc = [[desc substringToIndex:300] stringByAppendingString:@"..."];
                    printf("    opts=%-30s cached=NO -> %s\n", [extraNames[i] UTF8String], [desc UTF8String]);
                } @catch (NSException *ex) {
                    printf("    opts=%-30s cached=NO -> EXCEPTION: %s\n", [extraNames[i] UTF8String], [[ex reason] UTF8String]);
                }
            }

            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(probeMdl, @selector(unloadWithQoS:error:), 21, &err);
        } else {
            printf("  compilerOptionsWithOptions:isCompiledModelCached: not available\n");
        }

        // ════════════════════════════════════════════════
        // Section 10: Other interesting classes
        // ════════════════════════════════════════════════
        printf("\n  ── Other ANE classes of interest ──\n\n");

        const char *otherClasses[] = {
            "_ANEModel", "_ANEProgramForEvaluation", "_ANEDeviceController",
            "_ANEQoSMapper", "_ANEWeight", "_ANECloneHelper",
            NULL
        };
        for (int i = 0; otherClasses[i]; i++) {
            Class cls = NSClassFromString([NSString stringWithUTF8String:otherClasses[i]]);
            if (cls) {
                printf("  %s methods containing 'option/Option/compil/cache/perf/power':\n", otherClasses[i]);
                const char *subs[] = {"option", "Option", "compil", "cache", "Cache", "perf", "Perf", "power", "Power", "wired", "Wired", "latch", "Latch", NULL};
                BOOL found = NO;
                for (int j = 0; subs[j]; j++) {
                    unsigned int cnt = 0;
                    Method *m = class_copyMethodList(cls, &cnt);
                    for (unsigned int k = 0; k < cnt; k++) {
                        if (strstr(sel_getName(method_getName(m[k])), subs[j])) {
                            printf("    -%s\n", sel_getName(method_getName(m[k])));
                            found = YES;
                        }
                    }
                    free(m);
                    m = class_copyMethodList(object_getClass(cls), &cnt);
                    for (unsigned int k = 0; k < cnt; k++) {
                        if (strstr(sel_getName(method_getName(m[k])), subs[j])) {
                            printf("    +%s\n", sel_getName(method_getName(m[k])));
                            found = YES;
                        }
                    }
                    free(m);
                }
                if (!found) printf("    (none found)\n");
                printf("\n");
            }
        }

        // Cleanup
        CFRelease(ioIn);
        CFRelease(ioOut);

        printf("\n  ██████ PROBE COMPLETE ██████\n\n");
    }
    return 0;
}
