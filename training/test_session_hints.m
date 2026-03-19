// test_session_hints.m — Probe _ANESharedEvents, sessionHint, _ANEQoSMapper, modelAttributes
// Goal: find hidden ANE configuration for performance or new capabilities
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>

static mach_timebase_info_data_t g_tb;
static double tb_ms(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }
static double tb_us(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e3; }

// ── Dump all methods/properties of a class ──
static void dump_class(const char *name) {
    Class cls = NSClassFromString([NSString stringWithUTF8String:name]);
    if (!cls) { printf("  %s: NOT FOUND\n", name); return; }
    printf("\n=== %s ===\n", name);

    // Superclass
    Class super = class_getSuperclass(cls);
    if (super) printf("  Superclass: %s\n", class_getName(super));

    unsigned int count;
    Method *methods = class_copyMethodList(object_getClass(cls), &count);
    if (count) printf("  Class methods (%u):\n", count);
    for (unsigned int i = 0; i < count; i++) {
        SEL s = method_getName(methods[i]);
        const char *enc = method_getTypeEncoding(methods[i]);
        printf("    + %s  [%s]\n", sel_getName(s), enc ? enc : "?");
    }
    free(methods);

    methods = class_copyMethodList(cls, &count);
    if (count) printf("  Instance methods (%u):\n", count);
    for (unsigned int i = 0; i < count; i++) {
        SEL s = method_getName(methods[i]);
        const char *enc = method_getTypeEncoding(methods[i]);
        printf("    - %s  [%s]\n", sel_getName(s), enc ? enc : "?");
    }
    free(methods);

    unsigned int pcount;
    objc_property_t *props = class_copyPropertyList(cls, &pcount);
    if (pcount) printf("  Properties (%u):\n", pcount);
    for (unsigned int i = 0; i < pcount; i++) {
        const char *pname = property_getName(props[i]);
        const char *pattr = property_getAttributes(props[i]);
        printf("    @property %s  [%s]\n", pname, pattr ? pattr : "?");
    }
    free(props);

    // Ivars
    unsigned int icount;
    Ivar *ivars = class_copyIvarList(cls, &icount);
    if (icount) printf("  Ivars (%u):\n", icount);
    for (unsigned int i = 0; i < icount; i++) {
        const char *iname = ivar_getName(ivars[i]);
        const char *itype = ivar_getTypeEncoding(ivars[i]);
        printf("    %s  [%s]\n", iname, itype ? itype : "?");
    }
    free(ivars);

    // Protocols
    unsigned int protoCount;
    Protocol * __unsafe_unretained *protos = class_copyProtocolList(cls, &protoCount);
    if (protoCount) {
        printf("  Protocols (%u):", protoCount);
        for (unsigned int i = 0; i < protoCount; i++)
            printf(" %s", protocol_getName(protos[i]));
        printf("\n");
    }
    free(protos);
}

static IOSurfaceRef make_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0});
}

// ── Compile a kernel with given dimensions ──
static id compile_kernel(int CH, int SP) {
    Class g_D  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    Class g_I  = NSClassFromString(@"_ANEInMemoryModel");

    _Float16 *w = (_Float16*)calloc(CH*CH, sizeof(_Float16));
    for (int i = 0; i < CH; i++) w[i*CH+i] = (_Float16)1.0f;
    int ws = CH*CH*2, tot = 128+ws;
    uint8_t *blob = (uint8_t*)calloc(tot,1);
    blob[0]=1; blob[4]=2; blob[64]=0xEF; blob[65]=0xBE; blob[66]=0xAD; blob[67]=0xDE; blob[68]=1;
    *(uint32_t*)(blob+72)=ws; *(uint32_t*)(blob+80)=128;
    memcpy(blob+128, w, ws);
    NSData *wdata = [NSData dataWithBytesNoCopy:blob length:tot freeWhenDone:YES];
    free(w);

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
    BOOL cOK = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e);
    if (!cOK) { printf("  COMPILE FAILED: %s\n", [[e description] UTF8String]); return nil; }
    BOOL lOK = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
    if (!lOK) { printf("  LOAD FAILED: %s\n", [[e description] UTF8String]); return nil; }
    return mdl;
}

// ── Quick benchmark: returns ms/eval ──
static double bench(id mdl, id req, int N) {
    NSError *e = nil;
    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < N; i++) {
        ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            mdl, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
    }
    return tb_ms(mach_absolute_time() - t0) / N;
}

// ── Try reading a property safely ──
static void try_read_property(id obj, const char *propName) {
    @try {
        id val = [obj valueForKey:[NSString stringWithUTF8String:propName]];
        printf("    %s = %s\n", propName, val ? [[val description] UTF8String] : "nil");
    } @catch (NSException *ex) {
        printf("    %s = <exception: %s>\n", propName, [[ex reason] UTF8String]);
    }
}

// ── Scan for ANE-related classes ──
static void scan_ane_classes(void) {
    printf("\n=== All ANE-related classes ===\n");
    unsigned int classCount = 0;
    Class *classes = objc_copyClassList(&classCount);
    int found = 0;
    for (unsigned int i = 0; i < classCount; i++) {
        const char *name = class_getName(classes[i]);
        if (name && (strstr(name, "ANE") || strstr(name, "Ane") || strstr(name, "ane"))) {
            printf("  %s\n", name);
            found++;
        }
    }
    free(classes);
    printf("  Total ANE classes: %d\n", found);
}

int main() {
    @autoreleasepool {
        setbuf(stdout, NULL);
        mach_timebase_info(&g_tb);
        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);

        printf("\n  ====================================\n");
        printf("  ANE SESSION HINTS / EVENTS / QoS PROBE\n");
        printf("  ====================================\n");

        // ════════════════════════════════════════════════════════════
        // SECTION A: _ANESharedEvents
        // ════════════════════════════════════════════════════════════
        printf("\n\n");
        printf("  ╔═══════════════════════════════════════╗\n");
        printf("  ║  SECTION A: _ANESharedEvents          ║\n");
        printf("  ╚═══════════════════════════════════════╝\n");

        dump_class("_ANESharedEvents");
        dump_class("_ANEEvent");
        dump_class("_ANEEventLogger");
        dump_class("_ANEEventCounter");

        // Try to instantiate _ANESharedEvents
        printf("\n  -- Instantiation tests --\n");
        Class seClass = NSClassFromString(@"_ANESharedEvents");
        id sharedEvents = nil;
        if (seClass) {
            // Try +sharedEvents or +sharedInstance first
            @try {
                if ([seClass respondsToSelector:@selector(sharedEvents)]) {
                    sharedEvents = ((id(*)(Class,SEL))objc_msgSend)(seClass, @selector(sharedEvents));
                    printf("  +sharedEvents: %s\n", sharedEvents ? [[sharedEvents description] UTF8String] : "nil");
                } else {
                    printf("  +sharedEvents: selector NOT found\n");
                }
            } @catch (NSException *ex) {
                printf("  +sharedEvents: exception: %s\n", [[ex reason] UTF8String]);
            }

            @try {
                if ([seClass respondsToSelector:@selector(sharedInstance)]) {
                    id si = ((id(*)(Class,SEL))objc_msgSend)(seClass, @selector(sharedInstance));
                    printf("  +sharedInstance: %s\n", si ? [[si description] UTF8String] : "nil");
                    if (!sharedEvents) sharedEvents = si;
                } else {
                    printf("  +sharedInstance: selector NOT found\n");
                }
            } @catch (NSException *ex) {
                printf("  +sharedInstance: exception: %s\n", [[ex reason] UTF8String]);
            }

            @try {
                id inst = [[seClass alloc] init];
                printf("  alloc/init: %s\n", inst ? [[inst description] UTF8String] : "nil");
                if (!sharedEvents) sharedEvents = inst;
            } @catch (NSException *ex) {
                printf("  alloc/init: exception: %s\n", [[ex reason] UTF8String]);
            }

            // Dump _ANESharedSignalEvent and _ANESharedWaitEvent first (they're needed for SharedEvents)
            Class sigClass = NSClassFromString(@"_ANESharedSignalEvent");
            Class waitClass = NSClassFromString(@"_ANESharedWaitEvent");
            if (sigClass) dump_class("_ANESharedSignalEvent");
            if (waitClass) dump_class("_ANESharedWaitEvent");

            // Try creating signal/wait events first
            id sigEvent = nil;
            id waitEvent = nil;
            if (sigClass) {
                @try {
                    sigEvent = [[sigClass alloc] init];
                    printf("  _ANESharedSignalEvent alloc/init: %s\n", sigEvent ? [[sigEvent description] UTF8String] : "nil");
                } @catch (NSException *ex) {
                    printf("  signal event init: exception: %s\n", [[ex reason] UTF8String]);
                }
            }
            if (waitClass) {
                @try {
                    waitEvent = [[waitClass alloc] init];
                    printf("  _ANESharedWaitEvent alloc/init: %s\n", waitEvent ? [[waitEvent description] UTF8String] : "nil");
                } @catch (NSException *ex) {
                    printf("  wait event init: exception: %s\n", [[ex reason] UTF8String]);
                }
            }

            // Try initWithSignalEvents:waitEvents: with actual event objects (not empty)
            if (sigEvent || waitEvent) {
                NSArray *sigs = sigEvent ? @[sigEvent] : @[];
                NSArray *waits = waitEvent ? @[waitEvent] : @[];
                @try {
                    id inst2 = ((id(*)(id,SEL,id,id))objc_msgSend)(
                        [seClass alloc], @selector(initWithSignalEvents:waitEvents:), sigs, waits);
                    printf("  initWithSignalEvents:waitEvents: %s\n",
                        inst2 ? [[inst2 description] UTF8String] : "nil");
                    if (!sharedEvents) sharedEvents = inst2;
                } @catch (NSException *ex) {
                    printf("  initWithSignalEvents: exception: %s\n", [[ex reason] UTF8String]);
                }
            } else {
                printf("  Skipping SharedEvents construction (no signal/wait events available)\n");
            }

            // Read all properties if we have an instance
            if (sharedEvents) {
                printf("\n  -- SharedEvents properties --\n");
                unsigned int pcount;
                objc_property_t *props = class_copyPropertyList(seClass, &pcount);
                for (unsigned int i = 0; i < pcount; i++) {
                    try_read_property(sharedEvents, property_getName(props[i]));
                }
                free(props);
            }
        } else {
            printf("  _ANESharedEvents: CLASS NOT FOUND\n");
        }

        // ════════════════════════════════════════════════════════════
        // SECTION B: Session Hints
        // ════════════════════════════════════════════════════════════
        printf("\n\n");
        printf("  ╔═══════════════════════════════════════╗\n");
        printf("  ║  SECTION B: Session Hints             ║\n");
        printf("  ╚═══════════════════════════════════════╝\n");

        // First compile a model we can use
        int CH = 256, SP = 64;
        id mdl = compile_kernel(CH, SP);
        if (!mdl) {
            printf("  FATAL: Could not compile kernel\n");
            return 1;
        }

        // Create IO surfaces
        int ioBytes = CH * SP * 4;
        IOSurfaceRef ioIn = make_surface(ioBytes);
        IOSurfaceRef ioOut = make_surface(ioBytes);
        Class g_AIO = NSClassFromString(@"_ANEIOSurfaceObject");
        id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioIn);
        id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioOut);
        Class g_AR = NSClassFromString(@"_ANERequest");

        // Fill input
        IOSurfaceLock(ioIn, 0, NULL);
        float *inp = (float*)IOSurfaceGetBaseAddress(ioIn);
        for (int i = 0; i < CH*SP; i++) inp[i] = 1.0f;
        IOSurfaceUnlock(ioIn, 0, NULL);

        // Verify eval works
        id reqBase = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wI], @[@0], @[wO], @[@0], nil, nil, @0);
        NSError *e = nil;
        BOOL evalOK = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            mdl, @selector(evaluateWithQoS:options:request:error:), 21, @{}, reqBase, &e);
        printf("  Baseline eval: %s\n", evalOK ? "OK" : "FAIL");
        double baselineMs = bench(mdl, reqBase, 50);
        printf("  Baseline latency: %.3f ms/eval\n\n", baselineMs);

        // Get _ANEClient
        Class clientClass = NSClassFromString(@"_ANEClient");
        id client = nil;
        if (clientClass) {
            client = ((id(*)(Class,SEL))objc_msgSend)(clientClass, @selector(sharedConnection));
            printf("  _ANEClient sharedConnection: %s\n", client ? "obtained" : "nil");
        }

        // Dump _ANEClient session-related methods
        printf("\n  -- _ANEClient methods containing 'session' or 'hint' --\n");
        if (clientClass) {
            unsigned int count;
            // Check class methods
            Method *methods = class_copyMethodList(object_getClass(clientClass), &count);
            for (unsigned int i = 0; i < count; i++) {
                const char *sname = sel_getName(method_getName(methods[i]));
                if (strcasestr(sname, "session") || strcasestr(sname, "hint")) {
                    printf("    + %s  [%s]\n", sname, method_getTypeEncoding(methods[i]) ?: "?");
                }
            }
            free(methods);
            // Check instance methods
            methods = class_copyMethodList(clientClass, &count);
            for (unsigned int i = 0; i < count; i++) {
                const char *sname = sel_getName(method_getName(methods[i]));
                if (strcasestr(sname, "session") || strcasestr(sname, "hint")) {
                    printf("    - %s  [%s]\n", sname, method_getTypeEncoding(methods[i]) ?: "?");
                }
            }
            free(methods);
        }

        // Test sessionHintWithModel:hint:options:report:error:
        // Type encoding: B56@0:8@16@24@32@40^@48
        // hint is id (@24) - calls isEqualToString: on it, so it's an NSString
        // report is id (@40) - input param, not output pointer
        // error is NSError** (^@48)
        printf("\n  -- sessionHintWithModel tests (hint=NSString) --\n");
        if (client && [client respondsToSelector:@selector(sessionHintWithModel:hint:options:report:error:)]) {
            // Try various string hints
            NSString *hintStrings[] = {
                @"preload", @"prefetch", @"precompile", @"warmup", @"prepare",
                @"cache", @"persist", @"realtime", @"priority", @"low_latency",
                @"high_throughput", @"background", @"interactive", @"batch",
                @"reset", @"flush", @"evict", @"pin", @"unpin",
                @"power_on", @"power_off", @"turbo",
                @"0", @"1", @"2", @"3", @"4", @"5",
                @"default", @"utility", @"userInitiated", @"userInteractive",
            };
            int nhints = 31;

            for (int h = 0; h < nhints; h++) {
                @try {
                    NSError *err = nil;
                    BOOL ok = ((BOOL(*)(id,SEL,id,id,id,id,NSError**))objc_msgSend)(
                        client, @selector(sessionHintWithModel:hint:options:report:error:),
                        mdl, hintStrings[h], @{}, nil, &err);
                    printf("    hint=\"%s\": ok=%s", [hintStrings[h] UTF8String], ok ? "YES" : "NO ");
                    if (err) printf("  err=%s", [[err localizedDescription] UTF8String]);
                    printf("\n");

                    if (ok) {
                        // Bench to see if hint changed behavior
                        double ms = bench(mdl, reqBase, 30);
                        printf("      -> lat=%.3f ms/eval (baseline=%.3f)\n", ms, baselineMs);
                    }
                } @catch (NSException *ex) {
                    printf("    hint=\"%s\": EXCEPTION: %s\n", [hintStrings[h] UTF8String], [[ex reason] UTF8String]);
                }
            }

            // Try with report as mutable dict
            printf("\n  -- sessionHint with report dict --\n");
            NSMutableDictionary *report = [NSMutableDictionary dictionary];
            @try {
                NSError *err = nil;
                BOOL ok = ((BOOL(*)(id,SEL,id,id,id,id,NSError**))objc_msgSend)(
                    client, @selector(sessionHintWithModel:hint:options:report:error:),
                    mdl, @"preload", @{}, report, &err);
                printf("    with mutable report: ok=%s\n", ok ? "YES" : "NO");
                if ([report count] > 0) {
                    printf("    report contents: %s\n", [[report description] UTF8String]);
                }
                if (err) printf("    err=%s\n", [[err localizedDescription] UTF8String]);
            } @catch (NSException *ex) {
                printf("    EXCEPTION: %s\n", [[ex reason] UTF8String]);
            }

            // Try with nil model, hint only
            printf("\n  -- sessionHint with nil model --\n");
            @try {
                NSError *err = nil;
                BOOL ok = ((BOOL(*)(id,SEL,id,id,id,id,NSError**))objc_msgSend)(
                    client, @selector(sessionHintWithModel:hint:options:report:error:),
                    nil, @"preload", @{}, nil, &err);
                printf("    nil model: ok=%s", ok ? "YES" : "NO");
                if (err) printf("  err=%s", [[err localizedDescription] UTF8String]);
                printf("\n");
            } @catch (NSException *ex) {
                printf("    EXCEPTION: %s\n", [[ex reason] UTF8String]);
            }
        } else {
            printf("  sessionHintWithModel: NOT available\n");
        }

        // Also try sessionHint on the model directly
        printf("\n  -- Session hint on _ANEInMemoryModel --\n");
        if ([mdl respondsToSelector:@selector(setSessionHint:)]) {
            printf("  setSessionHint: available!\n");
            for (int h = 0; h < 11; h++) {
                @try {
                    ((void(*)(id,SEL,unsigned int))objc_msgSend)(mdl, @selector(setSessionHint:), (unsigned int)h);
                    unsigned int readBack = ((unsigned int(*)(id,SEL))objc_msgSend)(mdl, @selector(sessionHint));
                    printf("    set %d -> read %u\n", h, readBack);
                } @catch (NSException *ex) {
                    printf("    set %d -> EXCEPTION: %s\n", h, [[ex reason] UTF8String]);
                }
            }
        } else {
            printf("  setSessionHint: NOT available on model\n");
        }

        // Test if sharedEvents can be passed to request
        printf("\n  -- Passing sharedEvents to request --\n");
        if (sharedEvents) {
            // Check if _ANERequest has a sharedEvents property
            Class reqClass = NSClassFromString(@"_ANERequest");
            if (reqClass) {
                unsigned int count;
                Method *methods = class_copyMethodList(reqClass, &count);
                for (unsigned int i = 0; i < count; i++) {
                    const char *sname = sel_getName(method_getName(methods[i]));
                    if (strcasestr(sname, "event") || strcasestr(sname, "shared")) {
                        printf("    _ANERequest method: - %s\n", sname);
                    }
                }
                free(methods);

                // Try setSharedEvents: on request
                if ([reqBase respondsToSelector:@selector(setSharedEvents:)]) {
                    @try {
                        ((void(*)(id,SEL,id))objc_msgSend)(reqBase, @selector(setSharedEvents:), sharedEvents);
                        printf("    setSharedEvents on request: OK\n");
                        // Bench with events
                        double withEventsMs = bench(mdl, reqBase, 50);
                        printf("    Latency with sharedEvents: %.3f ms/eval (baseline: %.3f)\n", withEventsMs, baselineMs);
                    } @catch (NSException *ex) {
                        printf("    setSharedEvents: EXCEPTION: %s\n", [[ex reason] UTF8String]);
                    }
                } else {
                    printf("    setSharedEvents: not available on request\n");
                }
            }
        }

        // ════════════════════════════════════════════════════════════
        // SECTION C: _ANEQoSMapper
        // ════════════════════════════════════════════════════════════
        printf("\n\n");
        printf("  ╔═══════════════════════════════════════╗\n");
        printf("  ║  SECTION C: _ANEQoSMapper             ║\n");
        printf("  ╚═══════════════════════════════════════╝\n");

        dump_class("_ANEQoSMapper");
        dump_class("_ANEQoS");
        dump_class("ANEQoS");

        Class qosMapper = NSClassFromString(@"_ANEQoSMapper");
        if (qosMapper) {
            printf("\n  -- QoS Named Values --\n");
            // These are all class methods returning unsigned int (I)
            @try {
                unsigned int bg  = ((unsigned int(*)(Class,SEL))objc_msgSend)(qosMapper, @selector(aneBackgroundTaskQoS));
                unsigned int ut  = ((unsigned int(*)(Class,SEL))objc_msgSend)(qosMapper, @selector(aneUtilityTaskQoS));
                unsigned int df  = ((unsigned int(*)(Class,SEL))objc_msgSend)(qosMapper, @selector(aneDefaultTaskQoS));
                unsigned int ui  = ((unsigned int(*)(Class,SEL))objc_msgSend)(qosMapper, @selector(aneUserInitiatedTaskQoS));
                unsigned int uia = ((unsigned int(*)(Class,SEL))objc_msgSend)(qosMapper, @selector(aneUserInteractiveTaskQoS));
                unsigned int rt  = ((unsigned int(*)(Class,SEL))objc_msgSend)(qosMapper, @selector(aneRealTimeTaskQoS));
                printf("  aneBackgroundTaskQoS:       %u\n", bg);
                printf("  aneUtilityTaskQoS:          %u\n", ut);
                printf("  aneDefaultTaskQoS:          %u\n", df);
                printf("  aneUserInitiatedTaskQoS:    %u\n", ui);
                printf("  aneUserInteractiveTaskQoS:  %u\n", uia);
                printf("  aneRealTimeTaskQoS:         %u\n", rt);
            } @catch (NSException *ex) {
                printf("  Exception reading QoS values: %s\n", [[ex reason] UTF8String]);
            }

            // programPriorityForQoS: and qosForProgramPriority:
            printf("\n  -- programPriorityForQoS mapping --\n");
            printf("  %-8s -> %-8s\n", "QoS", "Priority");
            for (unsigned int q = 0; q < 40; q++) {
                @try {
                    int prio = ((int(*)(Class,SEL,unsigned int))objc_msgSend)(qosMapper, @selector(programPriorityForQoS:), q);
                    // Only print if non-zero or known QoS
                    if (prio != 0 || q == 0 || q == 9 || q == 17 || q == 21 || q == 25 || q == 33)
                        printf("  %-8u -> %-8d\n", q, prio);
                } @catch (NSException *ex) { }
            }

            printf("\n  -- qosForProgramPriority mapping --\n");
            printf("  %-10s -> %-8s\n", "Priority", "QoS");
            for (int p = -5; p < 20; p++) {
                @try {
                    unsigned int q = ((unsigned int(*)(Class,SEL,int))objc_msgSend)(qosMapper, @selector(qosForProgramPriority:), p);
                    printf("  %-10d -> %-8u\n", p, q);
                } @catch (NSException *ex) { }
            }

            // queueIndexForQoS:
            printf("\n  -- queueIndexForQoS mapping --\n");
            printf("  %-8s -> %-10s\n", "QoS", "QueueIndex");
            for (unsigned int q = 0; q < 40; q++) {
                @try {
                    unsigned long long qi = ((unsigned long long(*)(Class,SEL,unsigned int))objc_msgSend)(qosMapper, @selector(queueIndexForQoS:), q);
                    if (qi != 0 || q == 0 || q == 9 || q == 17 || q == 21 || q == 25 || q == 33)
                        printf("  %-8u -> %-10llu\n", q, qi);
                } @catch (NSException *ex) { }
            }

            // realTimeProgramPriority and realTimeQueueIndex
            printf("\n  -- Real-time values --\n");
            @try {
                int rtPrio = ((int(*)(Class,SEL))objc_msgSend)(qosMapper, @selector(realTimeProgramPriority));
                unsigned long long rtQI = ((unsigned long long(*)(Class,SEL))objc_msgSend)(qosMapper, @selector(realTimeQueueIndex));
                printf("  realTimeProgramPriority: %d\n", rtPrio);
                printf("  realTimeQueueIndex:      %llu\n", rtQI);
            } @catch (NSException *ex) {
                printf("  Exception: %s\n", [[ex reason] UTF8String]);
            }

            // dispatchQueueArrayByMappingPrioritiesWithTag:
            printf("\n  -- dispatchQueueArrayByMappingPrioritiesWithTag --\n");
            NSString *tags[] = {@"ane", @"com.apple.ane", @"ANE", @"test"};
            for (int t = 0; t < 4; t++) {
                @try {
                    id arr = ((id(*)(Class,SEL,id))objc_msgSend)(qosMapper, @selector(dispatchQueueArrayByMappingPrioritiesWithTag:), tags[t]);
                    printf("  tag=\"%s\": %s\n", [tags[t] UTF8String],
                        arr ? [[arr description] UTF8String] : "nil");
                } @catch (NSException *ex) {
                    printf("  tag=\"%s\": EXCEPTION\n", [tags[t] UTF8String]);
                }
            }
        }

        // Bench with different QoS values to see if there are hidden levels
        printf("\n  -- QoS Benchmark (beyond known 5 levels) --\n");
        printf("  %-6s  %s\n", "QoS", "ms/eval");
        printf("  %-6s  %s\n", "------", "--------");
        unsigned int qosValues[] = {0, 9, 17, 21, 25, 33, 0, 1, 2, 3, 4, 5, 8, 16, 32, 48, 64, 128, 255};
        const char *qosNames[] = {"0/bg", "9/util", "17/def", "21/uinit", "25/ui", "33/uact",
                                   "0", "1", "2", "3", "4", "5", "8", "16", "32", "48", "64", "128", "255"};
        for (int q = 0; q < 19; q++) {
            @try {
                NSError *err = nil;
                uint64_t t0 = mach_absolute_time();
                int N = 30;
                for (int i = 0; i < N; i++) {
                    ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                        mdl, @selector(evaluateWithQoS:options:request:error:), qosValues[q], @{}, reqBase, &err);
                }
                double ms = tb_ms(mach_absolute_time() - t0) / N;
                printf("  %-6s  %.3f%s\n", qosNames[q], ms, err ? " [error]" : "");
            } @catch (NSException *ex) {
                printf("  %-6s  EXCEPTION\n", qosNames[q]);
            }
        }

        // ════════════════════════════════════════════════════════════
        // SECTION D: Model Attributes
        // ════════════════════════════════════════════════════════════
        printf("\n\n");
        printf("  ╔═══════════════════════════════════════╗\n");
        printf("  ║  SECTION D: Model Attributes          ║\n");
        printf("  ╚═══════════════════════════════════════╝\n");

        // Dump _ANEModel and _ANEInMemoryModel attribute-related methods
        printf("\n  -- Attribute-related methods on _ANEInMemoryModel --\n");
        Class immClass = NSClassFromString(@"_ANEInMemoryModel");
        if (immClass) {
            unsigned int count;
            Method *methods = class_copyMethodList(immClass, &count);
            for (unsigned int i = 0; i < count; i++) {
                const char *sname = sel_getName(method_getName(methods[i]));
                if (strcasestr(sname, "attr") || strcasestr(sname, "queue") ||
                    strcasestr(sname, "state") || strcasestr(sname, "handle") ||
                    strcasestr(sname, "program") || strcasestr(sname, "update") ||
                    strcasestr(sname, "priority") || strcasestr(sname, "cache") ||
                    strcasestr(sname, "depth")) {
                    const char *enc = method_getTypeEncoding(methods[i]);
                    printf("    - %s  [%s]\n", sname, enc ? enc : "?");
                }
            }
            free(methods);
        }

        // Read default modelAttributes
        printf("\n  -- Default modelAttributes after compile/load --\n");
        @try {
            if ([mdl respondsToSelector:@selector(modelAttributes)]) {
                id attrs = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(modelAttributes));
                printf("  modelAttributes: %s\n", attrs ? [[attrs description] UTF8String] : "nil");
                if (attrs && [attrs isKindOfClass:[NSDictionary class]]) {
                    for (id key in attrs) {
                        printf("    [%s] = %s (class: %s)\n",
                            [[key description] UTF8String],
                            [[attrs[key] description] UTF8String],
                            class_getName([attrs[key] class]));
                    }
                }
            } else {
                printf("  modelAttributes: selector NOT available\n");
            }
        } @catch (NSException *ex) {
            printf("  modelAttributes: EXCEPTION: %s\n", [[ex reason] UTF8String]);
        }

        // Read other model properties
        printf("\n  -- Other model properties --\n");
        const char *modelProps[] = {
            "programHandle", "intermediateBufferHandle", "queueDepth",
            "state", "modelState", "priority", "identifier",
            "hexStringIdentifier", "compiledModelPath", "isLoaded",
            "perfStatsMask", "modelFileURL", "procedureNameIndexMap",
        };
        for (int i = 0; i < 13; i++) {
            try_read_property(mdl, modelProps[i]);
        }

        // Save original attributes
        id origAttrs = nil;
        @try { origAttrs = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(modelAttributes)); } @catch(id e) {}

        // Try setting custom attributes then restoring
        printf("\n  -- Setting custom modelAttributes --\n");
        NSDictionary *testAttrs[] = {
            @{@"priority": @1},
            @{@"priority": @0},
            @{@"priority": @2},
            @{@"cache": @YES},
            @{@"cache": @NO},
            @{@"persistent": @YES},
            @{@"batch_size": @4},
            @{@"prefetch": @YES},
            @{@"power_mode": @1},
            @{@"power_mode": @2},
        };
        const char *testAttrNames[] = {
            "{priority:1}", "{priority:0}", "{priority:2}",
            "{cache:YES}", "{cache:NO}",
            "{persistent:YES}", "{batch_size:4}", "{prefetch:YES}",
            "{power_mode:1}", "{power_mode:2}",
        };
        for (int i = 0; i < 10; i++) {
            @try {
                if ([mdl respondsToSelector:@selector(setModelAttributes:)]) {
                    ((void(*)(id,SEL,id))objc_msgSend)(mdl, @selector(setModelAttributes:), testAttrs[i]);
                    id readBack = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(modelAttributes));
                    printf("    set %s -> readback keys: ", testAttrNames[i]);
                    if (readBack && [readBack isKindOfClass:[NSDictionary class]]) {
                        printf("%s", [[[readBack allKeys] description] UTF8String]);
                    } else {
                        printf("%s", readBack ? "non-dict" : "nil");
                    }

                    // Quick bench (warmup then measure)
                    bench(mdl, reqBase, 5);  // warmup
                    double ms = bench(mdl, reqBase, 50);
                    printf("  lat=%.3f ms\n", ms);

                    // Restore original attrs
                    ((void(*)(id,SEL,id))objc_msgSend)(mdl, @selector(setModelAttributes:), origAttrs);
                } else {
                    printf("    setModelAttributes: NOT available\n");
                    break;
                }
            } @catch (NSException *ex) {
                printf("    set %s -> EXCEPTION: %s\n", testAttrNames[i], [[ex reason] UTF8String]);
                // Restore
                @try { ((void(*)(id,SEL,id))objc_msgSend)(mdl, @selector(setModelAttributes:), origAttrs); } @catch(id e2) {}
            }
        }

        // Restore attrs for subsequent tests
        @try { ((void(*)(id,SEL,id))objc_msgSend)(mdl, @selector(setModelAttributes:), origAttrs); } @catch(id e) {}

        // Test updateModelAttributes:state:programHandle:intermediateBufferHandle:queueDepth:
        printf("\n  -- updateModelAttributes:state:programHandle:intermediateBufferHandle:queueDepth: --\n");
        if ([mdl respondsToSelector:@selector(updateModelAttributes:state:programHandle:intermediateBufferHandle:queueDepth:)]) {
            printf("  Selector AVAILABLE\n");

            // Read current values first
            id curAttrs = nil;
            unsigned int curState = 0;
            unsigned long long curPH = 0, curIBH = 0;
            unsigned int curQD = 0;

            @try { curAttrs = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(modelAttributes)); } @catch(id e) {}
            @try { curState = ((unsigned int(*)(id,SEL))objc_msgSend)(mdl, @selector(state)); } @catch(id e) {}
            @try { curPH = ((unsigned long long(*)(id,SEL))objc_msgSend)(mdl, @selector(programHandle)); } @catch(id e) {}
            @try { curIBH = ((unsigned long long(*)(id,SEL))objc_msgSend)(mdl, @selector(intermediateBufferHandle)); } @catch(id e) {}
            @try { curQD = ((unsigned int(*)(id,SEL))objc_msgSend)(mdl, @selector(queueDepth)); } @catch(id e) {}

            printf("  Current: state=%u programHandle=%llu ibHandle=%llu queueDepth=%u\n", curState, curPH, curIBH, curQD);
            printf("  Current attrs: %s\n", curAttrs ? [[curAttrs description] UTF8String] : "nil");

            // Test different queueDepth values
            printf("\n  -- queueDepth tests --\n");
            unsigned int qdValues[] = {0, 1, 2, 4, 8, 16, 32, 64};
            for (int q = 0; q < 8; q++) {
                @try {
                    ((void(*)(id,SEL,id,unsigned int,unsigned long long,unsigned long long,unsigned int))objc_msgSend)(
                        mdl, @selector(updateModelAttributes:state:programHandle:intermediateBufferHandle:queueDepth:),
                        curAttrs ? curAttrs : @{}, curState, curPH, curIBH, qdValues[q]);

                    unsigned int readQD = ((unsigned int(*)(id,SEL))objc_msgSend)(mdl, @selector(queueDepth));
                    printf("    set queueDepth=%u -> read=%u", qdValues[q], readQD);

                    // Bench
                    double ms = bench(mdl, reqBase, 30);
                    printf("  lat=%.3f ms/eval\n", ms);
                } @catch (NSException *ex) {
                    printf("    queueDepth=%u -> EXCEPTION: %s\n", qdValues[q], [[ex reason] UTF8String]);
                }
            }
        } else {
            printf("  Selector NOT available\n");

            // Try setting queueDepth directly
            printf("\n  -- Direct queueDepth setter --\n");
            if ([mdl respondsToSelector:@selector(setQueueDepth:)]) {
                // queueDepth is char type (c) per encoding, so range is -128 to 127
                printf("  setQueueDepth: available (type=char)\n");
                // Note: default was 127
                int qdValues[] = {0, 1, 2, 4, 8, 16, 32, 64, 127, -1};
                const char *qdNames[] = {"0", "1", "2", "4", "8", "16", "32", "64", "127", "-1"};
                for (int q = 0; q < 10; q++) {
                    @try {
                        ((void(*)(id,SEL,char))objc_msgSend)(mdl, @selector(setQueueDepth:), (char)qdValues[q]);
                        char readQD = ((char(*)(id,SEL))objc_msgSend)(mdl, @selector(queueDepth));
                        printf("    set %s -> read %d", qdNames[q], (int)readQD);
                        bench(mdl, reqBase, 10); // warmup
                        double ms = bench(mdl, reqBase, 100);
                        printf("  lat=%.3f ms/eval\n", ms);
                    } @catch (NSException *ex) {
                        printf("    set %s -> EXCEPTION\n", qdNames[q]);
                    }
                }
                // Restore default
                ((void(*)(id,SEL,char))objc_msgSend)(mdl, @selector(setQueueDepth:), (char)127);
            } else {
                printf("  setQueueDepth: NOT available\n");
            }
        }

        // ════════════════════════════════════════════════════════════
        // SECTION E: Scan for undiscovered classes/methods
        // ════════════════════════════════════════════════════════════
        printf("\n\n");
        printf("  ╔═══════════════════════════════════════╗\n");
        printf("  ║  SECTION E: Undiscovered ANE classes   ║\n");
        printf("  ╚═══════════════════════════════════════╝\n");

        scan_ane_classes();

        // Dump any classes we haven't seen before
        const char *extraClasses[] = {
            "_ANEScheduler", "_ANEProgramCache", "_ANECompiler",
            "_ANESession", "_ANESessionHint", "_ANEPowerState",
            "_ANEPipelineModel", "_ANEMultiModel", "_ANEBatchRequest",
            "_ANEAsyncRequest", "_ANERequestQueue", "_ANEModelCache",
            "_ANEBufferPool", "_ANEProgramList", "_ANEDaemon",
            "_ANEProxy", "_ANEModelLoader", "_ANEConfiguration",
            "_ANEDeviceController", "_ANENNGraph",
        };
        for (int i = 0; i < 20; i++) {
            Class c = NSClassFromString([NSString stringWithUTF8String:extraClasses[i]]);
            if (c) dump_class(extraClasses[i]);
        }

        // ════════════════════════════════════════════════════════════
        // SECTION F: _ANEClient full method dump
        // ════════════════════════════════════════════════════════════
        printf("\n\n");
        printf("  ╔═══════════════════════════════════════╗\n");
        printf("  ║  SECTION F: _ANEClient full dump       ║\n");
        printf("  ╚═══════════════════════════════════════╝\n");

        dump_class("_ANEClient");
        dump_class("_ANEVirtualClient");

        // Cleanup
        NSError *cleanErr = nil;
        ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl, @selector(unloadWithQoS:error:), 21, &cleanErr);
        CFRelease(ioIn); CFRelease(ioOut);

        printf("\n  ====================================\n");
        printf("  PROBE COMPLETE\n");
        printf("  ====================================\n\n");
    }
    return 0;
}
