// test_buffer.m — Probe _ANEBuffer, SRAM/memory management, intermediateBufferHandle
// Goal: Can we control SRAM allocation or buffer management to prevent DRAM spills?
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>

static mach_timebase_info_data_t g_tb;
static double tb_ms(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }
static double tb_us(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e3; }

// ══════════════════════════════════════════════
// Dump all methods, properties, ivars of a class
// ══════════════════════════════════════════════
static void dump_class_full(const char *name) {
    Class cls = NSClassFromString([NSString stringWithUTF8String:name]);
    if (!cls) { printf("  %s: NOT FOUND\n\n", name); return; }
    printf("\n=== %s ===\n", name);

    // Superclass chain
    Class super = class_getSuperclass(cls);
    printf("  Superclass: %s\n", super ? class_getName(super) : "(none)");

    // Class methods
    unsigned int count;
    Method *methods = class_copyMethodList(object_getClass(cls), &count);
    if (count) printf("  Class methods (%u):\n", count);
    for (unsigned int i = 0; i < count; i++) {
        SEL s = method_getName(methods[i]);
        const char *enc = method_getTypeEncoding(methods[i]);
        printf("    + %s  [%s]\n", sel_getName(s), enc ? enc : "?");
    }
    free(methods);

    // Instance methods
    methods = class_copyMethodList(cls, &count);
    if (count) printf("  Instance methods (%u):\n", count);
    for (unsigned int i = 0; i < count; i++) {
        SEL s = method_getName(methods[i]);
        const char *enc = method_getTypeEncoding(methods[i]);
        printf("    - %s  [%s]\n", sel_getName(s), enc ? enc : "?");
    }
    free(methods);

    // Properties
    unsigned int pcount;
    objc_property_t *props = class_copyPropertyList(cls, &pcount);
    if (pcount) printf("  Properties (%u):\n", pcount);
    for (unsigned int i = 0; i < pcount; i++) {
        const char *pname = property_getName(props[i]);
        const char *pattr = property_getAttributes(props[i]);
        printf("    @property %s  [%s]\n", pname, pattr ? pattr : "?");
    }
    free(props);

    // Instance variables
    unsigned int icount;
    Ivar *ivars = class_copyIvarList(cls, &icount);
    if (icount) printf("  Ivars (%u):\n", icount);
    for (unsigned int i = 0; i < icount; i++) {
        const char *iname = ivar_getName(ivars[i]);
        const char *itype = ivar_getTypeEncoding(ivars[i]);
        printf("    %s  [%s]\n", iname ? iname : "?", itype ? itype : "?");
    }
    free(ivars);

    // Protocols
    unsigned int prcount;
    Protocol * __unsafe_unretained *protos = class_copyProtocolList(cls, &prcount);
    if (prcount) {
        printf("  Protocols (%u):", prcount);
        for (unsigned int i = 0; i < prcount; i++)
            printf(" %s", protocol_getName(protos[i]));
        printf("\n");
    }
    free(protos);

    printf("\n");
}

// ══════════════════════════════════════════════
// Scan all loaded classes for keywords
// ══════════════════════════════════════════════
static void scan_classes_for_keywords(void) {
    printf("\n══════════════════════════════════════════════\n");
    printf("  Scanning ALL classes for memory/SRAM-related names\n");
    printf("══════════════════════════════════════════════\n\n");

    const char *keywords[] = {
        "Buffer", "SRAM", "Memory", "Cache", "Intermediate",
        "Tile", "Scratch", "Pool", "Alloc", "DMA", "Spill",
        "Bandwidth", "Storage", NULL
    };

    unsigned int classCount = 0;
    Class *classes = objc_copyClassList(&classCount);
    printf("  Total loaded classes: %u\n\n", classCount);

    int found = 0;
    for (unsigned int i = 0; i < classCount; i++) {
        const char *cname = class_getName(classes[i]);
        if (!cname) continue;
        // Only ANE-related classes
        if (strstr(cname, "ANE") == NULL && strstr(cname, "ane") == NULL &&
            strstr(cname, "Espresso") == NULL && strstr(cname, "espresso") == NULL &&
            strstr(cname, "Neural") == NULL)
            continue;

        for (int k = 0; keywords[k]; k++) {
            if (strcasestr(cname, keywords[k])) {
                printf("  MATCH: %s (keyword: %s)\n", cname, keywords[k]);
                found++;
                break;
            }
        }
    }
    if (!found) {
        // Broader: just list all ANE classes
        printf("  No keyword matches. Listing ALL ANE-related classes:\n");
        for (unsigned int i = 0; i < classCount; i++) {
            const char *cname = class_getName(classes[i]);
            if (!cname) continue;
            if (strstr(cname, "ANE") || strstr(cname, "_ANE"))
                printf("    %s\n", cname);
        }
    }
    free(classes);
}

// ══════════════════════════════════════════════
// Also scan for buffer/memory-related methods on ANE classes
// ══════════════════════════════════════════════
static void scan_methods_for_keywords(void) {
    printf("\n══════════════════════════════════════════════\n");
    printf("  Scanning ANE class methods for buffer/memory keywords\n");
    printf("══════════════════════════════════════════════\n\n");

    const char *method_keywords[] = {
        "buffer", "sram", "memory", "cache", "intermediate",
        "tile", "scratch", "pool", "alloc", "dma", "spill",
        "bandwidth", "storage", "reserved", "mapped", NULL
    };

    unsigned int classCount = 0;
    Class *classes = objc_copyClassList(&classCount);

    for (unsigned int i = 0; i < classCount; i++) {
        const char *cname = class_getName(classes[i]);
        if (!cname) continue;
        if (strstr(cname, "ANE") == NULL && strstr(cname, "_ANE") == NULL)
            continue;

        unsigned int mcount;
        Method *methods = class_copyMethodList(classes[i], &mcount);
        for (unsigned int m = 0; m < mcount; m++) {
            const char *mname = sel_getName(method_getName(methods[m]));
            for (int k = 0; method_keywords[k]; k++) {
                if (strcasestr(mname, method_keywords[k])) {
                    printf("  %s -> -%s\n", cname, mname);
                    break;
                }
            }
        }
        free(methods);

        // Class methods too
        methods = class_copyMethodList(object_getClass(classes[i]), &mcount);
        for (unsigned int m = 0; m < mcount; m++) {
            const char *mname = sel_getName(method_getName(methods[m]));
            for (int k = 0; method_keywords[k]; k++) {
                if (strcasestr(mname, method_keywords[k])) {
                    printf("  %s -> +%s\n", cname, mname);
                    break;
                }
            }
        }
        free(methods);
    }
    free(classes);
}

static IOSurfaceRef make_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0});
}

// ══════════════════════════════════════════════
// Compile a kernel (reused from test_ane_monitor)
// ══════════════════════════════════════════════
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
    BOOL compOk = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
        mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e);
    if (!compOk) printf("  COMPILE FAILED: %s\n", [[e description] UTF8String]);
    BOOL loadOk = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
        mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
    if (!loadOk) printf("  LOAD FAILED: %s\n", [[e description] UTF8String]);
    return mdl;
}

// ══════════════════════════════════════════════
// Try to read a property safely via KVC
// ══════════════════════════════════════════════
static void read_property(id obj, const char *propName) {
    @try {
        id val = [obj valueForKey:[NSString stringWithUTF8String:propName]];
        if ([val isKindOfClass:[NSData class]]) {
            NSData *d = (NSData *)val;
            printf("    %s = <NSData %lu bytes>", propName, (unsigned long)[d length]);
            if ([d length] > 0 && [d length] <= 64) {
                const uint8_t *b = [d bytes];
                printf(" [");
                for (NSUInteger i = 0; i < [d length]; i++) printf("%02x", b[i]);
                printf("]");
            }
            printf("\n");
        } else {
            printf("    %s = %s\n", propName, val ? [[val description] UTF8String] : "nil");
        }
    } @catch (NSException *ex) {
        printf("    %s = <exception: %s>\n", propName, [[ex reason] UTF8String]);
    }
}

static void read_all_properties(id obj, Class cls) {
    unsigned int pcount;
    objc_property_t *props = class_copyPropertyList(cls, &pcount);
    for (unsigned int i = 0; i < pcount; i++) {
        read_property(obj, property_getName(props[i]));
    }
    free(props);
}

int main() {
    @autoreleasepool {
        setbuf(stdout, NULL);
        mach_timebase_info(&g_tb);
        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);

        printf("\n");
        printf("  ██████ ANE BUFFER / SRAM PROBE ██████\n\n");

        // ════════════════════════════════════════
        // Section 1: Full class dumps
        // ════════════════════════════════════════
        printf("══════════════════════════════════════════════\n");
        printf("  Section 1: Class Introspection\n");
        printf("══════════════════════════════════════════════\n");

        dump_class_full("_ANEBuffer");
        dump_class_full("_ANEIOSurfaceObject");
        dump_class_full("_ANEModel");
        dump_class_full("_ANEInMemoryModel");
        dump_class_full("_ANERequest");

        // Classes discovered during method scan
        dump_class_full("_ANEProgramForEvaluation");
        dump_class_full("_ANEProgramIOSurfacesMapper");
        dump_class_full("_ANEInputBuffersReady");
        dump_class_full("_ANEChainingRequest");
        dump_class_full("_ANEIOSurfaceOutputSets");
        dump_class_full("_ANESharedEvents");
        dump_class_full("_ANECompilerAnalytics");

        // Additional classes that might exist
        dump_class_full("_ANEMemoryPool");
        dump_class_full("_ANEMemoryManager");
        dump_class_full("_ANESRAMManager");
        dump_class_full("_ANEBufferPool");
        dump_class_full("_ANETileInfo");
        dump_class_full("_ANECache");
        dump_class_full("_ANEDMAInfo");
        dump_class_full("_ANEIntermediateBuffer");
        dump_class_full("_ANESharedMemory");
        dump_class_full("_ANEDeviceMemory");
        dump_class_full("_ANEAllocation");
        dump_class_full("_ANEWeightsBuffer");
        dump_class_full("_ANEProgramCache");
        dump_class_full("_ANEModelProgramCache");
        dump_class_full("_ANEBufferDescriptor");
        dump_class_full("_ANEResourcePool");
        dump_class_full("_ANEProgramHandle");
        dump_class_full("_ANEModelHandle");

        // ════════════════════════════════════════
        // Section 2: Keyword scan across all classes
        // ════════════════════════════════════════
        scan_classes_for_keywords();
        scan_methods_for_keywords();

        // ════════════════════════════════════════
        // Section 3: Try to instantiate _ANEBuffer
        // ════════════════════════════════════════
        printf("\n══════════════════════════════════════════════\n");
        printf("  Section 3: _ANEBuffer Instantiation Tests\n");
        printf("══════════════════════════════════════════════\n\n");

        Class bufClass = NSClassFromString(@"_ANEBuffer");
        if (bufClass) {
            // Try alloc/init
            @try {
                id buf = [[bufClass alloc] init];
                printf("  alloc/init: %s\n", buf ? [[buf description] UTF8String] : "nil");
                if (buf) read_all_properties(buf, bufClass);
            } @catch (NSException *ex) {
                printf("  alloc/init exception: %s\n", [[ex reason] UTF8String]);
            }

            // Try class factory methods — check for objectWith*, bufferWith*, etc.
            printf("\n  Trying factory methods:\n");
            unsigned int cmcount;
            Method *cmethods = class_copyMethodList(object_getClass(bufClass), &cmcount);
            for (unsigned int i = 0; i < cmcount; i++) {
                SEL s = method_getName(cmethods[i]);
                const char *sname = sel_getName(s);
                printf("    Checking +%s\n", sname);
            }
            free(cmethods);

            // Try creating with an IOSurface
            IOSurfaceRef testSurf = make_surface(4096);
            if (testSurf) {
                // Try various init patterns
                @try {
                    id buf = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                        bufClass, @selector(objectWithIOSurface:), testSurf);
                    printf("\n  +objectWithIOSurface: %s\n", buf ? [[buf description] UTF8String] : "nil");
                    if (buf) read_all_properties(buf, bufClass);
                } @catch (NSException *ex) {
                    printf("  +objectWithIOSurface: exception: %s\n", [[ex reason] UTF8String]);
                }

                @try {
                    id buf = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                        bufClass, @selector(bufferWithIOSurface:), testSurf);
                    printf("  +bufferWithIOSurface: %s\n", buf ? [[buf description] UTF8String] : "nil");
                    if (buf) read_all_properties(buf, bufClass);
                } @catch (NSException *ex) {
                    printf("  +bufferWithIOSurface: exception: %s\n", [[ex reason] UTF8String]);
                }

                @try {
                    id buf = [[bufClass alloc] performSelector:@selector(initWithIOSurface:) withObject:(__bridge id)testSurf];
                    printf("  -initWithIOSurface: %s\n", buf ? [[buf description] UTF8String] : "nil");
                    if (buf) read_all_properties(buf, bufClass);
                } @catch (NSException *ex) {
                    printf("  -initWithIOSurface: exception: %s\n", [[ex reason] UTF8String]);
                }

                // Try with size
                @try {
                    id buf = ((id(*)(Class,SEL,unsigned long))objc_msgSend)(
                        bufClass, @selector(bufferWithSize:), (unsigned long)4096);
                    printf("  +bufferWithSize:4096 -> %s\n", buf ? [[buf description] UTF8String] : "nil");
                    if (buf) read_all_properties(buf, bufClass);
                } @catch (NSException *ex) {
                    printf("  +bufferWithSize: exception: %s\n", [[ex reason] UTF8String]);
                }

                CFRelease(testSurf);
            }
        } else {
            printf("  _ANEBuffer: NOT FOUND\n");
        }

        // ════════════════════════════════════════
        // Section 4: intermediateBufferHandle on _ANEModel and _ANEInMemoryModel
        // ════════════════════════════════════════
        printf("\n══════════════════════════════════════════════\n");
        printf("  Section 4: intermediateBufferHandle Investigation\n");
        printf("══════════════════════════════════════════════\n\n");

        Class g_D  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class g_I  = NSClassFromString(@"_ANEInMemoryModel");
        Class g_AR = NSClassFromString(@"_ANERequest");
        Class g_AIO= NSClassFromString(@"_ANEIOSurfaceObject");

        // Compile a working kernel
        int CH = 256, SP = 64;
        NSData *wdata;
        id mdl = compile_kernel(g_D, g_I, CH, SP, &wdata);
        printf("  Model compiled: %s\n", mdl ? "YES" : "NO");

        if (mdl) {
            // Read intermediateBufferHandle
            printf("\n  Reading intermediateBufferHandle from _ANEInMemoryModel:\n");
            @try {
                id ibh = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(intermediateBufferHandle));
                printf("    intermediateBufferHandle = %s\n", ibh ? [[ibh description] UTF8String] : "nil");
                if (ibh) {
                    printf("    Class: %s\n", class_getName([ibh class]));
                    // Dump whatever this object is
                    dump_class_full(class_getName([ibh class]));

                    // Read its properties
                    unsigned int pc;
                    objc_property_t *pp = class_copyPropertyList([ibh class], &pc);
                    printf("    Properties of intermediateBufferHandle:\n");
                    for (unsigned int i = 0; i < pc; i++) {
                        read_property(ibh, property_getName(pp[i]));
                    }
                    free(pp);
                }
            } @catch (NSException *ex) {
                printf("    intermediateBufferHandle exception: %s\n", [[ex reason] UTF8String]);
            }

            // Check the _ANEProgramForEvaluation object which manages the buffer handle
            printf("\n  Examining _ANEProgramForEvaluation from model:\n");
            @try {
                id prog = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(program));
                printf("    program = %s\n", prog ? [[prog description] UTF8String] : "nil");
                if (prog) {
                    printf("    program class: %s\n", class_getName([prog class]));
                    read_all_properties(prog, [prog class]);

                    // Read intermediateBufferHandle as uint64 from program
                    uint64_t progIBH = ((uint64_t(*)(id,SEL))objc_msgSend)(prog, @selector(intermediateBufferHandle));
                    printf("    program.intermediateBufferHandle (uint64) = %llu (0x%llx)\n", progIBH, progIBH);

                    // Read queueDepth
                    @try {
                        char qd = ((char(*)(id,SEL))objc_msgSend)(prog, @selector(queueDepth));
                        printf("    program.queueDepth = %d\n", (int)qd);
                    } @catch (NSException *ex) {}
                }
            } @catch (NSException *ex) {
                printf("    program exception: %s\n", [[ex reason] UTF8String]);
            }

            // Read intermediateBufferHandle as raw uint64 from model
            uint64_t rawIBH = ((uint64_t(*)(id,SEL))objc_msgSend)(mdl, @selector(intermediateBufferHandle));
            printf("\n  intermediateBufferHandle as uint64: %llu (0x%llx)\n", rawIBH, rawIBH);

            // Read programHandle as raw uint64
            uint64_t rawPH = ((uint64_t(*)(id,SEL))objc_msgSend)(mdl, @selector(programHandle));
            printf("  programHandle as uint64: %llu (0x%llx)\n", rawPH, rawPH);

            // Check if there's an intermediateBufferSize or similar
            printf("\n  Checking related properties on model:\n");
            const char *model_props[] = {
                "intermediateBufferHandle", "intermediateBufferSize",
                "intermediateBuffer", "scratchBufferHandle",
                "scratchBuffer", "scratchBufferSize",
                "sramSize", "sramUsage", "memoryUsage",
                "estimatedMemory", "peakMemoryUsage",
                "tileInfo", "programHandle", "compiledModelHandle",
                "modelHandle", "bufferCount", "intermediateCount",
                NULL
            };
            for (int i = 0; model_props[i]; i++) {
                @try {
                    id val = [mdl valueForKey:[NSString stringWithUTF8String:model_props[i]]];
                    if (val) printf("    %s = %s\n", model_props[i], [[val description] UTF8String]);
                } @catch (NSException *ex) {
                    // Only print if not undefined key
                    NSString *reason = [ex reason];
                    if (![reason containsString:@"valueForUndefinedKey"])
                        printf("    %s = <exception: %s>\n", model_props[i], [reason UTF8String]);
                }
            }

            // Try setting intermediateBufferHandle
            printf("\n  Testing intermediateBufferHandle setter:\n");

            // Can we set it to nil?
            @try {
                ((void(*)(id,SEL,id))objc_msgSend)(mdl, @selector(setIntermediateBufferHandle:), nil);
                printf("    setIntermediateBufferHandle:nil -> OK\n");
                id ibh2 = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(intermediateBufferHandle));
                printf("    After set nil: intermediateBufferHandle = %s\n", ibh2 ? [[ibh2 description] UTF8String] : "nil");
            } @catch (NSException *ex) {
                printf("    setIntermediateBufferHandle:nil -> exception: %s\n", [[ex reason] UTF8String]);
            }

            // Try creating an IOSurface-backed buffer and setting it
            IOSurfaceRef ibSurf = make_surface(1024*1024); // 1MB
            if (ibSurf) {
                id ioObj = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                    g_AIO, @selector(objectWithIOSurface:), ibSurf);
                printf("    Created IOSurfaceObject (1MB) for intermediateBuffer: %s\n",
                       ioObj ? "OK" : "FAIL");

                @try {
                    ((void(*)(id,SEL,id))objc_msgSend)(mdl, @selector(setIntermediateBufferHandle:), ioObj);
                    printf("    setIntermediateBufferHandle:ioObj -> OK\n");
                    id ibh3 = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(intermediateBufferHandle));
                    printf("    After set: intermediateBufferHandle = %s\n",
                           ibh3 ? [[ibh3 description] UTF8String] : "nil");
                } @catch (NSException *ex) {
                    printf("    setIntermediateBufferHandle:ioObj -> exception: %s\n",
                           [[ex reason] UTF8String]);
                }

                // If we have an _ANEBuffer class, try that
                if (bufClass) {
                    @try {
                        id abuf = [[bufClass alloc] init];
                        if (abuf) {
                            ((void(*)(id,SEL,id))objc_msgSend)(mdl, @selector(setIntermediateBufferHandle:), abuf);
                            printf("    setIntermediateBufferHandle:_ANEBuffer -> OK\n");
                        }
                    } @catch (NSException *ex) {
                        printf("    setIntermediateBufferHandle:_ANEBuffer -> exception: %s\n",
                               [[ex reason] UTF8String]);
                    }
                }

                CFRelease(ibSurf);
            }
        }

        // ════════════════════════════════════════
        // Section 5: weightsBuffer in _ANERequest
        // ════════════════════════════════════════
        printf("\n══════════════════════════════════════════════\n");
        printf("  Section 5: weightsBuffer in _ANERequest\n");
        printf("══════════════════════════════════════════════\n\n");

        if (mdl) {
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

            // Create request with weightsBuffer
            IOSurfaceRef wbSurf = make_surface(1024*1024);
            id wbObj = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), wbSurf);

            // Normal request (no weightsBuffer)
            id reqNormal = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
                @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                @[wI], @[@0], @[wO], @[@0], nil, nil, @0);

            printf("  Normal request: %s\n", reqNormal ? "OK" : "FAIL");
            if (reqNormal) {
                printf("    Request properties:\n");
                read_all_properties(reqNormal, [reqNormal class]);
            }

            // Request WITH weightsBuffer
            id reqWithWB = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
                @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                @[wI], @[@0], @[wO], @[@0], wbObj, nil, @0);

            printf("\n  Request with weightsBuffer: %s\n", reqWithWB ? "OK" : "FAIL");
            if (reqWithWB) {
                printf("    Request properties:\n");
                read_all_properties(reqWithWB, [reqWithWB class]);
            }

            // ════════════════════════════════════════
            // Section 6: Performance comparison — with/without intermediateBufferHandle
            // ════════════════════════════════════════
            printf("\n══════════════════════════════════════════════\n");
            printf("  Section 6: Performance Impact of intermediateBufferHandle\n");
            printf("══════════════════════════════════════════════\n\n");

            // First, reload the model fresh
            NSError *e = nil;
            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl, @selector(unloadWithQoS:error:), 21, &e);

            // Recompile
            NSData *wdata2;
            id mdl2 = compile_kernel(g_D, g_I, CH, SP, &wdata2);

            // Benchmark: default intermediateBufferHandle
            int ioBytes2 = CH * SP * 4;
            IOSurfaceRef ioIn2 = make_surface(ioBytes2);
            IOSurfaceRef ioOut2 = make_surface(ioBytes2);
            id wI2 = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioIn2);
            id wO2 = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioOut2);

            IOSurfaceLock(ioIn2, 0, NULL);
            float *inp2 = (float*)IOSurfaceGetBaseAddress(ioIn2);
            for (int i = 0; i < CH*SP; i++) inp2[i] = 1.0f;
            IOSurfaceUnlock(ioIn2, 0, NULL);

            id reqBench = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
                @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                @[wI2], @[@0], @[wO2], @[@0], nil, nil, @0);

            // Read default intermediateBufferHandle
            id defaultIBH = ((id(*)(id,SEL))objc_msgSend)(mdl2, @selector(intermediateBufferHandle));
            printf("  Default intermediateBufferHandle: %s\n",
                   defaultIBH ? [[defaultIBH description] UTF8String] : "nil");

            int N = 500;

            // Warmup
            for (int i = 0; i < 50; i++) {
                ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                    mdl2, @selector(evaluateWithQoS:options:request:error:), 21, @{}, reqBench, &e);
            }

            // Benchmark default
            uint64_t t0 = mach_absolute_time();
            for (int i = 0; i < N; i++) {
                ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                    mdl2, @selector(evaluateWithQoS:options:request:error:), 21, @{}, reqBench, &e);
            }
            double default_ms = tb_ms(mach_absolute_time() - t0);
            printf("  Default:    %d evals in %.1f ms (%.3f ms/eval)\n", N, default_ms, default_ms/N);

            // Now try setting intermediateBufferHandle to nil
            @try {
                ((void(*)(id,SEL,id))objc_msgSend)(mdl2, @selector(setIntermediateBufferHandle:), nil);
                printf("  Set intermediateBufferHandle to nil\n");

                // Warmup
                for (int i = 0; i < 50; i++) {
                    ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                        mdl2, @selector(evaluateWithQoS:options:request:error:), 21, @{}, reqBench, &e);
                }

                t0 = mach_absolute_time();
                for (int i = 0; i < N; i++) {
                    ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                        mdl2, @selector(evaluateWithQoS:options:request:error:), 21, @{}, reqBench, &e);
                }
                double nil_ms = tb_ms(mach_absolute_time() - t0);
                printf("  Nil IBH:    %d evals in %.1f ms (%.3f ms/eval)\n", N, nil_ms, nil_ms/N);
            } @catch (NSException *ex) {
                printf("  Nil IBH test exception: %s\n", [[ex reason] UTF8String]);
            }

            // Try with a large IOSurface as intermediateBufferHandle
            IOSurfaceRef largeSurf = make_surface(16*1024*1024); // 16MB
            if (largeSurf) {
                id largeIO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                    g_AIO, @selector(objectWithIOSurface:), largeSurf);

                @try {
                    ((void(*)(id,SEL,id))objc_msgSend)(mdl2, @selector(setIntermediateBufferHandle:), largeIO);
                    printf("  Set intermediateBufferHandle to 16MB IOSurface\n");

                    // Warmup
                    for (int i = 0; i < 50; i++) {
                        ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                            mdl2, @selector(evaluateWithQoS:options:request:error:), 21, @{}, reqBench, &e);
                    }

                    t0 = mach_absolute_time();
                    for (int i = 0; i < N; i++) {
                        ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                            mdl2, @selector(evaluateWithQoS:options:request:error:), 21, @{}, reqBench, &e);
                    }
                    double large_ms = tb_ms(mach_absolute_time() - t0);
                    printf("  16MB IBH:   %d evals in %.1f ms (%.3f ms/eval)\n", N, large_ms, large_ms/N);
                } @catch (NSException *ex) {
                    printf("  16MB IBH test exception: %s\n", [[ex reason] UTF8String]);
                }

                CFRelease(largeSurf);
            }

            // Restore default and try a HUGE surface
            @try {
                ((void(*)(id,SEL,id))objc_msgSend)(mdl2, @selector(setIntermediateBufferHandle:), defaultIBH);
            } @catch (NSException *ex) {}

            // ════════════════════════════════════════
            // Section 7: _ANEIOSurfaceObject deep dive
            // ════════════════════════════════════════
            printf("\n══════════════════════════════════════════════\n");
            printf("  Section 7: _ANEIOSurfaceObject Deep Dive\n");
            printf("══════════════════════════════════════════════\n\n");

            // Create various sizes and check properties
            size_t sizes[] = {1024, 65536, 1024*1024, 16*1024*1024};
            const char *labels[] = {"1KB", "64KB", "1MB", "16MB"};

            for (int s = 0; s < 4; s++) {
                IOSurfaceRef surf = make_surface(sizes[s]);
                id ioObj2 = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                    g_AIO, @selector(objectWithIOSurface:), surf);

                printf("  %s IOSurfaceObject:\n", labels[s]);
                if (ioObj2) {
                    unsigned int pc;
                    objc_property_t *pp = class_copyPropertyList(g_AIO, &pc);
                    for (unsigned int i = 0; i < pc; i++) {
                        read_property(ioObj2, property_getName(pp[i]));
                    }
                    free(pp);

                    // Try memory/SRAM related methods
                    const char *try_methods[] = {
                        "size", "allocSize", "bytesPerRow", "baseAddress",
                        "isInUse", "lockCount", "seed", "surfaceID",
                        "isMapped", "isGlobal", "localUseCount",
                        NULL
                    };
                    for (int m = 0; try_methods[m]; m++) {
                        @try {
                            id val = [ioObj2 valueForKey:[NSString stringWithUTF8String:try_methods[m]]];
                            if (val) printf("    %s = %s\n", try_methods[m], [[val description] UTF8String]);
                        } @catch (NSException *ex) {}
                    }
                }
                printf("\n");
                CFRelease(surf);
            }

            // ════════════════════════════════════════
            // Section 8: Explore _ANEClient for memory management
            // ════════════════════════════════════════
            printf("══════════════════════════════════════════════\n");
            printf("  Section 8: _ANEClient Memory Management\n");
            printf("══════════════════════════════════════════════\n\n");

            dump_class_full("_ANEClient");
            dump_class_full("_ANEVirtualClient");

            Class clientClass = NSClassFromString(@"_ANEClient");
            if (clientClass) {
                id client = ((id(*)(Class,SEL))objc_msgSend)(clientClass, @selector(sharedConnection));
                if (client) {
                    printf("  Client properties:\n");
                    read_all_properties(client, clientClass);

                    // Try memory management methods
                    const char *mm_methods[] = {
                        "memoryPoolSize", "maxBufferSize", "sramSize",
                        "intermediateBufferPoolSize", "totalMemoryUsage",
                        "activeModelCount", "maxModelsLoaded",
                        NULL
                    };
                    printf("\n  Checking memory management properties on client:\n");
                    for (int i = 0; mm_methods[i]; i++) {
                        @try {
                            id val = [client valueForKey:[NSString stringWithUTF8String:mm_methods[i]]];
                            if (val) printf("    %s = %s\n", mm_methods[i], [[val description] UTF8String]);
                        } @catch (NSException *ex) {
                            NSString *reason = [ex reason];
                            if (![reason containsString:@"valueForUndefinedKey"])
                                printf("    %s = <exception: %s>\n", mm_methods[i], [reason UTF8String]);
                        }
                    }
                }
            }

            // Cleanup
            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl2, @selector(unloadWithQoS:error:), 21, &e);
            CFRelease(ioIn); CFRelease(ioOut);
            CFRelease(ioIn2); CFRelease(ioOut2);
            CFRelease(wbSurf);
        }

        printf("\n  ██████ PROBE COMPLETE ██████\n\n");
    }
    return 0;
}
