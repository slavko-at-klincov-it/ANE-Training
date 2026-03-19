// test_chaining.m — Deep probe of _ANEChainingRequest for kernel chaining
// Goal: understand if chaining reduces ~0.17ms dispatch overhead between kernels
// Tests: class introspection, object creation, validate, prepareChainingWithModel,
//        benchmarks of individual vs chained eval
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>
#include <math.h>

static mach_timebase_info_data_t g_tb;
static double tb_ms(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }
static double tb_us(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e3; }

// ── Class introspection ──
static void dump_class(const char *name) {
    Class cls = NSClassFromString([NSString stringWithUTF8String:name]);
    if (!cls) { printf("  %s: NOT FOUND\n", name); return; }
    printf("\n=== %s ===\n", name);

    // Superclass chain
    Class super = class_getSuperclass(cls);
    if (super) printf("  Superclass: %s\n", class_getName(super));

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

    // Protocols
    unsigned int protoCount;
    Protocol * __unsafe_unretained *protos = class_copyProtocolList(cls, &protoCount);
    if (protoCount) printf("  Protocols (%u):\n", protoCount);
    for (unsigned int i = 0; i < protoCount; i++) {
        printf("    <%s>\n", protocol_getName(protos[i]));
    }
    free(protos);

    // Ivar layout
    unsigned int ivarCount;
    Ivar *ivars = class_copyIvarList(cls, &ivarCount);
    if (ivarCount) printf("  Ivars (%u):\n", ivarCount);
    for (unsigned int i = 0; i < ivarCount; i++) {
        const char *iname = ivar_getName(ivars[i]);
        const char *itype = ivar_getTypeEncoding(ivars[i]);
        ptrdiff_t offset = ivar_getOffset(ivars[i]);
        printf("    [+%td] %s  (%s)\n", offset, iname ? iname : "?", itype ? itype : "?");
    }
    free(ivars);
}

static IOSurfaceRef make_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0});
}

// ── Compile a kernel ──
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
    BOOL ok;
    ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e);
    if (!ok) { printf("  COMPILE FAILED: %s\n", e ? [[e description] UTF8String] : "unknown"); return nil; }
    ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
    if (!ok) { printf("  LOAD FAILED: %s\n", e ? [[e description] UTF8String] : "unknown"); return nil; }
    return mdl;
}

// ── Try to read all properties of an object ──
static void dump_obj_properties(id obj) {
    if (!obj) { printf("    (nil object)\n"); return; }
    Class cls = object_getClass(obj);
    unsigned int pcount;
    objc_property_t *props = class_copyPropertyList(cls, &pcount);
    for (unsigned int i = 0; i < pcount; i++) {
        const char *pname = property_getName(props[i]);
        @try {
            id val = [obj valueForKey:[NSString stringWithUTF8String:pname]];
            printf("    %s = %s\n", pname, val ? [[val description] UTF8String] : "nil");
        } @catch (NSException *ex) {
            printf("    %s = <exception: %s>\n", pname, [[ex reason] UTF8String]);
        }
    }
    free(props);
}

int main() {
    @autoreleasepool {
        setbuf(stdout, NULL);
        mach_timebase_info(&g_tb);
        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);

        printf("\n");
        printf("  ====== ANE CHAINING REQUEST PROBE ======\n");
        printf("  Goal: understand kernel chaining for reducing dispatch overhead\n\n");

        // ════════════════════════════════════════════════════════════════
        // SECTION 1: Deep introspection of all chaining-related classes
        // ════════════════════════════════════════════════════════════════
        printf("  ── Section 1: Class Introspection ──\n");

        dump_class("_ANEChainingRequest");
        dump_class("_ANERequest");
        dump_class("_ANEMultiRequest");
        dump_class("_ANEBatchRequest");
        dump_class("_ANEClient");
        dump_class("_ANEBuffer");
        dump_class("_ANESharedEvents");
        dump_class("_ANESharedSignalEvent");
        dump_class("_ANESharedWaitEvent");
        dump_class("_ANEFenceEvent");

        // Scan for any other chaining/pipeline/multi classes we might have missed
        printf("\n  Scanning all classes for chaining/pipeline/multi/batch/enqueue keywords...\n");
        unsigned int classCount;
        Class *allClasses = objc_copyClassList(&classCount);
        for (unsigned int i = 0; i < classCount; i++) {
            const char *name = class_getName(allClasses[i]);
            if ((strstr(name, "ANE") || strstr(name, "ane")) &&
                (strstr(name, "hain") || strstr(name, "ipe") || strstr(name, "ulti") ||
                 strstr(name, "atch") || strstr(name, "nqueue") || strstr(name, "oop") ||
                 strstr(name, "equence") || strstr(name, "roup"))) {
                printf("    Found: %s\n", name);
            }
        }
        free(allClasses);

        // ════════════════════════════════════════════════════════════════
        // SECTION 2: Compile TWO kernels with different dimensions
        // ════════════════════════════════════════════════════════════════
        printf("\n  ── Section 2: Compile Two Kernels ──\n\n");

        Class g_D  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class g_I  = NSClassFromString(@"_ANEInMemoryModel");
        Class g_AR = NSClassFromString(@"_ANERequest");
        Class g_AIO= NSClassFromString(@"_ANEIOSurfaceObject");

        // Kernel A: 256ch x 64sp (smaller)
        int CH_A = 256, SP_A = 64;
        NSData *wdataA;
        printf("  Compiling kernel A: %dx%d sp%d...\n", CH_A, CH_A, SP_A);
        id mdlA = compile_kernel(g_D, g_I, CH_A, SP_A, &wdataA);
        printf("  Kernel A: %s\n", mdlA ? "OK" : "FAILED");

        // Kernel B: 512ch x 64sp (larger)
        int CH_B = 512, SP_B = 64;
        NSData *wdataB;
        printf("  Compiling kernel B: %dx%d sp%d...\n", CH_B, CH_B, SP_B);
        id mdlB = compile_kernel(g_D, g_I, CH_B, SP_B, &wdataB);
        printf("  Kernel B: %s\n", mdlB ? "OK" : "FAILED");

        if (!mdlA || !mdlB) {
            printf("  FATAL: Could not compile both kernels\n");
            return 1;
        }

        // Create IOSurfaces
        int ioBytesA = CH_A * SP_A * 4;
        int ioBytesB = CH_B * SP_B * 4;
        IOSurfaceRef ioInA  = make_surface(ioBytesA);
        IOSurfaceRef ioOutA = make_surface(ioBytesA);
        IOSurfaceRef ioInB  = make_surface(ioBytesB);
        IOSurfaceRef ioOutB = make_surface(ioBytesB);

        // Fill inputs
        IOSurfaceLock(ioInA, 0, NULL);
        float *inpA = (float*)IOSurfaceGetBaseAddress(ioInA);
        for (int i = 0; i < CH_A*SP_A; i++) inpA[i] = 1.0f;
        IOSurfaceUnlock(ioInA, 0, NULL);

        IOSurfaceLock(ioInB, 0, NULL);
        float *inpB = (float*)IOSurfaceGetBaseAddress(ioInB);
        for (int i = 0; i < CH_B*SP_B; i++) inpB[i] = 1.0f;
        IOSurfaceUnlock(ioInB, 0, NULL);

        // Verify both kernels work
        id wIA  = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioInA);
        id wOA  = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioOutA);
        id wIB  = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioInB);
        id wOB  = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioOutB);

        id reqA = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wIA], @[@0], @[wOA], @[@0], nil, nil, @0);
        id reqB = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wIB], @[@0], @[wOB], @[@0], nil, nil, @0);

        NSError *e = nil;
        BOOL okA = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            mdlA, @selector(evaluateWithQoS:options:request:error:), 21, @{}, reqA, &e);
        printf("  Kernel A eval: %s\n", okA ? "OK" : "FAIL");

        BOOL okB = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            mdlB, @selector(evaluateWithQoS:options:request:error:), 21, @{}, reqB, &e);
        printf("  Kernel B eval: %s\n", okB ? "OK" : "FAIL");

        // ════════════════════════════════════════════════════════════════
        // SECTION 3: Try to create _ANEChainingRequest — multiple approaches
        // ════════════════════════════════════════════════════════════════
        printf("\n  ── Section 3: ChainingRequest Creation Attempts ──\n\n");

        Class chainClass = NSClassFromString(@"_ANEChainingRequest");
        Class bufClass = NSClassFromString(@"_ANEBuffer");

        if (!chainClass) {
            printf("  FATAL: _ANEChainingRequest class NOT FOUND\n");
        } else {
            // ── 3a: Plain alloc/init ──
            printf("  3a: alloc/init...\n");
            @try {
                id chainObj = [[chainClass alloc] init];
                printf("    Result: %s\n", chainObj ? [[chainObj description] UTF8String] : "nil");
                if (chainObj) {
                    printf("    Properties after init:\n");
                    dump_obj_properties(chainObj);
                }
            } @catch (NSException *ex) {
                printf("    Exception: %s\n", [[ex reason] UTF8String]);
            }

            // ── 3b: Factory method with _ANEBuffer objects ──
            printf("\n  3b: Create _ANEBuffer wrappers via +bufferWithIOSurfaceObject:symbolIndex:source:...\n");

            // Create _ANEBuffer wrappers using the proper factory method
            id bufInA = nil, bufOutA = nil;
            if (bufClass) {
                // Factory: +bufferWithIOSurfaceObject:symbolIndex:source:
                // source is a long long (q type encoding) — try 0 (unknown), 1, 2
                @try {
                    bufInA = ((id(*)(Class,SEL,id,id,long long))objc_msgSend)(
                        bufClass, @selector(bufferWithIOSurfaceObject:symbolIndex:source:),
                        wIA, @0, (long long)0);
                    printf("    bufInA (source=0): %s\n", bufInA ? [[bufInA description] UTF8String] : "nil");
                } @catch (NSException *ex) {
                    printf("    bufInA factory failed: %s\n", [[ex reason] UTF8String]);
                }

                @try {
                    bufOutA = ((id(*)(Class,SEL,id,id,long long))objc_msgSend)(
                        bufClass, @selector(bufferWithIOSurfaceObject:symbolIndex:source:),
                        wOA, @0, (long long)0);
                    printf("    bufOutA (source=0): %s\n", bufOutA ? [[bufOutA description] UTF8String] : "nil");
                } @catch (NSException *ex) {
                    printf("    bufOutA factory failed: %s\n", [[ex reason] UTF8String]);
                }

                if (bufInA) {
                    printf("    bufInA properties:\n");
                    dump_obj_properties(bufInA);
                }
            } else {
                printf("    _ANEBuffer: NOT FOUND\n");
            }

            // ── 3c: Factory method chainingRequestWithInputs:... ──
            printf("\n  3c: chainingRequestWithInputs: factory...\n");

            // Try with _ANEIOSurfaceObject wrapped in arrays
            @try {
                // Based on the known signature:
                // +chainingRequestWithInputs:outputSets:lbInputSymbolId:lbOutputSymbolId:
                //   procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:

                // Approach: arrays of IOSurface objects, scalar params
                id chainReq = ((id(*)(Class,SEL,id,id,id,id,id,id,id,id,id))objc_msgSend)(
                    chainClass,
                    @selector(chainingRequestWithInputs:outputSets:lbInputSymbolId:lbOutputSymbolId:procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:),
                    @[wIA],         // inputs: array of IOSurface objects
                    @[@[wOA]],      // outputSets: array of arrays of IOSurface objects
                    @0,             // lbInputSymbolId
                    @0,             // lbOutputSymbolId
                    @0,             // procedureIndex
                    @[],            // signalEvents (empty)
                    @0,             // transactionHandle
                    @0,             // fwEnqueueDelay
                    @0              // memoryPoolId
                );
                printf("    Result: %s\n", chainReq ? [[chainReq description] UTF8String] : "nil");
                if (chainReq) {
                    printf("    Properties:\n");
                    dump_obj_properties(chainReq);
                }
            } @catch (NSException *ex) {
                printf("    Exception: %s — %s\n", [[ex name] UTF8String], [[ex reason] UTF8String]);
            }

            // ── 3d: Try with different param types based on "count" error ──
            printf("\n  3d: Try with NSArray for all params...\n");
            @try {
                id chainReq = ((id(*)(Class,SEL,id,id,id,id,id,id,id,id,id))objc_msgSend)(
                    chainClass,
                    @selector(chainingRequestWithInputs:outputSets:lbInputSymbolId:lbOutputSymbolId:procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:),
                    @[wIA],         // inputs
                    @[@[wOA]],      // outputSets
                    @[@0],          // lbInputSymbolId as array
                    @[@0],          // lbOutputSymbolId as array
                    @0,             // procedureIndex
                    @[],            // signalEvents
                    @0,             // transactionHandle
                    @0,             // fwEnqueueDelay
                    @0              // memoryPoolId
                );
                printf("    Result: %s\n", chainReq ? [[chainReq description] UTF8String] : "nil");
                if (chainReq) {
                    printf("    Properties:\n");
                    dump_obj_properties(chainReq);
                }
            } @catch (NSException *ex) {
                printf("    Exception: %s — %s\n", [[ex name] UTF8String], [[ex reason] UTF8String]);
            }

            // ── 3e: Try with nil for optional params ──
            printf("\n  3e: Try with nil for optional params...\n");
            @try {
                id chainReq = ((id(*)(Class,SEL,id,id,id,id,id,id,id,id,id))objc_msgSend)(
                    chainClass,
                    @selector(chainingRequestWithInputs:outputSets:lbInputSymbolId:lbOutputSymbolId:procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:),
                    @[wIA],         // inputs
                    @[@[wOA]],      // outputSets
                    @0,             // lbInputSymbolId
                    @0,             // lbOutputSymbolId
                    @0,             // procedureIndex
                    nil,            // signalEvents
                    nil,            // transactionHandle
                    @0,             // fwEnqueueDelay
                    @0              // memoryPoolId
                );
                printf("    Result: %s\n", chainReq ? [[chainReq description] UTF8String] : "nil");
                if (chainReq) {
                    printf("    Properties:\n");
                    dump_obj_properties(chainReq);
                }
            } @catch (NSException *ex) {
                printf("    Exception: %s — %s\n", [[ex name] UTF8String], [[ex reason] UTF8String]);
            }

            // ── 3f: Try with proper _ANEBuffer objects from factory ──
            printf("\n  3f: ChainingRequest with _ANEBuffer wrappers from factory...\n");
            if (bufClass && bufInA && bufOutA) {
                @try {
                    id chainReq = ((id(*)(Class,SEL,id,id,id,id,id,id,id,id,id))objc_msgSend)(
                        chainClass,
                        @selector(chainingRequestWithInputs:outputSets:lbInputSymbolId:lbOutputSymbolId:procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:),
                        @[bufInA],          // inputs as _ANEBuffer
                        @[@[bufOutA]],      // outputSets as _ANEBuffer
                        @0,                 // lbInputSymbolId
                        @0,                 // lbOutputSymbolId
                        @0,                 // procedureIndex
                        @[],                // signalEvents
                        @0,                 // transactionHandle
                        @0,                 // fwEnqueueDelay
                        @0                  // memoryPoolId
                    );
                    printf("    Result: %s\n", chainReq ? [[chainReq description] UTF8String] : "nil");
                    if (chainReq) {
                        printf("    Properties:\n");
                        dump_obj_properties(chainReq);

                        // Now try validate
                        printf("    validate: ");
                        @try {
                            BOOL valid = ((BOOL(*)(id,SEL))objc_msgSend)(chainReq, @selector(validate));
                            printf("%s\n", valid ? "YES" : "NO");
                        } @catch (NSException *ex) {
                            printf("Exception: %s\n", [[ex reason] UTF8String]);
                        }
                    }
                } @catch (NSException *ex) {
                    printf("    Exception: %s — %s\n", [[ex name] UTF8String], [[ex reason] UTF8String]);
                }

                // ── 3f2: Try with two outputSets for double-buffering ──
                printf("\n  3f2: ChainingRequest with _ANEBuffer + 2 outputSets...\n");
                @try {
                    id bufOutA2 = ((id(*)(Class,SEL,id,id,long long))objc_msgSend)(
                        bufClass, @selector(bufferWithIOSurfaceObject:symbolIndex:source:),
                        wOA, @0, (long long)0);

                    id chainReq = ((id(*)(Class,SEL,id,id,id,id,id,id,id,id,id))objc_msgSend)(
                        chainClass,
                        @selector(chainingRequestWithInputs:outputSets:lbInputSymbolId:lbOutputSymbolId:procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:),
                        @[bufInA],                      // inputs
                        @[@[bufOutA], @[bufOutA2]],     // TWO output sets
                        @0,                             // lbInputSymbolId
                        @0,                             // lbOutputSymbolId
                        @0,                             // procedureIndex
                        @[],                            // signalEvents
                        @0,                             // transactionHandle
                        @0,                             // fwEnqueueDelay
                        @0                              // memoryPoolId
                    );
                    printf("    Result: %s\n", chainReq ? [[chainReq description] UTF8String] : "nil");
                    if (chainReq) {
                        printf("    validate: ");
                        BOOL valid = ((BOOL(*)(id,SEL))objc_msgSend)(chainReq, @selector(validate));
                        printf("%s\n", valid ? "YES" : "NO");
                    }
                } @catch (NSException *ex) {
                    printf("    Exception: %s — %s\n", [[ex name] UTF8String], [[ex reason] UTF8String]);
                }
            }

            // ── 3g: Try with multiple outputSets (chaining implies multiple outputs) ──
            printf("\n  3g: Multiple outputSets (2 sets)...\n");
            @try {
                id chainReq = ((id(*)(Class,SEL,id,id,id,id,id,id,id,id,id))objc_msgSend)(
                    chainClass,
                    @selector(chainingRequestWithInputs:outputSets:lbInputSymbolId:lbOutputSymbolId:procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:),
                    @[wIA],                 // inputs
                    @[@[wOA], @[wOA]],      // TWO outputSets
                    @0,                     // lbInputSymbolId
                    @0,                     // lbOutputSymbolId
                    @0,                     // procedureIndex
                    @[],                    // signalEvents
                    @0,                     // transactionHandle
                    @0,                     // fwEnqueueDelay
                    @0                      // memoryPoolId
                );
                printf("    Result: %s\n", chainReq ? [[chainReq description] UTF8String] : "nil");
                if (chainReq) {
                    printf("    Properties:\n");
                    dump_obj_properties(chainReq);
                }
            } @catch (NSException *ex) {
                printf("    Exception: %s — %s\n", [[ex name] UTF8String], [[ex reason] UTF8String]);
            }

            // ── 3h: Loopback — output feeds back as input (lbInputSymbolId/lbOutputSymbolId) ──
            printf("\n  3h: Loopback test — lbInputSymbolId=0, lbOutputSymbolId=0...\n");
            @try {
                // The "lb" prefix likely means "loopback" — output index that feeds back to input index
                // This could be how chaining works: one eval's output becomes next eval's input
                id chainReq = ((id(*)(Class,SEL,id,id,id,id,id,id,id,id,id))objc_msgSend)(
                    chainClass,
                    @selector(chainingRequestWithInputs:outputSets:lbInputSymbolId:lbOutputSymbolId:procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:),
                    @[wIA],             // inputs
                    @[@[wOA], @[wOA]],  // two output sets for ping-pong
                    @0,                 // lbInputSymbolId = first input
                    @0,                 // lbOutputSymbolId = first output
                    @0,                 // procedureIndex
                    @[],                // signalEvents
                    @0,                 // transactionHandle
                    @0,                 // fwEnqueueDelay
                    @0                  // memoryPoolId
                );
                printf("    Result: %s\n", chainReq ? [[chainReq description] UTF8String] : "nil");
                if (chainReq) {
                    printf("    Properties:\n");
                    dump_obj_properties(chainReq);
                }
            } @catch (NSException *ex) {
                printf("    Exception: %s — %s\n", [[ex name] UTF8String], [[ex reason] UTF8String]);
            }
        }

        // ════════════════════════════════════════════════════════════════
        // SECTION 4: Try prepareChainingWithModel on _ANEClient
        // ════════════════════════════════════════════════════════════════
        printf("\n  ── Section 4: _ANEClient chaining methods ──\n\n");

        Class clientClass = NSClassFromString(@"_ANEClient");
        id client = nil;
        if (clientClass) {
            client = ((id(*)(Class,SEL))objc_msgSend)(clientClass, @selector(sharedConnection));
            printf("  _ANEClient sharedConnection: %s\n", client ? "OK" : "nil");

            if (client) {
                // List all chaining-related methods on the client
                printf("  Client methods containing 'hain' or 'nqueue' or 'uffer':\n");
                unsigned int mcount;
                Method *cmethods = class_copyMethodList([client class], &mcount);
                for (unsigned int i = 0; i < mcount; i++) {
                    const char *sname = sel_getName(method_getName(cmethods[i]));
                    if (strstr(sname, "hain") || strstr(sname, "nqueue") ||
                        strstr(sname, "uffer") || strstr(sname, "repare") ||
                        strstr(sname, "atch") || strstr(sname, "ulti")) {
                        const char *enc = method_getTypeEncoding(cmethods[i]);
                        printf("    - %s  [%s]\n", sname, enc ? enc : "?");
                    }
                }
                free(cmethods);

                // Try prepareChainingWithModel if we have a ChainingRequest
                if (chainClass && bufClass) {
                    printf("\n  4a: prepareChainingWithModel: with _ANEBuffer-based ChainingRequest...\n");

                    // Create _ANEBuffer wrappers for inputs/outputs
                    id bInA = ((id(*)(Class,SEL,id,id,long long))objc_msgSend)(
                        bufClass, @selector(bufferWithIOSurfaceObject:symbolIndex:source:),
                        wIA, @0, (long long)0);
                    id bOutA = ((id(*)(Class,SEL,id,id,long long))objc_msgSend)(
                        bufClass, @selector(bufferWithIOSurfaceObject:symbolIndex:source:),
                        wOA, @0, (long long)0);

                    printf("    bInA: %s\n", bInA ? [[bInA description] UTF8String] : "nil");
                    printf("    bOutA: %s\n", bOutA ? [[bOutA description] UTF8String] : "nil");

                    // Try with 1 outputSet
                    @try {
                        id chainReq = ((id(*)(Class,SEL,id,id,id,id,id,id,id,id,id))objc_msgSend)(
                            chainClass,
                            @selector(chainingRequestWithInputs:outputSets:lbInputSymbolId:lbOutputSymbolId:procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:),
                            @[bInA], @[@[bOutA]], @0, @0, @0, @[], @0, @0, @0
                        );
                        printf("    ChainingRequest (1 outputSet): %s\n", chainReq ? "created" : "nil");

                        if (chainReq) {
                            e = nil;
                            BOOL ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                                client,
                                @selector(prepareChainingWithModel:options:chainingReq:qos:error:),
                                mdlA, @{}, chainReq, 21, &e
                            );
                            printf("    prepareChainingWithModel (1 set): %s\n", ok ? "OK" : "FAIL");
                            if (e) printf("    Error: %s\n", [[e description] UTF8String]);
                        }
                    } @catch (NSException *ex) {
                        printf("    Exception (1 set): %s — %s\n", [[ex name] UTF8String], [[ex reason] UTF8String]);
                    }

                    // Try with 2 outputSets (double-buffer for chaining)
                    @try {
                        id bOutA2 = ((id(*)(Class,SEL,id,id,long long))objc_msgSend)(
                            bufClass, @selector(bufferWithIOSurfaceObject:symbolIndex:source:),
                            wOA, @0, (long long)0);

                        id chainReq = ((id(*)(Class,SEL,id,id,id,id,id,id,id,id,id))objc_msgSend)(
                            chainClass,
                            @selector(chainingRequestWithInputs:outputSets:lbInputSymbolId:lbOutputSymbolId:procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:),
                            @[bInA], @[@[bOutA], @[bOutA2]], @0, @0, @0, @[], @0, @0, @0
                        );
                        printf("    ChainingRequest (2 outputSets): %s\n", chainReq ? "created" : "nil");

                        if (chainReq) {
                            e = nil;
                            BOOL ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                                client,
                                @selector(prepareChainingWithModel:options:chainingReq:qos:error:),
                                mdlA, @{}, chainReq, 21, &e
                            );
                            printf("    prepareChainingWithModel (2 sets): %s\n", ok ? "OK" : "FAIL");
                            if (e) printf("    Error: %s\n", [[e description] UTF8String]);

                            if (ok) {
                                // If chaining prepared, try enqueueSets
                                printf("    Chaining prepared! Trying enqueueSetsWithModel...\n");
                                @try {
                                    e = nil;
                                    BOOL enqOk = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                                        client,
                                        @selector(enqueueSetsWithModel:outputSet:options:qos:error:),
                                        mdlA, @[bOutA], @{}, 21, &e
                                    );
                                    printf("    enqueueSetsWithModel: %s\n", enqOk ? "OK" : "FAIL");
                                    if (e) printf("    Error: %s\n", [[e description] UTF8String]);
                                } @catch (NSException *ex) {
                                    printf("    enqueue Exception: %s\n", [[ex reason] UTF8String]);
                                }
                            }
                        }
                    } @catch (NSException *ex) {
                        printf("    Exception (2 sets): %s — %s\n", [[ex name] UTF8String], [[ex reason] UTF8String]);
                    }

                    // Try with QoS=9
                    @try {
                        id chainReq = ((id(*)(Class,SEL,id,id,id,id,id,id,id,id,id))objc_msgSend)(
                            chainClass,
                            @selector(chainingRequestWithInputs:outputSets:lbInputSymbolId:lbOutputSymbolId:procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:),
                            @[bInA], @[@[bOutA]], @0, @0, @0, @[], @0, @0, @0
                        );
                        if (chainReq) {
                            e = nil;
                            BOOL ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                                client,
                                @selector(prepareChainingWithModel:options:chainingReq:qos:error:),
                                mdlA, @{}, chainReq, 9, &e
                            );
                            printf("    prepareChainingWithModel (QoS=9): %s\n", ok ? "OK" : "FAIL");
                            if (e) printf("    Error: %s\n", [[e description] UTF8String]);
                        }
                    } @catch (NSException *ex) {
                        printf("    Exception (QoS=9): %s\n", [[ex reason] UTF8String]);
                    }

                    // ── 4b: Try with different selector signatures ──
                    printf("\n  4b: Try alternative prepareChainingWithModel: signatures...\n");

                    // The method might take different params — check what selectors exist
                    SEL selectors[] = {
                        @selector(prepareChainingWithModel:options:chainingReq:qos:error:),
                        @selector(prepareChainingWithModel:options:chainingRequest:qos:error:),
                        @selector(prepareChainingWithModel:chainingReq:qos:error:),
                        @selector(prepareChainingWithModel:chainingRequest:error:),
                        @selector(prepareChainingForModel:options:chainingReq:qos:error:),
                        NSSelectorFromString(@"prepareChainingWithModel:options:chainingReq:qos:error:"),
                    };
                    for (int s = 0; s < 6; s++) {
                        BOOL responds = [client respondsToSelector:selectors[s]];
                        printf("    %s: %s\n", sel_getName(selectors[s]), responds ? "YES" : "no");
                    }
                }

                // ── 4c: Try enqueueSetsWithModel:outputSet:options:qos:error: ──
                printf("\n  4c: enqueueSetsWithModel:outputSet:options:qos:error:...\n");
                if (bufClass) {
                    // First, dump _ANEOutputSetEnqueue class
                    dump_class("_ANEOutputSetEnqueue");

                    // Try enqueueSetsWithModel with _ANEBuffer array
                    id bOutEnq = ((id(*)(Class,SEL,id,id,long long))objc_msgSend)(
                        bufClass, @selector(bufferWithIOSurfaceObject:symbolIndex:source:),
                        wOA, @0, (long long)0);

                    @try {
                        e = nil;
                        BOOL ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                            client,
                            @selector(enqueueSetsWithModel:outputSet:options:qos:error:),
                            mdlA, @[bOutEnq], @{}, 21, &e
                        );
                        printf("    enqueueSets (buffer array): %s\n", ok ? "OK" : "FAIL");
                        if (e) printf("    Error: %s\n", [[e description] UTF8String]);
                    } @catch (NSException *ex) {
                        printf("    Exception: %s — %s\n", [[ex name] UTF8String], [[ex reason] UTF8String]);
                    }

                    // Try with IOSurface objects directly
                    @try {
                        e = nil;
                        BOOL ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                            client,
                            @selector(enqueueSetsWithModel:outputSet:options:qos:error:),
                            mdlA, @[wOA], @{}, 21, &e
                        );
                        printf("    enqueueSets (iosurface array): %s\n", ok ? "OK" : "FAIL");
                        if (e) printf("    Error: %s\n", [[e description] UTF8String]);
                    } @catch (NSException *ex) {
                        printf("    Exception: %s — %s\n", [[ex name] UTF8String], [[ex reason] UTF8String]);
                    }
                }

                // ── 4d: buffersReadyWithModel + mapIOSurfaces ──
                printf("\n  4d: buffersReadyWithModel:inputBuffers:options:qos:error:...\n");
                if (bufClass) {
                    id bInBuf = ((id(*)(Class,SEL,id,id,long long))objc_msgSend)(
                        bufClass, @selector(bufferWithIOSurfaceObject:symbolIndex:source:),
                        wIA, @0, (long long)0);

                    @try {
                        e = nil;
                        BOOL ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                            client,
                            @selector(buffersReadyWithModel:inputBuffers:options:qos:error:),
                            mdlA, @[bInBuf], @{}, 21, &e
                        );
                        printf("    buffersReady: %s\n", ok ? "OK" : "FAIL");
                        if (e) printf("    Error: %s\n", [[e description] UTF8String]);
                    } @catch (NSException *ex) {
                        printf("    Exception: %s — %s\n", [[ex name] UTF8String], [[ex reason] UTF8String]);
                    }
                }

                // mapIOSurfacesWithModel:request:cacheInference:error:
                printf("\n  4e: mapIOSurfacesWithModel:request:cacheInference:error:...\n");
                @try {
                    e = nil;
                    BOOL ok = ((BOOL(*)(id,SEL,id,id,BOOL,NSError**))objc_msgSend)(
                        client,
                        @selector(mapIOSurfacesWithModel:request:cacheInference:error:),
                        mdlA, reqA, YES, &e
                    );
                    printf("    mapIOSurfaces (cache=YES): %s\n", ok ? "OK" : "FAIL");
                    if (e) printf("    Error: %s\n", [[e description] UTF8String]);

                    if (ok) {
                        // Try unmapping
                        ((void(*)(id,SEL,id,id))objc_msgSend)(
                            client, @selector(unmapIOSurfacesWithModel:request:), mdlA, reqA);
                        printf("    unmapIOSurfaces: done\n");
                    }
                } @catch (NSException *ex) {
                    printf("    Exception: %s — %s\n", [[ex name] UTF8String], [[ex reason] UTF8String]);
                }

                // ── 4f: Full chaining pipeline attempt ──
                printf("\n  4f: Full chaining pipeline: prepareChaining + enqueueSets + buffersReady...\n");
                if (bufClass) {
                    @try {
                        // Create fresh buffers
                        id bIn = ((id(*)(Class,SEL,id,id,long long))objc_msgSend)(
                            bufClass, @selector(bufferWithIOSurfaceObject:symbolIndex:source:),
                            wIA, @0, (long long)0);
                        id bOut1 = ((id(*)(Class,SEL,id,id,long long))objc_msgSend)(
                            bufClass, @selector(bufferWithIOSurfaceObject:symbolIndex:source:),
                            wOA, @0, (long long)0);
                        id bOut2 = ((id(*)(Class,SEL,id,id,long long))objc_msgSend)(
                            bufClass, @selector(bufferWithIOSurfaceObject:symbolIndex:source:),
                            wOA, @0, (long long)0);

                        // Create chaining request with 2 output sets
                        id chainReq = ((id(*)(Class,SEL,id,id,id,id,id,id,id,id,id))objc_msgSend)(
                            chainClass,
                            @selector(chainingRequestWithInputs:outputSets:lbInputSymbolId:lbOutputSymbolId:procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:),
                            @[bIn], @[@[bOut1], @[bOut2]], @0, @0, @0, @[], @0, @0, @0
                        );
                        printf("    ChainingRequest: %s\n", chainReq ? "created" : "nil");

                        // Step 1: prepareChaining
                        e = nil;
                        BOOL ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                            client,
                            @selector(prepareChainingWithModel:options:chainingReq:qos:error:),
                            mdlA, @{}, chainReq, 9, &e
                        );
                        printf("    Step 1 prepareChaining: %s%s\n", ok ? "OK" : "FAIL",
                               e ? [[NSString stringWithFormat:@" — %@", [e localizedDescription]] UTF8String] : "");

                        // Step 2: buffersReady
                        e = nil;
                        ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                            client,
                            @selector(buffersReadyWithModel:inputBuffers:options:qos:error:),
                            mdlA, @[bIn], @{}, 9, &e
                        );
                        printf("    Step 2 buffersReady: %s%s\n", ok ? "OK" : "FAIL",
                               e ? [[NSString stringWithFormat:@" — %@", [e localizedDescription]] UTF8String] : "");

                        // Step 3: enqueueSets
                        e = nil;
                        ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                            client,
                            @selector(enqueueSetsWithModel:outputSet:options:qos:error:),
                            mdlA, @[bOut1], @{}, 9, &e
                        );
                        printf("    Step 3 enqueueSets: %s%s\n", ok ? "OK" : "FAIL",
                               e ? [[NSString stringWithFormat:@" — %@", [e localizedDescription]] UTF8String] : "");

                    } @catch (NSException *ex) {
                        printf("    Pipeline exception: %s — %s\n", [[ex name] UTF8String], [[ex reason] UTF8String]);
                    }
                }
            }
        }

        // ════════════════════════════════════════════════════════════════
        // SECTION 5: Try _ANEInMemoryModel chaining methods
        // ════════════════════════════════════════════════════════════════
        printf("\n  ── Section 5: Model-level chaining methods ──\n\n");

        // Check if the model itself has chaining methods
        {
            printf("  Checking mdlA for chaining-related selectors...\n");
            SEL mdlSelectors[] = {
                NSSelectorFromString(@"prepareChainingWithQoS:options:chainingReq:error:"),
                NSSelectorFromString(@"evaluateChainedWithQoS:options:chainingReq:error:"),
                NSSelectorFromString(@"evaluateWithQoS:options:requests:error:"),
                NSSelectorFromString(@"evaluateBatchWithQoS:options:requests:error:"),
                NSSelectorFromString(@"evaluateMultiWithQoS:options:requests:error:"),
                NSSelectorFromString(@"enqueueSetsWithQoS:options:sets:error:"),
                NSSelectorFromString(@"setChaining:"),
                NSSelectorFromString(@"chainingRequest"),
                NSSelectorFromString(@"setChainingRequest:"),
                NSSelectorFromString(@"programOutputSets"),
                NSSelectorFromString(@"numberOfProgramOutputSets"),
            };
            for (int s = 0; s < 11; s++) {
                BOOL responds = [mdlA respondsToSelector:mdlSelectors[s]];
                printf("    %s: %s\n", sel_getName(mdlSelectors[s]), responds ? "YES" : "no");
            }

            // Search all methods on mdlA's class that might be relevant
            printf("\n  All model methods with 'hain', 'nqueue', 'atch', 'ets', 'ultiple':\n");
            unsigned int mcount;
            Method *mmethods = class_copyMethodList([mdlA class], &mcount);
            for (unsigned int i = 0; i < mcount; i++) {
                const char *sname = sel_getName(method_getName(mmethods[i]));
                if (strstr(sname, "hain") || strstr(sname, "nqueue") ||
                    strstr(sname, "atch") || strstr(sname, "ets") ||
                    strstr(sname, "ultiple") || strstr(sname, "ulti") ||
                    strstr(sname, "roup")) {
                    const char *enc = method_getTypeEncoding(mmethods[i]);
                    printf("    - %s  [%s]\n", sname, enc ? enc : "?");
                }
            }
            free(mmethods);
        }

        // ════════════════════════════════════════════════════════════════
        // SECTION 6: Benchmark baseline — sequential evals of two kernels
        // ════════════════════════════════════════════════════════════════
        printf("\n  ── Section 6: Benchmark — Sequential Eval of Two Kernels ──\n\n");

        int WARMUP = 50;
        int N = 500;

        // Warmup
        for (int i = 0; i < WARMUP; i++) {
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                mdlA, @selector(evaluateWithQoS:options:request:error:), 21, @{}, reqA, &e);
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                mdlB, @selector(evaluateWithQoS:options:request:error:), 21, @{}, reqB, &e);
        }

        // Benchmark: A alone
        uint64_t t0 = mach_absolute_time();
        for (int i = 0; i < N; i++) {
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                mdlA, @selector(evaluateWithQoS:options:request:error:), 21, @{}, reqA, &e);
        }
        double msA = tb_ms(mach_absolute_time() - t0);
        printf("  Kernel A alone:  %d evals in %.1f ms (%.3f ms/eval, %.0f us/eval)\n",
               N, msA, msA/N, msA/N*1000);

        // Benchmark: B alone
        t0 = mach_absolute_time();
        for (int i = 0; i < N; i++) {
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                mdlB, @selector(evaluateWithQoS:options:request:error:), 21, @{}, reqB, &e);
        }
        double msB = tb_ms(mach_absolute_time() - t0);
        printf("  Kernel B alone:  %d evals in %.1f ms (%.3f ms/eval, %.0f us/eval)\n",
               N, msB, msB/N, msB/N*1000);

        // Benchmark: A then B sequential
        t0 = mach_absolute_time();
        for (int i = 0; i < N; i++) {
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                mdlA, @selector(evaluateWithQoS:options:request:error:), 21, @{}, reqA, &e);
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                mdlB, @selector(evaluateWithQoS:options:request:error:), 21, @{}, reqB, &e);
        }
        double msAB = tb_ms(mach_absolute_time() - t0);
        printf("  A+B sequential:  %d pairs in %.1f ms (%.3f ms/pair, %.0f us/pair)\n",
               N, msAB, msAB/N, msAB/N*1000);

        double overhead = (msAB/N) - (msA/N + msB/N);
        printf("\n  Dispatch overhead per pair: %.3f ms (%.0f us)\n", overhead, overhead*1000);
        printf("  Expected if no overhead: %.3f ms/pair\n", (msA+msB)/N);
        printf("  Actual: %.3f ms/pair\n", msAB/N);

        // Benchmark: QoS=9 (background — fastest on M3 Pro)
        printf("\n  Same benchmark with QoS=9 (background)...\n");

        // Warmup QoS=9
        for (int i = 0; i < WARMUP; i++) {
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                mdlA, @selector(evaluateWithQoS:options:request:error:), 9, @{}, reqA, &e);
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                mdlB, @selector(evaluateWithQoS:options:request:error:), 9, @{}, reqB, &e);
        }

        t0 = mach_absolute_time();
        for (int i = 0; i < N; i++) {
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                mdlA, @selector(evaluateWithQoS:options:request:error:), 9, @{}, reqA, &e);
        }
        double msA9 = tb_ms(mach_absolute_time() - t0);
        printf("  Kernel A (QoS=9):  %.3f ms/eval\n", msA9/N);

        t0 = mach_absolute_time();
        for (int i = 0; i < N; i++) {
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                mdlB, @selector(evaluateWithQoS:options:request:error:), 9, @{}, reqB, &e);
        }
        double msB9 = tb_ms(mach_absolute_time() - t0);
        printf("  Kernel B (QoS=9):  %.3f ms/eval\n", msB9/N);

        t0 = mach_absolute_time();
        for (int i = 0; i < N; i++) {
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                mdlA, @selector(evaluateWithQoS:options:request:error:), 9, @{}, reqA, &e);
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                mdlB, @selector(evaluateWithQoS:options:request:error:), 9, @{}, reqB, &e);
        }
        double msAB9 = tb_ms(mach_absolute_time() - t0);
        printf("  A+B sequential (QoS=9): %.3f ms/pair\n", msAB9/N);
        double overhead9 = (msAB9/N) - (msA9/N + msB9/N);
        printf("  Dispatch overhead (QoS=9): %.3f ms (%.0f us)\n", overhead9, overhead9*1000);

        // ════════════════════════════════════════════════════════════════
        // SECTION 7: Try real-time task mode for chaining benefit
        // ════════════════════════════════════════════════════════════════
        printf("\n  ── Section 7: Real-Time Task + Sequential Eval ──\n\n");
        if (client) {
            BOOL rtOk = ((BOOL(*)(id,SEL))objc_msgSend)(client, @selector(beginRealTimeTask));
            printf("  beginRealTimeTask: %s\n", rtOk ? "OK" : "FAIL");
            if (rtOk) {
                // Warmup in RT mode
                for (int i = 0; i < WARMUP; i++) {
                    ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                        mdlA, @selector(evaluateWithQoS:options:request:error:), 9, @{}, reqA, &e);
                    ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                        mdlB, @selector(evaluateWithQoS:options:request:error:), 9, @{}, reqB, &e);
                }

                t0 = mach_absolute_time();
                for (int i = 0; i < N; i++) {
                    ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                        mdlA, @selector(evaluateWithQoS:options:request:error:), 9, @{}, reqA, &e);
                }
                double msArt = tb_ms(mach_absolute_time() - t0);

                t0 = mach_absolute_time();
                for (int i = 0; i < N; i++) {
                    ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                        mdlB, @selector(evaluateWithQoS:options:request:error:), 9, @{}, reqB, &e);
                }
                double msBrt = tb_ms(mach_absolute_time() - t0);

                t0 = mach_absolute_time();
                for (int i = 0; i < N; i++) {
                    ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                        mdlA, @selector(evaluateWithQoS:options:request:error:), 9, @{}, reqA, &e);
                    ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                        mdlB, @selector(evaluateWithQoS:options:request:error:), 9, @{}, reqB, &e);
                }
                double msABrt = tb_ms(mach_absolute_time() - t0);

                printf("  RT mode Kernel A:  %.3f ms/eval\n", msArt/N);
                printf("  RT mode Kernel B:  %.3f ms/eval\n", msBrt/N);
                printf("  RT mode A+B:       %.3f ms/pair\n", msABrt/N);
                double overheadRT = (msABrt/N) - (msArt/N + msBrt/N);
                printf("  RT dispatch overhead: %.3f ms (%.0f us)\n", overheadRT, overheadRT*1000);

                ((BOOL(*)(id,SEL))objc_msgSend)(client, @selector(endRealTimeTask));
            }
        }

        // ════════════════════════════════════════════════════════════════
        // SECTION 8: evaluateRealTimeWithModel — benchmark vs regular eval
        // ════════════════════════════════════════════════════════════════
        printf("\n  ── Section 8: evaluateRealTimeWithModel ──\n\n");
        if (client) {
            // Check which selectors exist
            SEL rtSelectors[] = {
                NSSelectorFromString(@"evaluateRealTimeWithModel:options:request:qos:error:"),
                NSSelectorFromString(@"evaluateRealTimeWithModel:options:request:error:"),
                NSSelectorFromString(@"evaluateRealTimeWithModel:request:qos:error:"),
                NSSelectorFromString(@"evaluateRealTimeWithModel:request:error:"),
            };
            for (int s = 0; s < 4; s++) {
                BOOL responds = [client respondsToSelector:rtSelectors[s]];
                printf("    %s: %s\n", sel_getName(rtSelectors[s]), responds ? "YES" : "no");
            }

            // evaluateRealTimeWithModel:options:request:error: exists — benchmark it!
            // Signature: B48@0:8@16@24@32^@40  (BOOL, self, sel, model, options, request, error**)
            SEL rtSel = NSSelectorFromString(@"evaluateRealTimeWithModel:options:request:error:");
            if ([client respondsToSelector:rtSel]) {
                printf("\n    Benchmarking evaluateRealTimeWithModel vs evaluateWithModel...\n");

                // First, load models as real-time
                @try {
                    e = nil;
                    BOOL ok = ((BOOL(*)(id,SEL,id,id,unsigned int,NSError**))objc_msgSend)(
                        client, @selector(loadRealTimeModel:options:qos:error:), mdlA, @{}, 9, &e);
                    printf("    loadRealTimeModel A: %s\n", ok ? "OK" : "FAIL");
                    if (e) printf("    Error: %s\n", [[e description] UTF8String]);
                    e = nil;
                    ok = ((BOOL(*)(id,SEL,id,id,unsigned int,NSError**))objc_msgSend)(
                        client, @selector(loadRealTimeModel:options:qos:error:), mdlB, @{}, 9, &e);
                    printf("    loadRealTimeModel B: %s\n", ok ? "OK" : "FAIL");
                    if (e) printf("    Error: %s\n", [[e description] UTF8String]);
                } @catch (NSException *ex) {
                    printf("    loadRealTime exception: %s\n", [[ex reason] UTF8String]);
                }

                // Warmup RT evals
                for (int i = 0; i < WARMUP; i++) {
                    @try {
                        ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
                            client, rtSel, mdlA, @{}, reqA, &e);
                        ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
                            client, rtSel, mdlB, @{}, reqB, &e);
                    } @catch (NSException *ex) { break; }
                }

                // Benchmark RT eval A alone
                @try {
                    t0 = mach_absolute_time();
                    for (int i = 0; i < N; i++) {
                        ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
                            client, rtSel, mdlA, @{}, reqA, &e);
                    }
                    double msARt = tb_ms(mach_absolute_time() - t0);
                    printf("    RT eval A alone: %.3f ms/eval\n", msARt/N);

                    // RT eval B alone
                    t0 = mach_absolute_time();
                    for (int i = 0; i < N; i++) {
                        ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
                            client, rtSel, mdlB, @{}, reqB, &e);
                    }
                    double msBRt = tb_ms(mach_absolute_time() - t0);
                    printf("    RT eval B alone: %.3f ms/eval\n", msBRt/N);

                    // RT eval A+B sequential
                    t0 = mach_absolute_time();
                    for (int i = 0; i < N; i++) {
                        ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
                            client, rtSel, mdlA, @{}, reqA, &e);
                        ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
                            client, rtSel, mdlB, @{}, reqB, &e);
                    }
                    double msABRt = tb_ms(mach_absolute_time() - t0);
                    printf("    RT eval A+B: %.3f ms/pair\n", msABRt/N);
                    double overheadRt = (msABRt/N) - (msARt/N + msBRt/N);
                    printf("    RT dispatch overhead: %.3f ms (%.0f us)\n", overheadRt, overheadRt*1000);

                    // Compare with doEvaluateDirectWithModel
                    printf("\n    Testing doEvaluateDirectWithModel...\n");
                    SEL directSel = @selector(doEvaluateDirectWithModel:options:request:qos:error:);
                    if ([client respondsToSelector:directSel]) {
                        // Warmup
                        for (int i = 0; i < WARMUP; i++) {
                            ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                                client, directSel, mdlA, @{}, reqA, 9, &e);
                        }
                        t0 = mach_absolute_time();
                        for (int i = 0; i < N; i++) {
                            ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                                client, directSel, mdlA, @{}, reqA, 9, &e);
                        }
                        double msDirect = tb_ms(mach_absolute_time() - t0);
                        printf("    doEvaluateDirect A: %.3f ms/eval\n", msDirect/N);

                        t0 = mach_absolute_time();
                        for (int i = 0; i < N; i++) {
                            ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                                client, directSel, mdlA, @{}, reqA, 9, &e);
                            ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                                client, directSel, mdlB, @{}, reqB, 9, &e);
                        }
                        double msDirectAB = tb_ms(mach_absolute_time() - t0);
                        printf("    doEvaluateDirect A+B: %.3f ms/pair\n", msDirectAB/N);
                    } else {
                        printf("    doEvaluateDirectWithModel: not available\n");
                    }
                } @catch (NSException *ex) {
                    printf("    RT benchmark exception: %s\n", [[ex reason] UTF8String]);
                }

                // Unload RT models
                @try {
                    ((BOOL(*)(id,SEL,id,id,unsigned int,NSError**))objc_msgSend)(
                        client, @selector(unloadRealTimeModel:options:qos:error:), mdlA, @{}, 9, &e);
                    ((BOOL(*)(id,SEL,id,id,unsigned int,NSError**))objc_msgSend)(
                        client, @selector(unloadRealTimeModel:options:qos:error:), mdlB, @{}, 9, &e);
                } @catch (NSException *ex) {}
            }
        }

        // ════════════════════════════════════════════════════════════════
        // SECTION 9: Same-model sequential — measure if reusing same
        // model has lower overhead than switching models
        // ════════════════════════════════════════════════════════════════
        printf("\n  ── Section 9: Model-Switch Overhead ──\n\n");

        // A->A->A->A sequential (same model, no switch)
        t0 = mach_absolute_time();
        for (int i = 0; i < N; i++) {
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                mdlA, @selector(evaluateWithQoS:options:request:error:), 9, @{}, reqA, &e);
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                mdlA, @selector(evaluateWithQoS:options:request:error:), 9, @{}, reqA, &e);
        }
        double msAA = tb_ms(mach_absolute_time() - t0);

        // A->B->A->B sequential (model switch every eval)
        t0 = mach_absolute_time();
        for (int i = 0; i < N; i++) {
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                mdlA, @selector(evaluateWithQoS:options:request:error:), 9, @{}, reqA, &e);
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                mdlB, @selector(evaluateWithQoS:options:request:error:), 9, @{}, reqB, &e);
        }
        double msABswitch = tb_ms(mach_absolute_time() - t0);

        printf("  A->A (same model):  %.3f ms/pair (%.0f us/pair)\n", msAA/N, msAA/N*1000);
        printf("  A->B (switch model): %.3f ms/pair (%.0f us/pair)\n", msABswitch/N, msABswitch/N*1000);
        printf("  Model-switch penalty: %.3f ms (%.0f us)\n",
               (msABswitch/N - msAA/N), (msABswitch/N - msAA/N)*1000);

        // ════════════════════════════════════════════════════════════════
        // SECTION 10: Detailed latency histogram for single eval
        // ════════════════════════════════════════════════════════════════
        printf("\n  ── Section 10: Eval Latency Distribution (1000 samples) ──\n\n");

        int SAMPLES = 1000;
        double *latencies = (double*)malloc(SAMPLES * sizeof(double));

        for (int i = 0; i < SAMPLES; i++) {
            uint64_t ts = mach_absolute_time();
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                mdlA, @selector(evaluateWithQoS:options:request:error:), 9, @{}, reqA, &e);
            latencies[i] = tb_us(mach_absolute_time() - ts);
        }

        // Sort for percentiles
        for (int i = 0; i < SAMPLES-1; i++)
            for (int j = i+1; j < SAMPLES; j++)
                if (latencies[j] < latencies[i]) { double t = latencies[i]; latencies[i] = latencies[j]; latencies[j] = t; }

        double sum = 0;
        for (int i = 0; i < SAMPLES; i++) sum += latencies[i];

        printf("  Kernel A (256ch x 64sp), QoS=9:\n");
        printf("    Mean:  %.0f us\n", sum/SAMPLES);
        printf("    Min:   %.0f us\n", latencies[0]);
        printf("    P50:   %.0f us\n", latencies[SAMPLES/2]);
        printf("    P90:   %.0f us\n", latencies[(int)(SAMPLES*0.9)]);
        printf("    P99:   %.0f us\n", latencies[(int)(SAMPLES*0.99)]);
        printf("    Max:   %.0f us\n", latencies[SAMPLES-1]);

        // Same for kernel B
        for (int i = 0; i < SAMPLES; i++) {
            uint64_t ts = mach_absolute_time();
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                mdlB, @selector(evaluateWithQoS:options:request:error:), 9, @{}, reqB, &e);
            latencies[i] = tb_us(mach_absolute_time() - ts);
        }
        for (int i = 0; i < SAMPLES-1; i++)
            for (int j = i+1; j < SAMPLES; j++)
                if (latencies[j] < latencies[i]) { double t = latencies[i]; latencies[i] = latencies[j]; latencies[j] = t; }
        sum = 0;
        for (int i = 0; i < SAMPLES; i++) sum += latencies[i];

        printf("\n  Kernel B (512ch x 64sp), QoS=9:\n");
        printf("    Mean:  %.0f us\n", sum/SAMPLES);
        printf("    Min:   %.0f us\n", latencies[0]);
        printf("    P50:   %.0f us\n", latencies[SAMPLES/2]);
        printf("    P90:   %.0f us\n", latencies[(int)(SAMPLES*0.9)]);
        printf("    P99:   %.0f us\n", latencies[(int)(SAMPLES*0.99)]);
        printf("    Max:   %.0f us\n", latencies[SAMPLES-1]);

        // A->B pair latency distribution
        for (int i = 0; i < SAMPLES; i++) {
            uint64_t ts = mach_absolute_time();
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                mdlA, @selector(evaluateWithQoS:options:request:error:), 9, @{}, reqA, &e);
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                mdlB, @selector(evaluateWithQoS:options:request:error:), 9, @{}, reqB, &e);
            latencies[i] = tb_us(mach_absolute_time() - ts);
        }
        for (int i = 0; i < SAMPLES-1; i++)
            for (int j = i+1; j < SAMPLES; j++)
                if (latencies[j] < latencies[i]) { double t = latencies[i]; latencies[i] = latencies[j]; latencies[j] = t; }
        sum = 0;
        for (int i = 0; i < SAMPLES; i++) sum += latencies[i];

        printf("\n  A+B pair, QoS=9:\n");
        printf("    Mean:  %.0f us\n", sum/SAMPLES);
        printf("    Min:   %.0f us\n", latencies[0]);
        printf("    P50:   %.0f us\n", latencies[SAMPLES/2]);
        printf("    P90:   %.0f us\n", latencies[(int)(SAMPLES*0.9)]);
        printf("    P99:   %.0f us\n", latencies[(int)(SAMPLES*0.99)]);
        printf("    Max:   %.0f us\n", latencies[SAMPLES-1]);

        free(latencies);

        // ════════════════════════════════════════════════════════════════
        // SUMMARY
        // ════════════════════════════════════════════════════════════════
        printf("\n  ══════════════════════════════════════════════\n");
        printf("  SUMMARY\n");
        printf("  ══════════════════════════════════════════════\n\n");
        printf("  Kernel A (256x256 sp64): %.3f ms/eval\n", msA9/N);
        printf("  Kernel B (512x512 sp64): %.3f ms/eval\n", msB9/N);
        printf("  A+B sequential:          %.3f ms/pair\n", msAB9/N);
        printf("  Expected (A+B no overhead): %.3f ms\n", msA9/N + msB9/N);
        printf("  Dispatch overhead:       %.3f ms (%.0f us)\n", overhead9, overhead9*1000);
        printf("  Model-switch penalty:    %.3f ms (%.0f us)\n",
               (msABswitch/N - msAA/N), (msABswitch/N - msAA/N)*1000);
        printf("\n  _ANEChainingRequest: %s\n", chainClass ? "EXISTS" : "NOT FOUND");
        printf("  prepareChainingWithModel: %s\n",
               client && [client respondsToSelector:@selector(prepareChainingWithModel:options:chainingReq:qos:error:)]
               ? "METHOD EXISTS" : "NOT FOUND");
        printf("\n");

        // Cleanup
        ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdlA, @selector(unloadWithQoS:error:), 21, &e);
        ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdlB, @selector(unloadWithQoS:error:), 21, &e);
        CFRelease(ioInA); CFRelease(ioOutA);
        CFRelease(ioInB); CFRelease(ioOutB);
    }
    return 0;
}
