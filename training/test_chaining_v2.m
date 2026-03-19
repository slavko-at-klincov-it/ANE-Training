// test_chaining_v2.m — Fully activate _ANEChainingRequest
// prepareChainingWithModel: SUCCESS (achieved!)
// Now: investigate enqueueSetsWithModel for chained execution
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

static IOSurfaceRef make_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0});
}

static void dump_obj_properties(id obj) {
    if (!obj) return;
    Class cls = object_getClass(obj);
    while (cls && cls != [NSObject class]) {
        unsigned int pcount;
        objc_property_t *props = class_copyPropertyList(cls, &pcount);
        for (unsigned int i = 0; i < pcount; i++) {
            const char *pname = property_getName(props[i]);
            @try {
                id val = [obj valueForKey:[NSString stringWithUTF8String:pname]];
                printf("    %s = %s\n", pname, val ? [[val description] UTF8String] : "nil");
            } @catch (NSException *ex) {
                printf("    %s = <exc>\n", pname);
            }
        }
        free(props);
        cls = class_getSuperclass(cls);
    }
}

// ── Dynamic patches ──
static const char kOutputBufferKey = 0;
static id outputBuffer_getter(id self, SEL _cmd) {
    return objc_getAssociatedObject(self, &kOutputBufferKey);
}
static BOOL supportsSecureCoding_YES(id self, SEL _cmd) { return YES; }
static void outputSetEnqueue_encode(id self, SEL _cmd, id coder) {
    unsigned int procIdx = ((unsigned int(*)(id,SEL))objc_msgSend)(self, @selector(procedureIndex));
    unsigned int setIdx  = ((unsigned int(*)(id,SEL))objc_msgSend)(self, @selector(setIndex));
    uint64_t sigVal      = ((uint64_t(*)(id,SEL))objc_msgSend)(self, @selector(signalValue));
    BOOL sigNotReq       = ((BOOL(*)(id,SEL))objc_msgSend)(self, @selector(signalNotRequired));
    BOOL openLoop        = ((BOOL(*)(id,SEL))objc_msgSend)(self, @selector(isOpenLoop));
    [coder encodeInt32:(int32_t)procIdx forKey:@"procedureIndex"];
    [coder encodeInt32:(int32_t)setIdx forKey:@"setIndex"];
    [coder encodeInt64:(int64_t)sigVal forKey:@"signalValue"];
    [coder encodeBool:sigNotReq forKey:@"signalNotRequired"];
    [coder encodeBool:openLoop forKey:@"isOpenLoop"];
    id outputBuf = objc_getAssociatedObject(self, &kOutputBufferKey);
    if (outputBuf) [coder encodeObject:outputBuf forKey:@"outputBuffer"];
}
static id outputSetEnqueue_decode(id self, SEL _cmd, id coder) {
    unsigned int procIdx = (unsigned int)[coder decodeInt32ForKey:@"procedureIndex"];
    unsigned int setIdx  = (unsigned int)[coder decodeInt32ForKey:@"setIndex"];
    uint64_t sigVal      = (uint64_t)[coder decodeInt64ForKey:@"signalValue"];
    BOOL sigNotReq       = [coder decodeBoolForKey:@"signalNotRequired"];
    BOOL openLoop        = [coder decodeBoolForKey:@"isOpenLoop"];
    self = ((id(*)(id,SEL,unsigned int,unsigned int,uint64_t,BOOL,BOOL))objc_msgSend)(
        self, @selector(initOutputSetWithProcedureIndex:setIndex:signalValue:signalNotRequired:isOpenLoop:),
        procIdx, setIdx, sigVal, sigNotReq, openLoop);
    id outputBuf = [coder decodeObjectForKey:@"outputBuffer"];
    if (outputBuf) objc_setAssociatedObject(self, &kOutputBufferKey, outputBuf, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    return self;
}

static void apply_patches(void) {
    Class outSetClass = NSClassFromString(@"_ANEOutputSetEnqueue");
    class_addMethod(outSetClass, @selector(outputBuffer), (IMP)outputBuffer_getter, "@@:");
    class_addMethod(object_getClass(outSetClass), @selector(supportsSecureCoding), (IMP)supportsSecureCoding_YES, "B@:");
    class_addMethod(outSetClass, @selector(encodeWithCoder:), (IMP)outputSetEnqueue_encode, "v@:@");
    class_addMethod(outSetClass, @selector(initWithCoder:), (IMP)outputSetEnqueue_decode, "@@:@");
    class_addProtocol(outSetClass, @protocol(NSSecureCoding));
}

int main() {
    @autoreleasepool {
        setbuf(stdout, NULL);
        mach_timebase_info(&g_tb);
        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);

        printf("\n");
        printf("  ╔══════════════════════════════════════════════════════════╗\n");
        printf("  ║  ANE CHAINING v2 — Enqueue Investigation               ║\n");
        printf("  ╚══════════════════════════════════════════════════════════╝\n\n");

        apply_patches();

        Class outSetClass = NSClassFromString(@"_ANEOutputSetEnqueue");
        Class bufClass    = NSClassFromString(@"_ANEBuffer");
        Class chainClass  = NSClassFromString(@"_ANEChainingRequest");
        Class g_D  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class g_I  = NSClassFromString(@"_ANEInMemoryModel");
        Class g_AR = NSClassFromString(@"_ANERequest");
        Class g_AIO= NSClassFromString(@"_ANEIOSurfaceObject");
        Class clientClass = NSClassFromString(@"_ANEClient");
        Class aneModelClass = NSClassFromString(@"_ANEModel");

        // ══════════════════════════════════════════════════════════════
        // Compile identity kernel
        // ══════════════════════════════════════════════════════════════
        int CH = 64, SP = 64;
        _Float16 *ww = (_Float16*)calloc(CH*CH, sizeof(_Float16));
        for (int i = 0; i < CH; i++) ww[i*CH+i] = (_Float16)1.0f;
        int ws = CH*CH*2, ttot = 128+ws;
        uint8_t *bblob = (uint8_t*)calloc(ttot,1);
        bblob[0]=1; bblob[4]=2; bblob[64]=0xEF; bblob[65]=0xBE; bblob[66]=0xAD; bblob[67]=0xDE; bblob[68]=1;
        *(uint32_t*)(bblob+72)=ws; *(uint32_t*)(bblob+80)=128;
        memcpy(bblob+128, ww, ws);
        NSData *wdata = [NSData dataWithBytesNoCopy:bblob length:ttot freeWhenDone:YES];
        free(ww);
        NSString *mil = [NSString stringWithFormat:
            @"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
            "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
            "{\"coremltools-version\", \"9.0\"}})]\n{\n"
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
            "    } -> (y);\n}\n", CH, SP, CH, SP, CH, CH, CH, CH, CH, SP, CH, SP];
        NSData *md = [mil dataUsingEncoding:NSUTF8StringEncoding];
        id descObj = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D, @selector(modelWithMILText:weights:optionsPlist:),
            md, @{@"@model_path/weights/weight.bin": @{@"offset":@0, @"data":wdata}}, nil);
        id inMemMdl = ((id(*)(Class,SEL,id))objc_msgSend)(g_I, @selector(inMemoryModelWithDescriptor:), descObj);
        id hx = ((id(*)(id,SEL))objc_msgSend)(inMemMdl, @selector(hexStringIdentifier));
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        [[NSFileManager defaultManager] createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
            withIntermediateDirectories:YES attributes:nil error:nil];
        [md writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        [wdata writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];
        NSError *e = nil;
        ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(inMemMdl, @selector(compileWithQoS:options:error:), 21, @{}, &e);
        ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(inMemMdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);

        // Create _ANEModel for XPC — use modelAtURL:key: with the actual compiled model path
        NSURL *modelURL = [NSURL fileURLWithPath:td];
        NSData *modelKey = [hx dataUsingEncoding:NSUTF8StringEncoding];
        id aneModel = ((id(*)(Class,SEL,id,id))objc_msgSend)(aneModelClass,
            @selector(modelAtURL:key:), modelURL, modelKey);
        printf("  InMemoryModel: compiled+loaded\n");
        printf("  _ANEModel: %s\n", aneModel ? "OK" : "FAIL");
        if (aneModel) {
            @try {
                printf("    modelURL: %s\n", [[[aneModel valueForKey:@"modelURL"] description] UTF8String]);
                printf("    state: %s\n", [[[aneModel valueForKey:@"state"] description] UTF8String]);
                printf("    cacheURLIdentifier: %s\n", [[[aneModel valueForKey:@"cacheURLIdentifier"] description] UTF8String]);
            } @catch (NSException *ex) {}
        }

        // Create buffers
        int ioBytes = CH * SP * 4;
        IOSurfaceRef ioIn   = make_surface(ioBytes);
        IOSurfaceRef ioOut1 = make_surface(ioBytes);
        IOSurfaceRef ioOut2 = make_surface(ioBytes);
        IOSurfaceLock(ioIn, 0, NULL);
        float *inp = (float*)IOSurfaceGetBaseAddress(ioIn);
        for (int i = 0; i < CH*SP; i++) inp[i] = 1.0f;
        IOSurfaceUnlock(ioIn, 0, NULL);

        id wIn  = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioIn);
        id wOut1= ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioOut1);
        id wOut2= ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioOut2);

        id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wIn], @[@0], @[wOut1], @[@0], nil, nil, @0);

        id client = ((id(*)(Class,SEL))objc_msgSend)(clientClass, @selector(sharedConnection));

        // ══════════════════════════════════════════════════════════════
        // SECTION 1: prepareChaining + investigate enqueue
        // ══════════════════════════════════════════════════════════════
        printf("\n  ── Section 1: Prepare + Enqueue Investigation ──\n\n");

        SEL prepSel = @selector(prepareChainingWithModel:options:chainingReq:qos:error:);
        SEL enqSel = @selector(enqueueSetsWithModel:outputSet:options:qos:error:);
        SEL doEnqSel = @selector(doEnqueueSetsWithModel:outputSet:options:qos:error:);

        // Create the chaining objects
        id bufIn = ((id(*)(Class,SEL,id,id,long long))objc_msgSend)(
            bufClass, @selector(bufferWithIOSurfaceObject:symbolIndex:source:), wIn, @0, (long long)0);
        id buf1 = ((id(*)(Class,SEL,id,id,long long))objc_msgSend)(
            bufClass, @selector(bufferWithIOSurfaceObject:symbolIndex:source:), wOut1, @0, (long long)0);
        id buf2 = ((id(*)(Class,SEL,id,id,long long))objc_msgSend)(
            bufClass, @selector(bufferWithIOSurfaceObject:symbolIndex:source:), wOut2, @0, (long long)0);

        id os0 = ((id(*)(Class,SEL,unsigned int,unsigned int,uint64_t,BOOL,BOOL))objc_msgSend)(
            outSetClass,
            @selector(outputSetWithProcedureIndex:setIndex:signalValue:signalNotRequired:isOpenLoop:),
            0u, 0u, (uint64_t)1, NO, NO);
        id os1 = ((id(*)(Class,SEL,unsigned int,unsigned int,uint64_t,BOOL,BOOL))objc_msgSend)(
            outSetClass,
            @selector(outputSetWithProcedureIndex:setIndex:signalValue:signalNotRequired:isOpenLoop:),
            0u, 1u, (uint64_t)2, NO, NO);
        objc_setAssociatedObject(os0, &kOutputBufferKey, @[buf1], OBJC_ASSOCIATION_RETAIN_NONATOMIC);
        objc_setAssociatedObject(os1, &kOutputBufferKey, @[buf2], OBJC_ASSOCIATION_RETAIN_NONATOMIC);

        id chainReq = ((id(*)(Class,SEL,id,id,id,id,id,id,id,id,id))objc_msgSend)(
            chainClass,
            @selector(chainingRequestWithInputs:outputSets:lbInputSymbolId:lbOutputSymbolId:procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:),
            @[bufIn], @[os0, os1], @[@0], @[@0], @0, @[], @0, @0, @0);

        // Prepare
        e = nil;
        BOOL ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
            client, prepSel, aneModel, @{}, chainReq, 21, &e);
        printf("  prepareChainingWithModel: %s\n", ok ? "SUCCESS" : "FAIL");
        if (e) printf("    Error: %s\n", [[e description] UTF8String]);

        if (ok) {
            printf("\n  Testing enqueueSets with different approaches...\n\n");

            // First try to compile+load the _ANEModel through client
            printf("  Trying to load _ANEModel via client methods...\n");

            // Try compileModel:options:qos:error:
            SEL compileSel = @selector(compileModel:options:qos:error:);
            if ([client respondsToSelector:compileSel]) {
                @try {
                    e = nil;
                    BOOL compOk = ((BOOL(*)(id,SEL,id,id,unsigned int,NSError**))objc_msgSend)(
                        client, compileSel, aneModel, @{}, 21, &e);
                    printf("    compileModel: %s\n", compOk ? "OK" : "FAIL");
                    if (e) printf("      Error: %s\n", [[e description] UTF8String]);
                } @catch (NSException *ex) {
                    printf("    compileModel exception: %s\n", [[ex reason] UTF8String]);
                }
            }

            // Try loadModel:options:qos:error:
            SEL loadModelSel = @selector(loadModel:options:qos:error:);
            if ([client respondsToSelector:loadModelSel]) {
                @try {
                    e = nil;
                    BOOL loadOk = ((BOOL(*)(id,SEL,id,id,unsigned int,NSError**))objc_msgSend)(
                        client, loadModelSel, aneModel, @{}, 21, &e);
                    printf("    loadModel: %s\n", loadOk ? "OK" : "FAIL");
                    if (e) printf("      Error: %s\n", [[e description] UTF8String]);
                } @catch (NSException *ex) {
                    printf("    loadModel exception: %s\n", [[ex reason] UTF8String]);
                }
            }

            // Check state
            @try {
                printf("    state=%s programHandle=%s\n",
                    [[[aneModel valueForKey:@"state"] description] UTF8String],
                    [[[aneModel valueForKey:@"programHandle"] description] UTF8String]);
            } @catch (NSException *ex) {}

            // Try eval directly with _ANEModel (like InMemoryModel)
            printf("  Trying eval with _ANEModel...\n");
            @try {
                e = nil;
                BOOL evalOk = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                    aneModel, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
                printf("    eval: %s\n", evalOk ? "OK" : "FAIL");
                if (e) printf("      Error: %s\n", [[e description] UTF8String]);
            } @catch (NSException *ex) {
                printf("    eval exception: %s\n", [[ex reason] UTF8String]);
            }

            // Try doEvaluateWithModel through client
            SEL doEvalSel = @selector(doEvaluateWithModel:options:request:qos:error:);
            if ([client respondsToSelector:doEvalSel]) {
                @try {
                    e = nil;
                    BOOL evalOk = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                        client, doEvalSel, aneModel, @{}, req, 21, &e);
                    printf("    doEvaluateWithModel: %s\n", evalOk ? "OK" : "FAIL");
                    if (e) printf("      Error: %s\n", [[e description] UTF8String]);
                } @catch (NSException *ex) {
                    printf("    doEval exception: %s\n", [[ex reason] UTF8String]);
                }
            }

            printf("\n  _ANEModel after load attempts:\n");
            @try { dump_obj_properties(aneModel); } @catch (NSException *ex) {}

            // Approach 1: single OutputSetEnqueue
            printf("\n  1a: enqueueSets(os0) — single OutputSetEnqueue\n");
            @try {
                e = nil;
                ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                    client, enqSel, aneModel, os0, @{}, 21, &e);
                printf("    Result: %s\n", ok ? "OK" : "FAIL");
                if (e) printf("    Error: %s\n", [[e description] UTF8String]);
            } @catch (NSException *ex) {
                printf("    Exception: %s\n", [[ex reason] UTF8String]);
            }

            // Approach 2: doEnqueueSets
            printf("\n  1b: doEnqueueSets(os0)\n");
            @try {
                e = nil;
                ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                    client, doEnqSel, aneModel, os0, @{}, 21, &e);
                printf("    Result: %s\n", ok ? "OK" : "FAIL");
                if (e) printf("    Error: %s\n", [[e description] UTF8String]);
            } @catch (NSException *ex) {
                printf("    Exception: %s\n", [[ex reason] UTF8String]);
            }

            // Approach 3: try with InMemoryModel for enqueue
            printf("\n  1c: enqueueSets with InMemoryModel\n");
            @try {
                e = nil;
                ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                    client, enqSel, inMemMdl, os0, @{}, 21, &e);
                printf("    Result: %s\n", ok ? "OK" : "FAIL");
                if (e) printf("    Error: %s\n", [[e description] UTF8String]);
            } @catch (NSException *ex) {
                printf("    Exception: %s\n", [[ex reason] UTF8String]);
            }

            // Approach 4: try with array of output sets
            printf("\n  1d: enqueueSets(@[os0, os1])\n");
            @try {
                e = nil;
                ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                    client, enqSel, aneModel, @[os0, os1], @{}, 21, &e);
                printf("    Result: %s\n", ok ? "OK" : "FAIL");
                if (e) printf("    Error: %s\n", [[e description] UTF8String]);
            } @catch (NSException *ex) {
                printf("    Exception: %s\n", [[ex reason] UTF8String]);
            }

            // Approach 5: Try buffersReadyWithModel first
            printf("\n  1e: buffersReadyWithModel then enqueueSets\n");
            SEL bufReadySel = @selector(buffersReadyWithModel:inputBuffers:options:qos:error:);
            if ([client respondsToSelector:bufReadySel]) {
                @try {
                    e = nil;
                    ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                        client, bufReadySel, aneModel, @[bufIn], @{}, 21, &e);
                    printf("    buffersReady: %s\n", ok ? "OK" : "FAIL");
                    if (e) printf("    Error: %s\n", [[e description] UTF8String]);

                    if (ok) {
                        e = nil;
                        ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                            client, enqSel, aneModel, os0, @{}, 21, &e);
                        printf("    enqueueSets after buffersReady: %s\n", ok ? "OK" : "FAIL");
                        if (e) printf("    Error: %s\n", [[e description] UTF8String]);
                    }
                } @catch (NSException *ex) {
                    printf("    Exception: %s\n", [[ex reason] UTF8String]);
                }
            }

            // Approach 6: try with QoS=9
            printf("\n  1f: enqueueSets with QoS=9\n");
            @try {
                e = nil;
                ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                    client, enqSel, aneModel, os0, @{}, 9, &e);
                printf("    Result: %s\n", ok ? "OK" : "FAIL");
                if (e) printf("    Error: %s\n", [[e description] UTF8String]);
            } @catch (NSException *ex) {
                printf("    Exception: %s\n", [[ex reason] UTF8String]);
            }

            // Approach 7: prepare with QoS=9, then enqueue with QoS=9
            printf("\n  1g: prepareChaining(q9) then enqueue(q9)\n");
            @try {
                e = nil;
                ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                    client, prepSel, aneModel, @{}, chainReq, 9, &e);
                printf("    prepareChaining(q9): %s\n", ok ? "OK" : "FAIL");
                if (e) printf("    Error: %s\n", [[e description] UTF8String]);

                if (ok) {
                    e = nil;
                    ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                        client, enqSel, aneModel, os0, @{}, 9, &e);
                    printf("    enqueueSets(q9): %s\n", ok ? "OK" : "FAIL");
                    if (e) printf("    Error: %s\n", [[e description] UTF8String]);
                }
            } @catch (NSException *ex) {
                printf("    Exception: %s\n", [[ex reason] UTF8String]);
            }

            // Approach 8: try open-loop prepare + enqueue
            printf("\n  1h: open-loop prepare + enqueue\n");
            @try {
                id osOL0 = ((id(*)(Class,SEL,unsigned int,unsigned int,uint64_t,BOOL,BOOL))objc_msgSend)(
                    outSetClass,
                    @selector(outputSetWithProcedureIndex:setIndex:signalValue:signalNotRequired:isOpenLoop:),
                    0u, 0u, (uint64_t)1, YES, YES);
                id osOL1 = ((id(*)(Class,SEL,unsigned int,unsigned int,uint64_t,BOOL,BOOL))objc_msgSend)(
                    outSetClass,
                    @selector(outputSetWithProcedureIndex:setIndex:signalValue:signalNotRequired:isOpenLoop:),
                    0u, 1u, (uint64_t)2, YES, YES);
                objc_setAssociatedObject(osOL0, &kOutputBufferKey, @[buf1], OBJC_ASSOCIATION_RETAIN_NONATOMIC);
                objc_setAssociatedObject(osOL1, &kOutputBufferKey, @[buf2], OBJC_ASSOCIATION_RETAIN_NONATOMIC);

                id chainOL = ((id(*)(Class,SEL,id,id,id,id,id,id,id,id,id))objc_msgSend)(
                    chainClass,
                    @selector(chainingRequestWithInputs:outputSets:lbInputSymbolId:lbOutputSymbolId:procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:),
                    @[bufIn], @[osOL0, osOL1], @[@0], @[@0], @0, @[], @0, @0, @0);

                e = nil;
                ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                    client, prepSel, aneModel, @{}, chainOL, 21, &e);
                printf("    prepareChaining(openLoop): %s\n", ok ? "OK" : "FAIL");
                if (e) printf("    Error: %s\n", [[e description] UTF8String]);

                if (ok) {
                    e = nil;
                    ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                        client, enqSel, aneModel, osOL0, @{}, 21, &e);
                    printf("    enqueueSets(openLoop): %s\n", ok ? "OK" : "FAIL");
                    if (e) printf("    Error: %s\n", [[e description] UTF8String]);
                }
            } @catch (NSException *ex) {
                printf("    Exception: %s\n", [[ex reason] UTF8String]);
            }

            // Approach 9: Try the method signature more carefully
            // enqueueSetsWithModel:outputSet:options:qos:error:
            // Type: B52@0:8@16@24@32I40^@44
            // outputSet is the second id param (offset 24) — could it be an index or something else?
            printf("\n  1i: enqueueSets with NSNumber(0) as outputSet\n");
            @try {
                e = nil;
                ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                    client, enqSel, aneModel, @0, @{}, 21, &e);
                printf("    Result: %s\n", ok ? "OK" : "FAIL");
                if (e) printf("    Error: %s\n", [[e description] UTF8String]);
            } @catch (NSException *ex) {
                printf("    Exception: %s\n", [[ex reason] UTF8String]);
            }

            printf("\n  1j: enqueueSets with nil as outputSet\n");
            @try {
                e = nil;
                ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                    client, enqSel, aneModel, nil, @{}, 21, &e);
                printf("    Result: %s\n", ok ? "OK" : "FAIL");
                if (e) printf("    Error: %s\n", [[e description] UTF8String]);
            } @catch (NSException *ex) {
                printf("    Exception: %s\n", [[ex reason] UTF8String]);
            }
        }

        // ══════════════════════════════════════════════════════════════
        // SECTION 2: Check if we need to load the _ANEModel first
        // ══════════════════════════════════════════════════════════════
        printf("\n  ── Section 2: Load _ANEModel + Retry ──\n\n");

        // The _ANEModel has state=1 (compiled but not loaded?)
        // Try loading it through the client
        SEL loadSel = @selector(loadModel:options:qos:error:);
        if ([client respondsToSelector:loadSel]) {
            @try {
                e = nil;
                ok = ((BOOL(*)(id,SEL,id,id,unsigned int,NSError**))objc_msgSend)(
                    client, loadSel, aneModel, @{}, 21, &e);
                printf("  loadModel via client: %s\n", ok ? "OK" : "FAIL");
                if (e) printf("    Error: %s\n", [[e description] UTF8String]);
            } @catch (NSException *ex) {
                printf("  loadModel exception: %s\n", [[ex reason] UTF8String]);
            }
        } else {
            printf("  loadModel: method not found\n");
            // Check what load methods the client has
            unsigned int mc;
            Method *ms = class_copyMethodList([client class], &mc);
            printf("  Client 'load' methods:\n");
            for (unsigned int i = 0; i < mc; i++) {
                const char *sname = sel_getName(method_getName(ms[i]));
                if (strstr(sname, "load") || strstr(sname, "Load")) {
                    const char *enc = method_getTypeEncoding(ms[i]);
                    printf("    - %s  [%s]\n", sname, enc ? enc : "?");
                }
            }
            free(ms);
        }

        // Check model state
        @try {
            id state = [aneModel valueForKey:@"state"];
            id pgmHandle = [aneModel valueForKey:@"programHandle"];
            printf("  _ANEModel state: %s  programHandle: %s\n",
                   [state description].UTF8String, [pgmHandle description].UTF8String);
        } @catch (NSException *ex) {}

        // Try compile+load the _ANEModel through the client
        SEL compileLoadSel = @selector(compileModel:options:qos:error:);
        if ([client respondsToSelector:compileLoadSel]) {
            @try {
                e = nil;
                ok = ((BOOL(*)(id,SEL,id,id,unsigned int,NSError**))objc_msgSend)(
                    client, compileLoadSel, aneModel, @{}, 21, &e);
                printf("  compileModel via client: %s\n", ok ? "OK" : "FAIL");
                if (e) printf("    Error: %s\n", [[e description] UTF8String]);
            } @catch (NSException *ex) {
                printf("  compileModel exception: %s\n", [[ex reason] UTF8String]);
            }
        }

        // Now try prepare + enqueue again
        printf("\n  Retry after potential load...\n");
        @try {
            id state = [aneModel valueForKey:@"state"];
            id pgmHandle = [aneModel valueForKey:@"programHandle"];
            printf("  State: %s  programHandle: %s\n",
                   [state description].UTF8String, [pgmHandle description].UTF8String);
        } @catch (NSException *ex) {}

        {
            id bufInR = ((id(*)(Class,SEL,id,id,long long))objc_msgSend)(
                bufClass, @selector(bufferWithIOSurfaceObject:symbolIndex:source:), wIn, @0, (long long)0);
            id buf1R = ((id(*)(Class,SEL,id,id,long long))objc_msgSend)(
                bufClass, @selector(bufferWithIOSurfaceObject:symbolIndex:source:), wOut1, @0, (long long)0);
            id buf2R = ((id(*)(Class,SEL,id,id,long long))objc_msgSend)(
                bufClass, @selector(bufferWithIOSurfaceObject:symbolIndex:source:), wOut2, @0, (long long)0);
            id os0R = ((id(*)(Class,SEL,unsigned int,unsigned int,uint64_t,BOOL,BOOL))objc_msgSend)(
                outSetClass,
                @selector(outputSetWithProcedureIndex:setIndex:signalValue:signalNotRequired:isOpenLoop:),
                0u, 0u, (uint64_t)1, NO, NO);
            id os1R = ((id(*)(Class,SEL,unsigned int,unsigned int,uint64_t,BOOL,BOOL))objc_msgSend)(
                outSetClass,
                @selector(outputSetWithProcedureIndex:setIndex:signalValue:signalNotRequired:isOpenLoop:),
                0u, 1u, (uint64_t)2, NO, NO);
            objc_setAssociatedObject(os0R, &kOutputBufferKey, @[buf1R], OBJC_ASSOCIATION_RETAIN_NONATOMIC);
            objc_setAssociatedObject(os1R, &kOutputBufferKey, @[buf2R], OBJC_ASSOCIATION_RETAIN_NONATOMIC);

            id chainR = ((id(*)(Class,SEL,id,id,id,id,id,id,id,id,id))objc_msgSend)(
                chainClass,
                @selector(chainingRequestWithInputs:outputSets:lbInputSymbolId:lbOutputSymbolId:procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:),
                @[bufInR], @[os0R, os1R], @[@0], @[@0], @0, @[], @0, @0, @0);

            @try {
                e = nil;
                ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                    client, prepSel, aneModel, @{}, chainR, 21, &e);
                printf("  prepareChainingWithModel: %s\n", ok ? "SUCCESS" : "FAIL");
                if (e) printf("    Error: %s\n", [[e description] UTF8String]);

                if (ok) {
                    e = nil;
                    ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                        client, enqSel, aneModel, os0R, @{}, 21, &e);
                    printf("  enqueueSets: %s\n", ok ? "*** OK ***" : "FAIL");
                    if (e) printf("    Error: %s\n", [[e description] UTF8String]);

                    if (!ok) {
                        // Try doEnqueueSets
                        e = nil;
                        ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                            client, doEnqSel, aneModel, os0R, @{}, 21, &e);
                        printf("  doEnqueueSets: %s\n", ok ? "*** OK ***" : "FAIL");
                        if (e) printf("    Error: %s\n", [[e description] UTF8String]);
                    }

                    if (ok) {
                        // Check output
                        IOSurfaceLock(ioOut1, kIOSurfaceLockReadOnly, NULL);
                        float *out = (float*)IOSurfaceGetBaseAddress(ioOut1);
                        float osum = 0;
                        for (int i = 0; i < 64; i++) osum += out[i];
                        printf("  Output: first 64 floats sum = %.1f\n", osum);
                        IOSurfaceUnlock(ioOut1, kIOSurfaceLockReadOnly, NULL);
                    }
                }
            } @catch (NSException *ex) {
                printf("  Exception: %s\n", [[ex reason] UTF8String]);
            }
        }

        // ══════════════════════════════════════════════════════════════
        // SECTION 3: Benchmark baseline
        // ══════════════════════════════════════════════════════════════
        printf("\n  ── Section 3: Benchmark ──\n\n");

        int N = 100, WARMUP = 50;
        for (int i = 0; i < WARMUP; i++)
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                inMemMdl, @selector(evaluateWithQoS:options:request:error:), 9, @{}, req, &e);

        uint64_t t0 = mach_absolute_time();
        for (int i = 0; i < N; i++)
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                inMemMdl, @selector(evaluateWithQoS:options:request:error:), 9, @{}, req, &e);
        double stdMs = tb_ms(mach_absolute_time() - t0);

        id reqLoop = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wOut1], @[@0], @[wOut1], @[@0], nil, nil, @0);
        IOSurfaceLock(ioOut1, 0, NULL);
        float *o1 = (float*)IOSurfaceGetBaseAddress(ioOut1);
        for (int i = 0; i < CH*SP; i++) o1[i] = 1.0f;
        IOSurfaceUnlock(ioOut1, 0, NULL);
        for (int i = 0; i < WARMUP; i++)
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                inMemMdl, @selector(evaluateWithQoS:options:request:error:), 9, @{}, reqLoop, &e);
        t0 = mach_absolute_time();
        for (int i = 0; i < N; i++)
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                inMemMdl, @selector(evaluateWithQoS:options:request:error:), 9, @{}, reqLoop, &e);
        double loopMs = tb_ms(mach_absolute_time() - t0);

        printf("  Standard eval: %.0f us/iter (QoS=9)\n", stdMs/N*1000);
        printf("  Same-surface loopback: %.0f us/iter (QoS=9)\n", loopMs/N*1000);

        // ══════════════════════════════════════════════════════════════
        // SUMMARY
        // ══════════════════════════════════════════════════════════════
        printf("\n  ══════════════════════════════════════════════════════════\n");
        printf("  SUMMARY\n");
        printf("  ══════════════════════════════════════════════════════════\n\n");
        printf("  prepareChainingWithModel: SUCCESS (validated + XPC transport works)\n");
        printf("  enqueueSetsWithModel: see results above\n");
        printf("  Standard eval: %.0f us/iter\n", stdMs/N*1000);
        printf("  Loopback eval: %.0f us/iter\n\n", loopMs/N*1000);

        ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(inMemMdl, @selector(unloadWithQoS:error:), 21, &e);
        CFRelease(ioIn); CFRelease(ioOut1); CFRelease(ioOut2);
    }
    return 0;
}
