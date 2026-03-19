// test_compiler_analytics.m — Probe _ANECompilerAnalytics and sub-structures
// Goal: Extract per-layer SRAM usage and spill information from ANE compiler
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>

// ── Utility: dump all methods/properties/ivars of a class ──
static void dump_class(const char *name) {
    Class cls = NSClassFromString([NSString stringWithUTF8String:name]);
    if (!cls) { printf("  [%s] NOT FOUND\n\n", name); return; }
    printf("  [%s] (super: %s)\n", name, class_getName(class_getSuperclass(cls)));
    unsigned int count = 0;
    Method *methods = class_copyMethodList(cls, &count);
    for (unsigned int i = 0; i < count; i++) {
        SEL sel = method_getName(methods[i]);
        printf("    - %s  (%s)\n", sel_getName(sel), method_getTypeEncoding(methods[i]) ?: "?");
    }
    free(methods);
    Class meta = object_getClass((id)cls);
    methods = class_copyMethodList(meta, &count);
    for (unsigned int i = 0; i < count; i++) {
        SEL sel = method_getName(methods[i]);
        printf("    + %s  (%s)\n", sel_getName(sel), method_getTypeEncoding(methods[i]) ?: "?");
    }
    free(methods);
    unsigned int ic = 0;
    Ivar *ivars = class_copyIvarList(cls, &ic);
    for (unsigned int i = 0; i < ic; i++)
        printf("    @ %s  (%s)\n", ivar_getName(ivars[i]), ivar_getTypeEncoding(ivars[i]) ?: "?");
    free(ivars);
    printf("\n");
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
    BOOL compOk = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e);
    if (!compOk) { printf("    compile FAILED\n"); return nil; }
    e = nil;
    ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
    return mdl;
}

int main() {
    @autoreleasepool {
        setbuf(stdout, NULL);
        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);

        printf("\n  ██████ ANE COMPILER ANALYTICS PROBE ██████\n\n");

        // ════════════════════════════════════════════════════════════════
        // Section 1: Dump ALL _ANE* classes
        // ════════════════════════════════════════════════════════════════
        printf("  ══ All _ANE* classes ══\n\n");
        {
            unsigned int cc = 0;
            Class *classes = objc_copyClassList(&cc);
            NSMutableArray *names = [NSMutableArray array];
            for (unsigned int i = 0; i < cc; i++) {
                const char *cn = class_getName(classes[i]);
                if (strncmp(cn, "_ANE", 4) == 0)
                    [names addObject:[NSString stringWithUTF8String:cn]];
            }
            free(classes);
            [names sortUsingSelector:@selector(compare:)];
            for (NSString *n in names) printf("    %s\n", [n UTF8String]);
            printf("    Total: %lu\n\n", (unsigned long)[names count]);
        }

        // ════════════════════════════════════════════════════════════════
        // Section 2: Dump analytics class hierarchy
        // ════════════════════════════════════════════════════════════════
        printf("  ══ Analytics class hierarchy ══\n\n");
        dump_class("_ANECompilerAnalytics");
        dump_class("_ANEAnalyticsProcedure");
        dump_class("_ANEAnalyticsLayer");
        dump_class("_ANEAnalyticsTask");
        dump_class("_ANEAnalyticsGroup");
        dump_class("_ANEDataReporter");
        dump_class("_ANEProcedureData");

        // ════════════════════════════════════════════════════════════════
        // Section 3: Dump _ANEDaemonConnection (the XPC proxy)
        // ════════════════════════════════════════════════════════════════
        printf("  ══ _ANEDaemonConnection (XPC to aned) ══\n\n");
        dump_class("_ANEDaemonConnection");

        // ════════════════════════════════════════════════════════════════
        // Section 4: Try to instantiate _ANECompilerAnalytics with buffer
        // ════════════════════════════════════════════════════════════════
        printf("  ══ _ANECompilerAnalytics instantiation ══\n\n");
        Class analyticsCls = NSClassFromString(@"_ANECompilerAnalytics");
        if (analyticsCls) {
            // alloc/init
            @try {
                id a = [[analyticsCls alloc] init];
                printf("  alloc/init: %s\n", a ? [[a description] UTF8String] : "nil");
            } @catch (NSException *ex) { printf("  alloc/init: EXC %s\n", [[ex reason] UTF8String]); }

            // +new
            @try {
                id a = ((id(*)(Class,SEL))objc_msgSend)(analyticsCls, @selector(new));
                printf("  +new: %s\n", a ? [[a description] UTF8String] : "nil");
            } @catch (NSException *ex) { printf("  +new: EXC %s\n", [[ex reason] UTF8String]); }

            // +objectWithBuffer: empty
            @try {
                id a = ((id(*)(Class,SEL,id))objc_msgSend)(analyticsCls, @selector(objectWithBuffer:), [NSData data]);
                printf("  +objectWithBuffer:(0 bytes): %s\n", a ? [[a description] UTF8String] : "nil");
            } @catch (NSException *ex) { printf("  +objectWithBuffer:(0): EXC\n"); }

            // +objectWithBuffer: 4KB zeros
            @try {
                NSMutableData *buf = [NSMutableData dataWithLength:4096];
                id a = ((id(*)(Class,SEL,id))objc_msgSend)(analyticsCls, @selector(objectWithBuffer:), buf);
                printf("  +objectWithBuffer:(4096 bytes zeros): %s\n", a ? [[a description] UTF8String] : "nil");
                if (a) {
                    BOOL pop = ((BOOL(*)(id,SEL))objc_msgSend)(a, @selector(populateAnalytics));
                    printf("    populateAnalytics: %s\n", pop ? "YES" : "NO");
                    id pa = ((id(*)(id,SEL))objc_msgSend)(a, @selector(procedureAnalytics));
                    printf("    procedureAnalytics: %s\n", pa ? [[pa description] UTF8String] : "nil");
                    id ser = ((id(*)(id,SEL))objc_msgSend)(a, @selector(serialize));
                    printf("    serialize: %s\n", ser ? [[ser description] UTF8String] : "nil");

                    // Try stringForAnalyticsType for various IDs
                    printf("    stringForAnalyticsType:\n");
                    for (unsigned int t = 0; t < 32; t++) {
                        @try {
                            id s = ((id(*)(id,SEL,unsigned int))objc_msgSend)(a, @selector(stringForAnalyticsType:), t);
                            if (s && [(NSString*)s length] > 0)
                                printf("      [%u] = %s\n", t, [(NSString*)s UTF8String]);
                        } @catch (NSException *ex) {}
                    }
                }
            } @catch (NSException *ex) { printf("  +objectWithBuffer:(4096): EXC %s\n", [[ex reason] UTF8String]); }

            // +objectWithBuffer: nil
            @try {
                id a = ((id(*)(Class,SEL,id))objc_msgSend)(analyticsCls, @selector(objectWithBuffer:), nil);
                printf("  +objectWithBuffer:(nil): %s\n", a ? [[a description] UTF8String] : "nil");
            } @catch (NSException *ex) { printf("  +objectWithBuffer:(nil): EXC\n"); }
        }
        printf("\n");

        // ════════════════════════════════════════════════════════════════
        // Section 5: Try sub-classes
        // ════════════════════════════════════════════════════════════════
        printf("  ══ Sub-class instantiation ══\n\n");
        const char *subNames[] = {
            "_ANEAnalyticsProcedure", "_ANEAnalyticsLayer",
            "_ANEAnalyticsTask", "_ANEAnalyticsGroup", NULL
        };
        for (int i = 0; subNames[i]; i++) {
            Class cls = NSClassFromString([NSString stringWithUTF8String:subNames[i]]);
            if (!cls) { printf("  [%s] NOT FOUND\n", subNames[i]); continue; }
            @try {
                id inst = [[cls alloc] init];
                printf("  [%s] alloc/init: %s\n", subNames[i], inst ? "created" : "nil");
                if (inst) {
                    unsigned int mc = 0;
                    Method *meths = class_copyMethodList(cls, &mc);
                    for (unsigned int m = 0; m < mc; m++) {
                        SEL sel = method_getName(meths[m]);
                        const char *sn = sel_getName(sel);
                        if (strncmp(sn, "set", 3) == 0 || strcmp(sn, ".cxx_destruct") == 0 ||
                            strcmp(sn, "dealloc") == 0 || strcmp(sn, "init") == 0) continue;
                        int colons = 0;
                        for (const char *p = sn; *p; p++) if (*p == ':') colons++;
                        if (colons > 0) continue;
                        @try {
                            id val = ((id(*)(id,SEL))objc_msgSend)(inst, sel);
                            printf("    .%s -> %s\n", sn, val ? [[val description] UTF8String] : "nil");
                        } @catch (NSException *ex) { printf("    .%s -> EXC\n", sn); }
                    }
                    free(meths);
                }
            } @catch (NSException *ex) { printf("  [%s] EXC\n", subNames[i]); }
        }
        printf("\n");

        // ════════════════════════════════════════════════════════════════
        // Section 6: Compile kernels and probe for analytics via model
        // ════════════════════════════════════════════════════════════════
        printf("  ══ Compile + probe for analytics ══\n\n");

        Class g_D = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class g_I = NSClassFromString(@"_ANEInMemoryModel");

        struct { int ch; int sp; const char *tag; } kernels[] = {
            {256, 64, "small"},
            {1024, 256, "medium"},
            {2048, 128, "large"},
        };

        for (int k = 0; k < 3; k++) {
            printf("  ── %s (%dx%d sp%d) ──\n", kernels[k].tag, kernels[k].ch, kernels[k].ch, kernels[k].sp);
            NSData *wdata = nil;
            id mdl = compile_kernel(g_D, g_I, kernels[k].ch, kernels[k].sp, &wdata);
            if (!mdl) continue;
            printf("    OK\n");

            // Get _ANEModel
            Ivar modelIvar = class_getInstanceVariable([mdl class], "_model");
            id aneModel = modelIvar ? object_getIvar(mdl, modelIvar) : nil;

            // Dump modelAttributes
            NSDictionary *attrs = nil;
            if (aneModel) {
                attrs = ((id(*)(id,SEL))objc_msgSend)(aneModel, @selector(modelAttributes));
            }
            if (attrs) {
                NSData *json = [NSJSONSerialization dataWithJSONObject:attrs
                    options:NSJSONWritingPrettyPrinted|NSJSONWritingSortedKeys error:nil];
                if (json) {
                    NSString *str = [[NSString alloc] initWithData:json encoding:NSUTF8StringEncoding];
                    printf("    modelAttributes:\n%s\n", [str UTF8String]);
                }
            }

            // Check programHandle and intermediateBufferHandle
            if (aneModel) {
                uint64_t ph = ((uint64_t(*)(id,SEL))objc_msgSend)(aneModel, @selector(programHandle));
                uint64_t ibh = ((uint64_t(*)(id,SEL))objc_msgSend)(aneModel, @selector(intermediateBufferHandle));
                printf("    programHandle: %llu\n", ph);
                printf("    intermediateBufferHandle: %llu\n", ibh);
            }

            // Try to get connection and call compileModel with analytics
            @try {
                Ivar connIvar = class_getInstanceVariable([mdl class], "_sharedConnection");
                id client = connIvar ? object_getIvar(mdl, connIvar) : nil;
                if (client) {
                    printf("    _ANEClient: %s\n", [[client description] UTF8String]);
                    Ivar daemonIvar = class_getInstanceVariable([client class], "_conn");
                    if (!daemonIvar) daemonIvar = class_getInstanceVariable([client class], "_connections");
                    id daemon = daemonIvar ? object_getIvar(client, daemonIvar) : nil;
                    printf("    daemon conn: %s (%s)\n",
                        daemon ? [[daemon description] UTF8String] : "nil",
                        daemon ? class_getName(object_getClass(daemon)) : "?");

                    // If it's a dictionary/array of connections, show it
                    if (daemon && [daemon isKindOfClass:[NSDictionary class]]) {
                        printf("    connections dict:\n");
                        for (id key in (NSDictionary*)daemon) {
                            id val = [(NSDictionary*)daemon objectForKey:key];
                            printf("      %s -> %s (%s)\n",
                                [[key description] UTF8String],
                                [[val description] UTF8String],
                                class_getName(object_getClass(val)));
                        }
                    }
                    if (daemon && [daemon isKindOfClass:[NSArray class]]) {
                        for (id item in (NSArray*)daemon) {
                            printf("      connection: %s (%s)\n",
                                [[item description] UTF8String],
                                class_getName(object_getClass(item)));
                        }
                    }
                }
            } @catch (NSException *ex) {
                printf("    Client probe EXC: %s\n", [[ex reason] UTF8String]);
            }

            // Unload
            NSError *ue = nil;
            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl, @selector(unloadWithQoS:error:), 21, &ue);
            printf("\n");
        }

        // ════════════════════════════════════════════════════════════════
        // Section 7: Espresso analytics classes
        // ════════════════════════════════════════════════════════════════
        printf("  ══ Espresso analytics ══\n\n");
        void *esp = dlopen("/System/Library/PrivateFrameworks/Espresso.framework/Espresso", RTLD_NOW);
        printf("  Espresso: %s\n\n", esp ? "loaded" : "not found");
        if (esp) {
            // Scan for all Espresso profiling classes
            unsigned int cc = 0;
            Class *classes = objc_copyClassList(&cc);
            printf("  Espresso profiling/ANE classes:\n");
            for (unsigned int i = 0; i < cc; i++) {
                const char *cn = class_getName(classes[i]);
                if (strncmp(cn, "EspressoProfiling", 17) == 0 ||
                    (strncmp(cn, "Espresso", 8) == 0 && strstr(cn, "ANE"))) {
                    printf("    %s\n", cn);
                }
            }
            free(classes);
            printf("\n");

            dump_class("EspressoProfilingANEcompilerAnalytics");
            dump_class("EspressoProfilingNetworkANEInfo");

            // Try to instantiate
            Class espAnaCls = NSClassFromString(@"EspressoProfilingANEcompilerAnalytics");
            if (espAnaCls) {
                @try {
                    id inst = [[espAnaCls alloc] init];
                    printf("  EspressoProfilingANEcompilerAnalytics: %s\n",
                        inst ? "created" : "nil");
                    if (inst) {
                        unsigned int mc = 0;
                        Method *meths = class_copyMethodList(espAnaCls, &mc);
                        for (unsigned int m = 0; m < mc; m++) {
                            SEL sel = method_getName(meths[m]);
                            const char *sn = sel_getName(sel);
                            if (strncmp(sn, "set", 3) == 0 || strcmp(sn, ".cxx_destruct") == 0 ||
                                strcmp(sn, "dealloc") == 0 || strncmp(sn, "init", 4) == 0) continue;
                            int colons = 0;
                            for (const char *p = sn; *p; p++) if (*p == ':') colons++;
                            if (colons > 0) continue;
                            @try {
                                id val = ((id(*)(id,SEL))objc_msgSend)(inst, sel);
                                printf("    .%s -> %s\n", sn, val ? [[val description] UTF8String] : "nil");
                            } @catch (NSException *ex) {
                                printf("    .%s -> EXC: %s\n", sn, [[ex reason] UTF8String]);
                            }
                        }
                        free(meths);
                    }
                } @catch (NSException *ex) {
                    printf("  EXC: %s\n", [[ex reason] UTF8String]);
                }
            }
        }

        // ════════════════════════════════════════════════════════════════
        // Section 8: Summary of struct layouts
        // ════════════════════════════════════════════════════════════════
        printf("\n  ══ Struct layouts (from type encodings) ══\n\n");
        printf("  _AnalyticsProcedureInfo = {IIIIIQIQ}\n");
        printf("    uint32 field0\n");
        printf("    uint32 field1\n");
        printf("    uint32 field2\n");
        printf("    uint32 field3\n");
        printf("    uint32 field4\n");
        printf("    uint64 field5\n");
        printf("    uint32 field6\n");
        printf("    uint64 field7\n");
        printf("    (total 48 bytes with padding)\n\n");

        printf("  _AnalyticsLayerInfo = {[64c][64c]f}\n");
        printf("    char name[64]\n");
        printf("    char type[64]\n");
        printf("    float value  (likely SRAM weight/cost)\n");
        printf("    (total 132 bytes)\n\n");

        printf("  _AnalyticsTaskInfo = {IQ}\n");
        printf("    uint32 field0\n");
        printf("    uint64 field1\n");
        printf("    (total 16 bytes)\n\n");

        printf("  _AnalyticsGroupInfo = {IQIQ}\n");
        printf("    uint32 field0\n");
        printf("    uint64 field1\n");
        printf("    uint32 field2\n");
        printf("    uint64 field3\n");
        printf("    (total 32 bytes)\n\n");

        printf("  _AnalyticsData = {II[0c]}\n");
        printf("    uint32 field0\n");
        printf("    uint32 field1\n");
        printf("    char data[]  (flexible)\n\n");

        printf("\n  ██████ PROBE COMPLETE ██████\n\n");
    }
    return 0;
}
