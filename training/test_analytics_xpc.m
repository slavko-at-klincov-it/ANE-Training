// test_analytics_xpc.m — Intercept ANE compiler analytics via XPC reply swizzling
// Goal: Capture the raw analytics buffer from aned daemon's compile reply
//
// Strategy:
//   1. Dump all compile-related methods on _ANEDaemonConnection, _ANEClient, _ANEVirtualClient
//   2. Swizzle compile methods at multiple levels to intercept reply blocks
//   3. Inspect reply block arguments for analytics buffer (NSData)
//   4. If found, parse with _ANECompilerAnalytics +objectWithBuffer:
//   5. Also try compile options that might trigger analytics (kANEFPerformanceStatsMask, etc.)

#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>

// ── Globals for captured data ──
static NSMutableArray *g_capturedReplies = nil;
static NSMutableArray *g_capturedBlocks = nil;
static BOOL g_swizzleActive = NO;

// ── Utility: dump methods matching substring ──
static void dump_methods_matching(const char *className, const char *substr) {
    Class cls = NSClassFromString([NSString stringWithUTF8String:className]);
    if (!cls) { printf("  [%s] NOT FOUND\n", className); return; }
    printf("  [%s] methods containing '%s':\n", className, substr);

    // Instance methods
    unsigned int count = 0;
    Method *methods = class_copyMethodList(cls, &count);
    for (unsigned int i = 0; i < count; i++) {
        const char *name = sel_getName(method_getName(methods[i]));
        if (strstr(name, substr)) {
            printf("    - %s\n      type: %s\n", name, method_getTypeEncoding(methods[i]) ?: "?");
        }
    }
    free(methods);

    // Class methods
    Class meta = object_getClass((id)cls);
    methods = class_copyMethodList(meta, &count);
    for (unsigned int i = 0; i < count; i++) {
        const char *name = sel_getName(method_getName(methods[i]));
        if (strstr(name, substr)) {
            printf("    + %s\n      type: %s\n", name, method_getTypeEncoding(methods[i]) ?: "?");
        }
    }
    free(methods);
    printf("\n");
}

// ── Utility: dump ALL methods of a class ──
static void dump_all_methods(const char *className) {
    Class cls = NSClassFromString([NSString stringWithUTF8String:className]);
    if (!cls) { printf("  [%s] NOT FOUND\n\n", className); return; }
    printf("  [%s] (super: %s)\n", className, class_getName(class_getSuperclass(cls)));

    unsigned int count = 0;
    Method *methods = class_copyMethodList(cls, &count);
    for (unsigned int i = 0; i < count; i++) {
        SEL sel = method_getName(methods[i]);
        printf("    - %s\n      %s\n", sel_getName(sel), method_getTypeEncoding(methods[i]) ?: "?");
    }
    free(methods);

    Class meta = object_getClass((id)cls);
    methods = class_copyMethodList(meta, &count);
    for (unsigned int i = 0; i < count; i++) {
        SEL sel = method_getName(methods[i]);
        printf("    + %s\n      %s\n", sel_getName(sel), method_getTypeEncoding(methods[i]) ?: "?");
    }
    free(methods);

    unsigned int ic = 0;
    Ivar *ivars = class_copyIvarList(cls, &ic);
    for (unsigned int i = 0; i < ic; i++)
        printf("    @ %s (%s)\n", ivar_getName(ivars[i]), ivar_getTypeEncoding(ivars[i]) ?: "?");
    free(ivars);
    printf("\n");
}

// ── Utility: describe any ObjC object with deep inspection ──
static void deep_inspect(id obj, const char *label, int depth) {
    if (!obj) { printf("%*s%s: nil\n", depth*2, "", label); return; }

    const char *cn = class_getName(object_getClass(obj));
    printf("%*s%s: <%s> ", depth*2, "", label, cn);

    if ([obj isKindOfClass:[NSData class]]) {
        NSData *data = (NSData *)obj;
        printf("(%lu bytes)", (unsigned long)[data length]);
        if ([data length] > 0 && [data length] <= 256) {
            printf(" hex: ");
            const uint8_t *bytes = [data bytes];
            for (NSUInteger i = 0; i < MIN([data length], 64); i++)
                printf("%02x", bytes[i]);
            if ([data length] > 64) printf("...");
        }
        printf("\n");

        // Try to parse as analytics buffer
        if ([data length] > 64) {
            Class analyticsCls = NSClassFromString(@"_ANECompilerAnalytics");
            if (analyticsCls) {
                @try {
                    id analytics = ((id(*)(Class,SEL,id))objc_msgSend)(analyticsCls, @selector(objectWithBuffer:), data);
                    if (analytics) {
                        printf("%*s  >>> ANALYTICS BUFFER FOUND! <<<\n", depth*2, "");
                        BOOL pop = ((BOOL(*)(id,SEL))objc_msgSend)(analytics, @selector(populateAnalytics));
                        printf("%*s  populateAnalytics: %s\n", depth*2, "", pop ? "YES" : "NO");
                        id procs = ((id(*)(id,SEL))objc_msgSend)(analytics, @selector(procedureAnalytics));
                        if (procs && [procs isKindOfClass:[NSArray class]]) {
                            printf("%*s  procedures: %lu\n", depth*2, "", (unsigned long)[(NSArray*)procs count]);
                            for (id proc in (NSArray*)procs) {
                                id ident = ((id(*)(id,SEL))objc_msgSend)(proc, @selector(identifier));
                                printf("%*s    procedure: %s\n", depth*2, "", ident ? [[ident description] UTF8String] : "?");
                                id groups = ((id(*)(id,SEL))objc_msgSend)(proc, @selector(groupInfo));
                                if (groups && [groups isKindOfClass:[NSArray class]]) {
                                    for (id group in (NSArray*)groups) {
                                        id layers = ((id(*)(id,SEL))objc_msgSend)(group, @selector(layerInfo));
                                        if (layers && [layers isKindOfClass:[NSArray class]]) {
                                            for (id layer in (NSArray*)layers) {
                                                id ln = ((id(*)(id,SEL))objc_msgSend)(layer, @selector(layerName));
                                                float w = ((float(*)(id,SEL))objc_msgSend)(layer, @selector(weight));
                                                printf("%*s      layer: %s  weight: %.6f\n", depth*2, "",
                                                    ln ? [[ln description] UTF8String] : "?", w);
                                            }
                                        }
                                    }
                                }
                                id metrics = ((id(*)(id,SEL))objc_msgSend)(proc, @selector(procedureMetrics));
                                if (metrics) printf("%*s    metrics: %s\n", depth*2, "", [[metrics description] UTF8String]);
                            }
                        }
                    }
                } @catch (NSException *ex) {}
            }
        }
    } else if ([obj isKindOfClass:[NSDictionary class]]) {
        NSDictionary *dict = (NSDictionary *)obj;
        printf("(%lu keys)\n", (unsigned long)[dict count]);
        for (id key in dict) {
            char sublabel[256];
            snprintf(sublabel, sizeof(sublabel), "[%s]", [[key description] UTF8String]);
            deep_inspect(dict[key], sublabel, depth+1);
        }
    } else if ([obj isKindOfClass:[NSArray class]]) {
        NSArray *arr = (NSArray *)obj;
        printf("(%lu items)\n", (unsigned long)[arr count]);
        for (NSUInteger i = 0; i < MIN([arr count], 20); i++) {
            char sublabel[64];
            snprintf(sublabel, sizeof(sublabel), "[%lu]", (unsigned long)i);
            deep_inspect(arr[i], sublabel, depth+1);
        }
        if ([arr count] > 20) printf("%*s  ... and %lu more\n", depth*2, "", (unsigned long)([arr count]-20));
    } else if ([obj isKindOfClass:[NSNumber class]]) {
        printf("%s\n", [[obj description] UTF8String]);
    } else if ([obj isKindOfClass:[NSString class]]) {
        printf("\"%s\"\n", [(NSString*)obj UTF8String]);
    } else {
        printf("%s\n", [[obj description] UTF8String]);
        // Try to dump ivars
        unsigned int ic = 0;
        Ivar *ivars = class_copyIvarList(object_getClass(obj), &ic);
        for (unsigned int i = 0; i < ic; i++) {
            const char *iname = ivar_getName(ivars[i]);
            const char *itype = ivar_getTypeEncoding(ivars[i]);
            if (itype && itype[0] == '@') {
                @try {
                    id val = object_getIvar(obj, ivars[i]);
                    char sublabel[256];
                    snprintf(sublabel, sizeof(sublabel), ".%s", iname);
                    deep_inspect(val, sublabel, depth+1);
                } @catch (NSException *ex) {}
            } else {
                printf("%*s  .%s (%s)\n", depth*2, "", iname, itype ?: "?");
            }
        }
        free(ivars);
    }
}

// ════════════════════════════════════════════════════════════════════════════
// METHOD SWIZZLING — Interception Layer 1: _ANEDaemonConnection
// ════════════════════════════════════════════════════════════════════════════

// We'll use a dynamic subclass approach to intercept messages
// This is safer than method_exchangeImplementations on system classes

// Store original IMPs
static IMP g_origCompileDaemon = NULL;
// static IMP g_origCompileClient = NULL;  // reserved for future use
// static IMP g_origCompileVirtual = NULL; // reserved for future use

// ── Discovered reply block signature ──
// From block descriptor: v36@?0B8@"NSDictionary"12@"NSString"20@"NSError"28
// Meaning: ^(BOOL success, NSDictionary *result, NSString *info, NSError *error)
typedef void (^DaemonCompileReply)(BOOL success, NSDictionary *result, NSString *info, NSError *error);

// ── Replacement for _ANEDaemonConnection compile method ──
static void swizzled_daemon_compile(id self, SEL _cmd, id model, id sandbox, id options,
                                      unsigned int qos, DaemonCompileReply reply) {
    printf("\n  +------ INTERCEPTED _ANEDaemonConnection.compileModel ------+\n");
    printf("  | options keys: %s\n", options ? [[[options allKeys] description] UTF8String] : "nil");
    printf("  | qos: %u\n", qos);

    // Create a wrapper reply block that intercepts the daemon's response
    DaemonCompileReply wrappedReply = ^(BOOL success, NSDictionary *result, NSString *info, NSError *error) {
        printf("\n  +============= DAEMON COMPILE REPLY =============+\n");
        printf("  | success: %s\n", success ? "YES" : "NO");
        printf("  | info: %s\n", info ? [info UTF8String] : "nil");
        printf("  | error: %s\n", error ? [[error description] UTF8String] : "nil");

        if (result) {
            printf("  | result: <%s> (%lu keys)\n",
                class_getName(object_getClass(result)), (unsigned long)[result count]);
            // Deep inspect every key in the result dictionary
            for (id key in result) {
                id val = result[key];
                printf("  |   [%s]: <%s>", [[key description] UTF8String],
                    class_getName(object_getClass(val)));
                if ([val isKindOfClass:[NSData class]]) {
                    NSData *data = (NSData *)val;
                    printf(" (%lu bytes)", (unsigned long)[data length]);
                    // Check if this could be an analytics buffer
                    if ([data length] > 32) {
                        printf("\n  |     hex(first 64): ");
                        const uint8_t *bytes = [data bytes];
                        for (NSUInteger i = 0; i < MIN([data length], 64); i++)
                            printf("%02x", bytes[i]);
                        if ([data length] > 64) printf("...");

                        // Try parsing as analytics
                        Class analyticsCls = NSClassFromString(@"_ANECompilerAnalytics");
                        if (analyticsCls) {
                            @try {
                                id analytics = ((id(*)(Class,SEL,id))objc_msgSend)(
                                    analyticsCls, @selector(objectWithBuffer:), data);
                                if (analytics) {
                                    BOOL pop = ((BOOL(*)(id,SEL))objc_msgSend)(analytics, @selector(populateAnalytics));
                                    printf("\n  |     >>> ANALYTICS PARSE: populated=%s", pop ? "YES" : "NO");
                                    id procs = ((id(*)(id,SEL))objc_msgSend)(analytics, @selector(procedureAnalytics));
                                    if (procs && [procs isKindOfClass:[NSArray class]] && [(NSArray*)procs count] > 0) {
                                        printf(" procedures=%lu", (unsigned long)[(NSArray*)procs count]);
                                        for (id proc in (NSArray*)procs) {
                                            id ident = ((id(*)(id,SEL))objc_msgSend)(proc, @selector(identifier));
                                            printf("\n  |       proc: %s", ident ? [[ident description] UTF8String] : "?");
                                            id groups = ((id(*)(id,SEL))objc_msgSend)(proc, @selector(groupInfo));
                                            if (groups && [groups isKindOfClass:[NSArray class]]) {
                                                for (id group in (NSArray*)groups) {
                                                    id gid = ((id(*)(id,SEL))objc_msgSend)(group, @selector(groupID));
                                                    printf("\n  |         group: %s", gid ? [[gid description] UTF8String] : "?");
                                                    id layers = ((id(*)(id,SEL))objc_msgSend)(group, @selector(layerInfo));
                                                    if (layers && [layers isKindOfClass:[NSArray class]]) {
                                                        for (id layer in (NSArray*)layers) {
                                                            id ln = ((id(*)(id,SEL))objc_msgSend)(layer, @selector(layerName));
                                                            float w = ((float(*)(id,SEL))objc_msgSend)(layer, @selector(weight));
                                                            printf("\n  |           layer: %-40s  weight: %.6f",
                                                                ln ? [[ln description] UTF8String] : "?", w);
                                                        }
                                                    }
                                                    id tasks = ((id(*)(id,SEL))objc_msgSend)(group, @selector(taskInfo));
                                                    if (tasks && [tasks isKindOfClass:[NSArray class]]) {
                                                        for (id task in (NSArray*)tasks) {
                                                            id tm = ((id(*)(id,SEL))objc_msgSend)(task, @selector(metrics));
                                                            printf("\n  |           task metrics: %s",
                                                                tm ? [[tm description] UTF8String] : "nil");
                                                        }
                                                    }
                                                }
                                            }
                                            id metrics = ((id(*)(id,SEL))objc_msgSend)(proc, @selector(procedureMetrics));
                                            if (metrics) printf("\n  |       metrics: %s", [[metrics description] UTF8String]);
                                        }
                                    }
                                }
                            } @catch (NSException *ex) {
                                printf("\n  |     analytics parse EXC: %s", [[ex reason] UTF8String]);
                            }
                        }
                    }
                } else if ([val isKindOfClass:[NSDictionary class]]) {
                    printf(" (%lu keys)", (unsigned long)[(NSDictionary*)val count]);
                    for (id subkey in (NSDictionary*)val) {
                        id subval = [(NSDictionary*)val objectForKey:subkey];
                        printf("\n  |     [%s]: <%s>", [[subkey description] UTF8String],
                            class_getName(object_getClass(subval)));
                        if ([subval isKindOfClass:[NSData class]])
                            printf(" (%lu bytes)", (unsigned long)[(NSData*)subval length]);
                        else if ([subval isKindOfClass:[NSString class]])
                            printf(" \"%s\"", [(NSString*)subval UTF8String]);
                        else if ([subval isKindOfClass:[NSNumber class]])
                            printf(" %s", [[subval description] UTF8String]);
                    }
                } else if ([val isKindOfClass:[NSArray class]]) {
                    NSArray *arr = (NSArray *)val;
                    printf(" (%lu items)", (unsigned long)[arr count]);
                    for (NSUInteger idx = 0; idx < MIN([arr count], 5); idx++) {
                        id item = arr[idx];
                        printf("\n  |     [%lu]: <%s>", (unsigned long)idx,
                            class_getName(object_getClass(item)));
                        if ([item isKindOfClass:[NSData class]])
                            printf(" (%lu bytes)", (unsigned long)[(NSData*)item length]);
                        else if ([item isKindOfClass:[NSDictionary class]]) {
                            NSDictionary *d = (NSDictionary *)item;
                            printf(" (%lu keys)", (unsigned long)[d count]);
                            for (id sk in d) {
                                id sv = d[sk];
                                printf("\n  |       .%s: <%s>", [[sk description] UTF8String],
                                    class_getName(object_getClass(sv)));
                                if ([sv isKindOfClass:[NSData class]])
                                    printf(" (%lu bytes)", (unsigned long)[(NSData*)sv length]);
                                else if ([sv isKindOfClass:[NSNumber class]])
                                    printf(" %s", [[sv description] UTF8String]);
                                else if ([sv isKindOfClass:[NSString class]])
                                    printf(" \"%s\"", [(NSString*)sv UTF8String]);
                                else if ([sv isKindOfClass:[NSArray class]]) {
                                    NSArray *sa = (NSArray *)sv;
                                    printf(" (%lu items)", (unsigned long)[sa count]);
                                    for (NSUInteger si = 0; si < MIN([sa count], 3); si++) {
                                        id sitem = sa[si];
                                        printf("\n  |         [%lu]: <%s>", (unsigned long)si,
                                            class_getName(object_getClass(sitem)));
                                        if ([sitem isKindOfClass:[NSDictionary class]]) {
                                            for (id ssk in (NSDictionary*)sitem)
                                                printf(" %s=%s", [[ssk description] UTF8String],
                                                    [[[(NSDictionary*)sitem objectForKey:ssk] description] UTF8String]);
                                        }
                                    }
                                }
                                else if ([sv isKindOfClass:[NSDictionary class]]) {
                                    printf(" (%lu keys)", (unsigned long)[(NSDictionary*)sv count]);
                                }
                            }
                        }
                    }
                } else if ([val isKindOfClass:[NSString class]]) {
                    printf(" \"%s\"", [(NSString*)val UTF8String]);
                } else if ([val isKindOfClass:[NSNumber class]]) {
                    printf(" %s", [[val description] UTF8String]);
                } else {
                    printf(" %s", [[val description] UTF8String]);
                }
                printf("\n");
            }
        } else {
            printf("  | result: nil\n");
        }

        printf("  +===============================================+\n\n");

        // Store for later inspection
        if (result) [g_capturedReplies addObject:result];

        // Call the original reply
        if (reply) reply(success, result, info, error);
    };

    // Call original with wrapped reply
    if (g_origCompileDaemon) {
        ((void(*)(id,SEL,id,id,id,unsigned int,DaemonCompileReply))g_origCompileDaemon)(
            self, _cmd, model, sandbox, options, qos, wrappedReply);
    }
}

// ════════════════════════════════════════════════════════════════════════════
// METHOD SWIZZLING — Interception Layer 2: _ANEDaemonConnection.loadModel
// ════════════════════════════════════════════════════════════════════════════
// loadModel has the same type encoding as compileModel — might also return analytics
static IMP g_origLoadDaemon = NULL;

// Load reply block signature (discovered):
// v56@?0B8@"NSDictionary"12Q20Q28c36@"NSString"40@"NSError"48
// = ^(BOOL success, NSDictionary *result, uint64_t programHandle, uint64_t intermediateHandle, char queueDepth, NSString *info, NSError *error)
typedef void (^DaemonLoadReply)(BOOL success, NSDictionary *result, uint64_t programHandle,
                                 uint64_t intermediateHandle, char queueDepth, NSString *info, NSError *error);

static void swizzled_daemon_load(id self, SEL _cmd, id model, id sandbox, id options,
                                   unsigned int qos, DaemonLoadReply reply) {
    printf("\n  +------ INTERCEPTED _ANEDaemonConnection.loadModel ------+\n");
    printf("  | options: %s\n", options ? [[[options allKeys] description] UTF8String] : "nil");

    DaemonLoadReply wrappedReply = ^(BOOL success, NSDictionary *result, uint64_t programHandle,
                                      uint64_t intermediateHandle, char queueDepth, NSString *info, NSError *error) {
        printf("\n  +============= DAEMON LOAD REPLY =============+\n");
        printf("  | success: %s\n", success ? "YES" : "NO");
        printf("  | programHandle: %llu (0x%llx)\n", programHandle, programHandle);
        printf("  | intermediateHandle: %llu (0x%llx)\n", intermediateHandle, intermediateHandle);
        printf("  | queueDepth: %d\n", (int)queueDepth);
        printf("  | info: %s\n", info ? [info UTF8String] : "nil");
        printf("  | error: %s\n", error ? [[error description] UTF8String] : "nil");

        if (result) {
            printf("  | result: <%s> (%lu keys)\n",
                class_getName(object_getClass(result)), (unsigned long)[result count]);
            for (id key in result) {
                id val = result[key];
                printf("  |   [%s]: <%s>", [[key description] UTF8String],
                    class_getName(object_getClass(val)));
                if ([val isKindOfClass:[NSData class]]) {
                    NSData *data = (NSData *)val;
                    printf(" (%lu bytes)", (unsigned long)[data length]);
                    // Try analytics parse on any NSData > 32 bytes
                    if ([data length] > 32) {
                        Class ac = NSClassFromString(@"_ANECompilerAnalytics");
                        if (ac) {
                            @try {
                                id an = ((id(*)(Class,SEL,id))objc_msgSend)(ac, @selector(objectWithBuffer:), data);
                                if (an) {
                                    BOOL pop = ((BOOL(*)(id,SEL))objc_msgSend)(an, @selector(populateAnalytics));
                                    id procs = ((id(*)(id,SEL))objc_msgSend)(an, @selector(procedureAnalytics));
                                    printf("\n  |     ANALYTICS: populated=%s procs=%lu",
                                        pop ? "YES" : "NO",
                                        procs ? (unsigned long)[(NSArray*)procs count] : 0);
                                }
                            } @catch (NSException *ex) {}
                        }
                    }
                } else if ([val isKindOfClass:[NSNumber class]]) {
                    printf(" %s", [[val description] UTF8String]);
                } else if ([val isKindOfClass:[NSString class]]) {
                    printf(" \"%s\"", [(NSString*)val UTF8String]);
                } else if ([val isKindOfClass:[NSDictionary class]]) {
                    NSDictionary *d = (NSDictionary *)val;
                    printf(" (%lu keys)", (unsigned long)[d count]);
                    for (id sk in d) {
                        id sv = d[sk];
                        printf("\n  |     .%s: <%s>", [[sk description] UTF8String],
                            class_getName(object_getClass(sv)));
                        if ([sv isKindOfClass:[NSData class]])
                            printf(" (%lu bytes)", (unsigned long)[(NSData*)sv length]);
                        else if ([sv isKindOfClass:[NSNumber class]])
                            printf(" %s", [[sv description] UTF8String]);
                    }
                } else if ([val isKindOfClass:[NSArray class]]) {
                    NSArray *arr = (NSArray *)val;
                    printf(" (%lu items)", (unsigned long)[arr count]);
                    for (NSUInteger idx = 0; idx < MIN([arr count], 5); idx++) {
                        id item = arr[idx];
                        printf("\n  |     [%lu]: <%s>", (unsigned long)idx,
                            class_getName(object_getClass(item)));
                        if ([item isKindOfClass:[NSDictionary class]]) {
                            NSDictionary *d = (NSDictionary *)item;
                            printf(" (%lu keys)", (unsigned long)[d count]);
                            for (id sk in d) {
                                id sv = d[sk];
                                printf("\n  |       .%s: <%s>", [[sk description] UTF8String],
                                    class_getName(object_getClass(sv)));
                                if ([sv isKindOfClass:[NSData class]])
                                    printf(" (%lu bytes)", (unsigned long)[(NSData*)sv length]);
                                else if ([sv isKindOfClass:[NSNumber class]])
                                    printf(" %s", [[sv description] UTF8String]);
                                else if ([sv isKindOfClass:[NSString class]])
                                    printf(" \"%s\"", [(NSString*)sv UTF8String]);
                            }
                        }
                    }
                }
                printf("\n");
            }
            [g_capturedReplies addObject:result];
        } else {
            printf("  | result: nil\n");
        }
        printf("  +===============================================+\n\n");

        if (reply) reply(success, result, programHandle, intermediateHandle, queueDepth, info, error);
    };

    if (g_origLoadDaemon) {
        ((void(*)(id,SEL,id,id,id,unsigned int,DaemonLoadReply))g_origLoadDaemon)(
            self, _cmd, model, sandbox, options, qos, wrappedReply);
    }
}

// ════════════════════════════════════════════════════════════════════════════
// _ANEClient.compileModel: synchronous wrapper (no reply block)
// ════════════════════════════════════════════════════════════════════════════

// ════════════════════════════════════════════════════════════════════════════
// METHOD SWIZZLING — Interception Layer 3: _ANEInMemoryModel
// ════════════════════════════════════════════════════════════════════════════

// Replacement for _ANEInMemoryModel.compileWithQoS:options:error:
static IMP g_origCompileIMM = NULL;

static BOOL swizzled_imm_compile(id self, SEL _cmd, unsigned int qos, id options, NSError **error) {
    printf("\n  ╔══ INTERCEPTED _ANEInMemoryModel.compileWithQoS ══╗\n");
    printf("  ║ self: %s\n", [[self description] UTF8String]);
    printf("  ║ qos: %u\n", qos);
    printf("  ║ options: %s\n", options ? [[options description] UTF8String] : "nil");

    // Try adding analytics-triggering options
    NSMutableDictionary *enhancedOpts = [NSMutableDictionary dictionaryWithDictionary:options ?: @{}];

    // Try various option keys that might enable analytics
    [enhancedOpts setObject:@YES forKey:@"ANECompilerAnalytics"];
    [enhancedOpts setObject:@YES forKey:@"compilerAnalytics"];
    [enhancedOpts setObject:@YES forKey:@"enableAnalytics"];
    [enhancedOpts setObject:@(0xFFFFFFFF) forKey:@"kANEFPerformanceStatsMask"];
    [enhancedOpts setObject:@YES forKey:@"enablePerformanceStats"];
    [enhancedOpts setObject:@YES forKey:@"ANEPerformanceStats"];

    printf("  ║ enhanced options: %s\n", [[enhancedOpts description] UTF8String]);
    printf("  ╚════════════════════════════════════════════════════╝\n");

    // Call original with enhanced options
    BOOL result;
    if (g_origCompileIMM) {
        result = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))g_origCompileIMM)(self, _cmd, qos, enhancedOpts, error);
    } else {
        result = NO;
    }

    printf("  compile result: %s\n", result ? "YES" : "NO");
    if (error && *error) printf("  compile error: %s\n", [[*error description] UTF8String]);

    // After compile, inspect the model for analytics data
    if (result) {
        printf("\n  ── Post-compile inspection ──\n");

        // Check all ivars of self for NSData that could be analytics
        unsigned int ic = 0;
        Ivar *ivars = class_copyIvarList(object_getClass(self), &ic);
        for (unsigned int i = 0; i < ic; i++) {
            const char *iname = ivar_getName(ivars[i]);
            const char *itype = ivar_getTypeEncoding(ivars[i]);
            if (itype && itype[0] == '@') {
                @try {
                    id val = object_getIvar(self, ivars[i]);
                    if (val) {
                        printf("    ivar .%s = <%s>\n", iname, class_getName(object_getClass(val)));
                        if ([val isKindOfClass:[NSData class]]) {
                            printf("      DATA (%lu bytes) — potential analytics!\n", (unsigned long)[(NSData*)val length]);
                            deep_inspect(val, iname, 3);
                        }
                    }
                } @catch (NSException *ex) {}
            }
        }
        free(ivars);

        // Check _ANEModel inside
        Ivar modelIvar = class_getInstanceVariable(object_getClass(self), "_model");
        if (modelIvar) {
            id aneModel = object_getIvar(self, modelIvar);
            if (aneModel) {
                printf("    _model = <%s>\n", class_getName(object_getClass(aneModel)));
                unsigned int mic = 0;
                Ivar *mivars = class_copyIvarList(object_getClass(aneModel), &mic);
                for (unsigned int i = 0; i < mic; i++) {
                    const char *iname = ivar_getName(mivars[i]);
                    const char *itype = ivar_getTypeEncoding(mivars[i]);
                    if (itype && itype[0] == '@') {
                        @try {
                            id val = object_getIvar(aneModel, mivars[i]);
                            if (val) {
                                printf("      _model.%s = <%s>\n", iname, class_getName(object_getClass(val)));
                                if ([val isKindOfClass:[NSData class]]) {
                                    printf("        DATA (%lu bytes)\n", (unsigned long)[(NSData*)val length]);
                                    deep_inspect(val, iname, 4);
                                }
                            }
                        } @catch (NSException *ex) {}
                    }
                }
                free(mivars);
            }
        }

        // Check _sharedConnection -> _ANEClient for analytics
        Ivar connIvar = class_getInstanceVariable(object_getClass(self), "_sharedConnection");
        if (connIvar) {
            id client = object_getIvar(self, connIvar);
            if (client) {
                printf("    _sharedConnection = <%s>\n", class_getName(object_getClass(client)));
                unsigned int cic = 0;
                Ivar *civars = class_copyIvarList(object_getClass(client), &cic);
                for (unsigned int i = 0; i < cic; i++) {
                    const char *iname = ivar_getName(civars[i]);
                    const char *itype = ivar_getTypeEncoding(civars[i]);
                    printf("      client.%s (%s)\n", iname, itype ?: "?");
                    if (itype && itype[0] == '@') {
                        @try {
                            id val = object_getIvar(client, civars[i]);
                            if (val) {
                                printf("        = <%s>", class_getName(object_getClass(val)));
                                if ([val isKindOfClass:[NSData class]])
                                    printf(" (%lu bytes)", (unsigned long)[(NSData*)val length]);
                                printf("\n");
                            }
                        } @catch (NSException *ex) {}
                    }
                }
                free(civars);
            }
        }
    }

    return result;
}

// ════════════════════════════════════════════════════════════════════════════
// APPROACH 2: objc_msgSend logging via forwarding
// ════════════════════════════════════════════════════════════════════════════

// Instead of swizzling, we can also use NSProxy to intercept ALL messages
// to a _ANEDaemonConnection. But swizzling is more targeted.

// ════════════════════════════════════════════════════════════════════════════
// APPROACH 3: Inspect the XPC connection directly
// ════════════════════════════════════════════════════════════════════════════

static void inspect_xpc_connection(id client) {
    printf("\n  ── XPC Connection Inspection ──\n");

    // Walk through _ANEClient ivars to find NSXPCConnection
    unsigned int ic = 0;
    Class cls = object_getClass(client);
    while (cls && cls != [NSObject class]) {
        Ivar *ivars = class_copyIvarList(cls, &ic);
        for (unsigned int i = 0; i < ic; i++) {
            const char *iname = ivar_getName(ivars[i]);
            const char *itype = ivar_getTypeEncoding(ivars[i]);
            printf("    %s.%s (%s)", class_getName(cls), iname, itype ?: "?");
            if (itype && itype[0] == '@') {
                @try {
                    id val = object_getIvar(client, ivars[i]);
                    if (val) {
                        printf(" = <%s>", class_getName(object_getClass(val)));
                        if ([val isKindOfClass:[NSData class]])
                            printf(" %lu bytes", (unsigned long)[(NSData*)val length]);
                        else if ([val isKindOfClass:[NSString class]])
                            printf(" \"%s\"", [(NSString*)val UTF8String]);
                        else if ([val isKindOfClass:[NSNumber class]])
                            printf(" %s", [[val description] UTF8String]);

                        // If it's an NSXPCConnection, dump its interface
                        if ([val isKindOfClass:[NSXPCConnection class]]) {
                            NSXPCConnection *xpc = (NSXPCConnection *)val;
                            printf("\n      endpoint: %s", [[[xpc endpoint] description] UTF8String]);
                            id ri = [xpc remoteObjectInterface];
                            if (ri) {
                                printf("\n      remoteInterface: <%s>", class_getName(object_getClass(ri)));
                                // NSXPCInterface has protocol property
                                @try {
                                    Protocol *proto = ((Protocol*(*)(id,SEL))objc_msgSend)(ri, @selector(protocol));
                                    if (proto) {
                                        printf("\n      protocol: %s", protocol_getName(proto));
                                        // Dump protocol methods
                                        unsigned int pmc = 0;
                                        struct objc_method_description *descs;
                                        descs = protocol_copyMethodDescriptionList(proto, YES, YES, &pmc);
                                        printf("\n      required instance methods:");
                                        for (unsigned int j = 0; j < pmc; j++)
                                            printf("\n        %s  (%s)", sel_getName(descs[j].name), descs[j].types ?: "?");
                                        free(descs);
                                        descs = protocol_copyMethodDescriptionList(proto, NO, YES, &pmc);
                                        printf("\n      optional instance methods:");
                                        for (unsigned int j = 0; j < pmc; j++)
                                            printf("\n        %s  (%s)", sel_getName(descs[j].name), descs[j].types ?: "?");
                                        free(descs);
                                    }
                                } @catch (NSException *ex) {}
                            }
                        }

                        // Recurse into _ANEDaemonConnection
                        const char *vcn = class_getName(object_getClass(val));
                        if (strstr(vcn, "Daemon") || strstr(vcn, "Connection") || strstr(vcn, "Virtual")) {
                            printf("\n      --- recurse into %s ---\n", vcn);
                            unsigned int ric = 0;
                            Ivar *rivars = class_copyIvarList(object_getClass(val), &ric);
                            for (unsigned int j = 0; j < ric; j++) {
                                const char *rn = ivar_getName(rivars[j]);
                                const char *rt = ivar_getTypeEncoding(rivars[j]);
                                printf("        .%s (%s)", rn, rt ?: "?");
                                if (rt && rt[0] == '@') {
                                    @try {
                                        id rv = object_getIvar(val, rivars[j]);
                                        if (rv) printf(" = <%s> %s",
                                            class_getName(object_getClass(rv)),
                                            [[rv description] UTF8String]);
                                    } @catch (NSException *ex) {}
                                }
                                printf("\n");
                            }
                            free(rivars);
                        }
                    }
                } @catch (NSException *ex) {
                    printf(" EXC");
                }
            }
            printf("\n");
        }
        free(ivars);
        cls = class_getSuperclass(cls);
    }
}

// ════════════════════════════════════════════════════════════════════════════
// APPROACH 4: Forge a compile call with reply block we control
// ════════════════════════════════════════════════════════════════════════════

typedef void (^CompileReply1)(BOOL success, NSError *error);
typedef void (^CompileReply2)(BOOL success, NSError *error, id result);
typedef void (^CompileReply3)(id result, NSError *error);
typedef void (^CompileReply4)(BOOL success, NSError *error, NSData *data, id extra);

// ════════════════════════════════════════════════════════════════════════════
// Compile a test kernel (same as test_compiler_analytics.m)
// ════════════════════════════════════════════════════════════════════════════

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

    return mdl;
}

// ════════════════════════════════════════════════════════════════════════════
// MAIN
// ════════════════════════════════════════════════════════════════════════════

int main() {
    @autoreleasepool {
        setbuf(stdout, NULL);
        g_capturedReplies = [NSMutableArray array];
        g_capturedBlocks = [NSMutableArray array];

        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
        dlopen("/System/Library/PrivateFrameworks/Espresso.framework/Espresso", RTLD_NOW);

        printf("\n  ██████ ANE XPC ANALYTICS INTERCEPTION ██████\n\n");

        // ════════════════════════════════════════════════════════════════
        // Section 1: Dump compile-related methods across ALL relevant classes
        // ════════════════════════════════════════════════════════════════
        printf("  ══ Section 1: Compile-related methods ══\n\n");

        const char *targetClasses[] = {
            "_ANEDaemonConnection",
            "_ANEClient",
            "_ANEVirtualClient",
            "_ANEInMemoryModel",
            "_ANEModel",
            "_ANEProgramForEvaluation",
            NULL
        };

        for (int i = 0; targetClasses[i]; i++) {
            dump_methods_matching(targetClasses[i], "ompile");
        }

        // Also dump analytics/reply/result related
        printf("  ── Analytics/reply/result methods ──\n\n");
        for (int i = 0; targetClasses[i]; i++) {
            dump_methods_matching(targetClasses[i], "nalytics");
            dump_methods_matching(targetClasses[i], "eply");
            dump_methods_matching(targetClasses[i], "esult");
        }

        // ════════════════════════════════════════════════════════════════
        // Section 2: Full dump of _ANEDaemonConnection
        // ════════════════════════════════════════════════════════════════
        printf("  ══ Section 2: Full _ANEDaemonConnection dump ══\n\n");
        dump_all_methods("_ANEDaemonConnection");

        printf("  ── Full _ANEClient dump ──\n\n");
        dump_all_methods("_ANEClient");

        printf("  ── Full _ANEVirtualClient dump ──\n\n");
        dump_all_methods("_ANEVirtualClient");

        // ════════════════════════════════════════════════════════════════
        // Section 3: Discover the compile reply block signature
        // ════════════════════════════════════════════════════════════════
        printf("  ══ Section 3: Compile method type encodings ══\n\n");
        {
            const char *classes[] = {"_ANEDaemonConnection", "_ANEClient", "_ANEVirtualClient", NULL};
            for (int c = 0; classes[c]; c++) {
                Class cls = NSClassFromString([NSString stringWithUTF8String:classes[c]]);
                if (!cls) continue;
                unsigned int mc = 0;
                Method *methods = class_copyMethodList(cls, &mc);
                for (unsigned int m = 0; m < mc; m++) {
                    const char *name = sel_getName(method_getName(methods[m]));
                    if (strstr(name, "ompile") || strstr(name, "eply")) {
                        printf("  [%s] %s\n", classes[c], name);
                        printf("    encoding: %s\n", method_getTypeEncoding(methods[m]) ?: "?");

                        // Decode the encoding character by character
                        const char *enc = method_getTypeEncoding(methods[m]);
                        if (enc) {
                            printf("    decoded args:\n");
                            unsigned int nargs = method_getNumberOfArguments(methods[m]);
                            for (unsigned int a = 0; a < nargs; a++) {
                                char argType[256] = {0};
                                method_getArgumentType(methods[m], a, argType, sizeof(argType));
                                printf("      arg[%u]: %s", a, argType);
                                if (strcmp(argType, "@") == 0) printf(" (id)");
                                else if (strcmp(argType, ":") == 0) printf(" (SEL)");
                                else if (strcmp(argType, "I") == 0) printf(" (unsigned int)");
                                else if (strcmp(argType, "Q") == 0) printf(" (uint64)");
                                else if (strcmp(argType, "B") == 0) printf(" (BOOL)");
                                else if (strcmp(argType, "@?") == 0) printf(" (BLOCK)");
                                else if (strcmp(argType, "^@") == 0) printf(" (id*)");
                                else if (strcmp(argType, "v") == 0) printf(" (void)");
                                printf("\n");
                            }
                        }
                        printf("\n");
                    }
                }
                free(methods);
            }
        }

        // ════════════════════════════════════════════════════════════════
        // Section 4: Install swizzle on _ANEInMemoryModel.compileWithQoS:
        // ════════════════════════════════════════════════════════════════
        printf("  ══ Section 4: Swizzling _ANEInMemoryModel.compileWithQoS ══\n\n");
        {
            Class immCls = NSClassFromString(@"_ANEInMemoryModel");
            if (immCls) {
                SEL compileSel = @selector(compileWithQoS:options:error:);
                Method m = class_getInstanceMethod(immCls, compileSel);
                if (m) {
                    g_origCompileIMM = method_getImplementation(m);
                    method_setImplementation(m, (IMP)swizzled_imm_compile);
                    printf("  Swizzled _ANEInMemoryModel.compileWithQoS:options:error:\n");
                    printf("  Original IMP: %p -> New IMP: %p\n\n", g_origCompileIMM, (void*)swizzled_imm_compile);
                    g_swizzleActive = YES;
                } else {
                    printf("  compileWithQoS:options:error: not found on _ANEInMemoryModel\n\n");
                }
            }
        }

        // ════════════════════════════════════════════════════════════════
        // Section 5: Install swizzle on _ANEDaemonConnection.compileModel
        // ════════════════════════════════════════════════════════════════
        printf("  ══ Section 5: Swizzling _ANEDaemonConnection.compileModel ══\n\n");
        {
            Class daemonCls = NSClassFromString(@"_ANEDaemonConnection");
            if (daemonCls) {
                // Find the compile method dynamically
                unsigned int mc = 0;
                Method *methods = class_copyMethodList(daemonCls, &mc);
                Method compileMethod = NULL;
                const char *compileSelName = NULL;

                for (unsigned int i = 0; i < mc; i++) {
                    const char *name = sel_getName(method_getName(methods[i]));
                    if (strstr(name, "compileModel") && strstr(name, "withReply")) {
                        compileMethod = methods[i];
                        compileSelName = name;
                        break;
                    }
                }

                if (compileMethod) {
                    printf("  Found: %s\n", compileSelName);
                    printf("  Encoding: %s\n", method_getTypeEncoding(compileMethod) ?: "?");
                    g_origCompileDaemon = method_getImplementation(compileMethod);
                    method_setImplementation(compileMethod, (IMP)swizzled_daemon_compile);
                    printf("  Swizzled! Original: %p -> New: %p\n\n", g_origCompileDaemon, (void*)swizzled_daemon_compile);
                } else {
                    printf("  No compileModel...withReply method found\n");
                    // List all methods for debugging
                    printf("  All methods:\n");
                    for (unsigned int i = 0; i < mc; i++)
                        printf("    %s\n", sel_getName(method_getName(methods[i])));
                    printf("\n");
                }
                // Also swizzle loadModel
                methods = class_copyMethodList(daemonCls, &mc);
                for (unsigned int i = 0; i < mc; i++) {
                    const char *name = sel_getName(method_getName(methods[i]));
                    if (strstr(name, "loadModel:sandbox") && strstr(name, "withReply")) {
                        g_origLoadDaemon = method_getImplementation(methods[i]);
                        method_setImplementation(methods[i], (IMP)swizzled_daemon_load);
                        printf("  Also swizzled: %s\n\n", name);
                        break;
                    }
                }
                free(methods);
            }
        }

        // ════════════════════════════════════════════════════════════════
        // Section 6: Compile a test kernel and capture everything
        // ════════════════════════════════════════════════════════════════
        printf("  ══ Section 6: Test compile with interception active ══\n\n");
        {
            Class g_D = NSClassFromString(@"_ANEInMemoryModelDescriptor");
            Class g_I = NSClassFromString(@"_ANEInMemoryModel");

            NSData *wdata = nil;
            id mdl = compile_kernel(g_D, g_I, 256, 64, &wdata);

            if (mdl) {
                printf("  Model created, triggering compile...\n\n");

                NSError *e = nil;
                BOOL compOk = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                    mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e);
                printf("\n  Compile result: %s\n", compOk ? "SUCCESS" : "FAILED");
                if (e) printf("  Error: %s\n", [[e description] UTF8String]);

                if (compOk) {
                    // Load and inspect
                    e = nil;
                    BOOL loaded = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                        mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
                    printf("  Load result: %s\n", loaded ? "SUCCESS" : "FAILED");

                    if (loaded) {
                        printf("\n  ── Post-load deep inspection ──\n");

                        // Get _ANEModel
                        Ivar modelIvar = class_getInstanceVariable([mdl class], "_model");
                        id aneModel = modelIvar ? object_getIvar(mdl, modelIvar) : nil;

                        if (aneModel) {
                            printf("  _ANEModel class: %s\n", class_getName(object_getClass(aneModel)));

                            // Dump ALL ivars of _ANEModel
                            unsigned int ic = 0;
                            Ivar *ivars = class_copyIvarList(object_getClass(aneModel), &ic);
                            printf("  _ANEModel ivars (%u):\n", ic);
                            for (unsigned int i = 0; i < ic; i++) {
                                const char *iname = ivar_getName(ivars[i]);
                                const char *itype = ivar_getTypeEncoding(ivars[i]);
                                printf("    .%s (%s)", iname, itype ?: "?");
                                if (itype && itype[0] == '@') {
                                    @try {
                                        id val = object_getIvar(aneModel, ivars[i]);
                                        if (val) {
                                            printf(" = <%s>", class_getName(object_getClass(val)));
                                            if ([val isKindOfClass:[NSData class]])
                                                printf(" %lu bytes", (unsigned long)[(NSData*)val length]);
                                            else if ([val isKindOfClass:[NSString class]])
                                                printf(" \"%s\"", [(NSString*)val UTF8String]);
                                            else if ([val isKindOfClass:[NSNumber class]])
                                                printf(" %s", [[val description] UTF8String]);
                                        } else {
                                            printf(" = nil");
                                        }
                                    } @catch (NSException *ex) { printf(" EXC"); }
                                }
                                printf("\n");
                            }
                            free(ivars);

                            // Check modelAttributes for analytics data
                            @try {
                                NSDictionary *attrs = ((id(*)(id,SEL))objc_msgSend)(aneModel, @selector(modelAttributes));
                                if (attrs) {
                                    printf("\n  modelAttributes keys:\n");
                                    for (id key in attrs) {
                                        id val = attrs[key];
                                        printf("    %s: <%s>", [[key description] UTF8String], class_getName(object_getClass(val)));
                                        if ([val isKindOfClass:[NSData class]])
                                            printf(" %lu bytes", (unsigned long)[(NSData*)val length]);
                                        printf("\n");
                                    }
                                }
                            } @catch (NSException *ex) {}
                        }

                        // Inspect the XPC connection
                        Ivar connIvar = class_getInstanceVariable([mdl class], "_sharedConnection");
                        id client = connIvar ? object_getIvar(mdl, connIvar) : nil;
                        if (client) inspect_xpc_connection(client);

                        // Unload
                        ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
                            mdl, @selector(unloadWithQoS:error:), 21, &e);
                    }
                }
            }
        }

        // ════════════════════════════════════════════════════════════════
        // Section 7: Try direct XPC call with custom reply block
        // ════════════════════════════════════════════════════════════════
        // ════════════════════════════════════════════════════════════════
        // Section 7: Summary of captured replies
        // ════════════════════════════════════════════════════════════════
        printf("\n  ══ Section 7: Captured reply summary ══\n\n");
        printf("  Total captured replies: %lu\n", (unsigned long)[g_capturedReplies count]);
        for (NSUInteger i = 0; i < [g_capturedReplies count]; i++) {
            printf("  Reply [%lu]:\n", (unsigned long)i);
            deep_inspect(g_capturedReplies[i], "reply", 2);
        }

        // ════════════════════════════════════════════════════════════════
        // Section 8: Try environment variables for analytics
        // ════════════════════════════════════════════════════════════════
        printf("\n  ══ Section 8: Environment variable check ══\n\n");
        {
            const char *envVars[] = {
                "COREML_PROFILING",
                "ANE_PROFILING",
                "ANE_COMPILER_ANALYTICS",
                "ESPRESSO_PROFILING",
                "ANE_DEBUG",
                "ANED_DEBUG",
                "ANE_LOG_LEVEL",
                NULL
            };
            for (int i = 0; envVars[i]; i++) {
                const char *val = getenv(envVars[i]);
                printf("  %s = %s\n", envVars[i], val ?: "(not set)");
            }
            printf("\n  Tip: try running with:\n");
            printf("    COREML_PROFILING=1 ANE_COMPILER_ANALYTICS=1 ./test_analytics_xpc\n\n");
        }

        // ════════════════════════════════════════════════════════════════
        // Section 9: Look for analytics in temp directories
        // ════════════════════════════════════════════════════════════════
        printf("  ══ Section 9: Search temp directories for analytics files ══\n\n");
        {
            NSFileManager *fm = [NSFileManager defaultManager];
            NSString *tmpDir = NSTemporaryDirectory();
            NSArray *contents = [fm contentsOfDirectoryAtPath:tmpDir error:nil];
            printf("  Temp dir: %s (%lu items)\n", [tmpDir UTF8String], (unsigned long)[contents count]);
            for (NSString *item in contents) {
                NSString *full = [tmpDir stringByAppendingPathComponent:item];
                BOOL isDir = NO;
                [fm fileExistsAtPath:full isDirectory:&isDir];
                if (isDir) {
                    NSArray *sub = [fm contentsOfDirectoryAtPath:full error:nil];
                    for (NSString *s in sub) {
                        if ([s containsString:@"analytics"] || [s containsString:@"Analytics"] ||
                            [s containsString:@"stats"] || [s containsString:@"profile"]) {
                            printf("    %s/%s\n", [item UTF8String], [s UTF8String]);
                            // Read the file
                            NSString *fp = [full stringByAppendingPathComponent:s];
                            NSData *d = [NSData dataWithContentsOfFile:fp];
                            if (d) deep_inspect(d, [s UTF8String], 3);
                        }
                    }
                }
            }
        }

        // ════════════════════════════════════════════════════════════════
        // Section 10: Scan for any _ANE*Protocol
        // ════════════════════════════════════════════════════════════════
        printf("\n  ══ Section 10: ANE protocols (XPC interfaces) ══\n\n");
        {
            unsigned int pc = 0;
            Protocol * __unsafe_unretained *protos = objc_copyProtocolList(&pc);
            for (unsigned int i = 0; i < pc; i++) {
                const char *pn = protocol_getName(protos[i]);
                if (strstr(pn, "ANE") || strstr(pn, "ane")) {
                    printf("  Protocol: %s\n", pn);

                    unsigned int mc = 0;
                    struct objc_method_description *descs;

                    descs = protocol_copyMethodDescriptionList(protos[i], YES, YES, &mc);
                    if (mc > 0) {
                        printf("    Required instance methods:\n");
                        for (unsigned int j = 0; j < mc; j++)
                            printf("      %s  (%s)\n", sel_getName(descs[j].name), descs[j].types ?: "?");
                    }
                    free(descs);

                    descs = protocol_copyMethodDescriptionList(protos[i], NO, YES, &mc);
                    if (mc > 0) {
                        printf("    Optional instance methods:\n");
                        for (unsigned int j = 0; j < mc; j++)
                            printf("      %s  (%s)\n", sel_getName(descs[j].name), descs[j].types ?: "?");
                    }
                    free(descs);
                    printf("\n");
                }
            }
            free(protos);
        }

        // ════════════════════════════════════════════════════════════════
        // Section 11: Try _ANEClient.compileModel directly
        // ════════════════════════════════════════════════════════════════
        printf("  ══ Section 11: _ANEClient compile methods ══\n\n");
        {
            Class clientCls = NSClassFromString(@"_ANEClient");
            if (clientCls) {
                unsigned int mc = 0;
                Method *methods = class_copyMethodList(clientCls, &mc);
                for (unsigned int i = 0; i < mc; i++) {
                    const char *name = sel_getName(method_getName(methods[i]));
                    if (strstr(name, "ompile") || strstr(name, "nalytics") || strstr(name, "erf")) {
                        printf("  %s\n    %s\n", name, method_getTypeEncoding(methods[i]) ?: "?");
                        unsigned int nargs = method_getNumberOfArguments(methods[i]);
                        for (unsigned int a = 0; a < nargs; a++) {
                            char argType[256] = {0};
                            method_getArgumentType(methods[i], a, argType, sizeof(argType));
                            printf("    arg[%u]: %s\n", a, argType);
                        }
                        printf("\n");
                    }
                }
                free(methods);
            }
        }

        // ════════════════════════════════════════════════════════════════
        // Section 12: Restore original implementations
        // ════════════════════════════════════════════════════════════════
        printf("  ══ Section 12: Cleanup ══\n\n");
        {
            if (g_origCompileIMM) {
                Class immCls = NSClassFromString(@"_ANEInMemoryModel");
                Method m = class_getInstanceMethod(immCls, @selector(compileWithQoS:options:error:));
                if (m) method_setImplementation(m, g_origCompileIMM);
                printf("  Restored _ANEInMemoryModel.compileWithQoS\n");
            }
            if (g_origCompileDaemon) {
                Class daemonCls = NSClassFromString(@"_ANEDaemonConnection");
                unsigned int mc = 0;
                Method *methods = class_copyMethodList(daemonCls, &mc);
                for (unsigned int i = 0; i < mc; i++) {
                    const char *name = sel_getName(method_getName(methods[i]));
                    if (strstr(name, "compileModel") && strstr(name, "withReply")) {
                        method_setImplementation(methods[i], g_origCompileDaemon);
                        printf("  Restored _ANEDaemonConnection.compileModel\n");
                        break;
                    }
                }
                free(methods);
            }
        }

        printf("\n  ██████ XPC INTERCEPTION COMPLETE ██████\n\n");
    }
    return 0;
}
