// ane.m — Implementation of libane: Clean C API for Apple Neural Engine
// Wraps 35 private classes from AppleNeuralEngine.framework

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <objc/message.h>
#include <dlfcn.h>
#include "ane.h"

// ===== Private class references =====
static Class g_Desc     = nil;  // _ANEInMemoryModelDescriptor (or alternative)
static Class g_InMem    = nil;  // _ANEInMemoryModel (or alternative)
static Class g_Req      = nil;  // _ANERequest (or alternative)
static Class g_IO       = nil;  // _ANEIOSurfaceObject (or alternative)
static Class g_Client   = nil;  // _ANEClient (or alternative)
static Class g_DevInfo  = nil;  // _ANEDeviceInfo (or alternative)
static bool g_init = false;
static int g_compiles = 0;

// ===== Resolved selector cache =====
// If Apple renames selectors, we try alternatives and cache what works.
static SEL g_sel_modelWithMIL    = nil;  // descriptor factory
static SEL g_sel_inMemoryModel   = nil;  // model factory
static SEL g_sel_compile         = nil;  // compile method
static SEL g_sel_load            = nil;  // load method
static SEL g_sel_evaluate        = nil;  // evaluate method
static SEL g_sel_unload          = nil;  // unload method
static SEL g_sel_hexId           = nil;  // hex identifier
static SEL g_sel_objWithSurface  = nil;  // IOSurface wrapper factory
static SEL g_sel_reqWithInputs   = nil;  // request factory

// ===== API version info =====
static ANEAPIInfo g_api_info = {0};
static char g_fw_path[256] = {0};
static char g_desc_class_name[128] = {0};
static char g_model_class_name[128] = {0};

// ===== Known framework paths (tried in order) =====
static const char *FRAMEWORK_PATHS[] = {
    "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
    "/System/Library/Frameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
    "/System/Library/PrivateFrameworks/ANECompiler.framework/ANECompiler",
    NULL
};

// ===== Known class name alternatives (tried in order) =====
// If Apple renames classes in a future macOS, add the new names here.

static const char *DESC_CLASS_NAMES[] = {
    "_ANEInMemoryModelDescriptor",
    "_ANEModelDescriptor",
    "ANEInMemoryModelDescriptor",
    "ANEModelDescriptor",
    NULL
};

static const char *MODEL_CLASS_NAMES[] = {
    "_ANEInMemoryModel",
    "_ANEModel",
    "ANEInMemoryModel",
    "ANEModel",
    NULL
};

static const char *REQ_CLASS_NAMES[] = {
    "_ANERequest",
    "ANERequest",
    "_ANEEvaluationRequest",
    NULL
};

static const char *IO_CLASS_NAMES[] = {
    "_ANEIOSurfaceObject",
    "ANEIOSurfaceObject",
    "_ANESurfaceObject",
    NULL
};

static const char *CLIENT_CLASS_NAMES[] = {
    "_ANEClient",
    "ANEClient",
    NULL
};

static const char *DEVINFO_CLASS_NAMES[] = {
    "_ANEDeviceInfo",
    "ANEDeviceInfo",
    NULL
};

// ===== Helper: try class names until one resolves =====
static Class resolve_class(const char **names, char *out_name, size_t out_len) {
    for (int i = 0; names[i]; i++) {
        Class cls = NSClassFromString([NSString stringWithUTF8String:names[i]]);
        if (cls) {
            if (out_name) strncpy(out_name, names[i], out_len - 1);
            return cls;
        }
    }
    return nil;
}

// ===== Helper: try selectors until one exists on class =====
static SEL resolve_selector(Class cls, const char **sel_names) {
    for (int i = 0; sel_names[i]; i++) {
        SEL s = sel_registerName(sel_names[i]);
        if ([cls respondsToSelector:s] || [cls instancesRespondToSelector:s]) return s;
    }
    return nil;
}

// ===== Count all ANE classes in runtime =====
static int count_ane_classes(void) {
    unsigned int count = 0;
    Class *classes = objc_copyClassList(&count);
    int ane_count = 0;
    for (unsigned int i = 0; i < count; i++) {
        NSString *name = NSStringFromClass(classes[i]);
        if ([name hasPrefix:@"_ANE"] || [name hasPrefix:@"ANE"]) ane_count++;
    }
    free(classes);
    return ane_count;
}

// ===== Kernel handle =====
struct ANEKernel {
    id model;               // _ANEInMemoryModel (ARC-managed)
    IOSurfaceRef *ioIn;
    IOSurfaceRef *ioOut;
    id request;             // _ANERequest (ARC-managed)
    NSString *tmpDir;       // (ARC-managed)
    int nIn, nOut;
    size_t *inBytes;
    size_t *outBytes;
    // Delta compilation support
    void *milData;          // Copy of MIL text for reload
    size_t milLen;
};

// ===== IOSurface creation =====
static IOSurfaceRef make_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes),
        (id)kIOSurfacePixelFormat: @0
    });
}

// ===== Public API =====

int ane_init(void) {
    if (g_init) return 0;

    memset(&g_api_info, 0, sizeof(g_api_info));

    // --- Step 1: Load framework (try known paths) ---
    void *h = NULL;
    for (int i = 0; FRAMEWORK_PATHS[i]; i++) {
        h = dlopen(FRAMEWORK_PATHS[i], RTLD_NOW);
        if (h) {
            strncpy(g_fw_path, FRAMEWORK_PATHS[i], sizeof(g_fw_path) - 1);
            g_api_info.framework_path = g_fw_path;
            break;
        }
    }
    if (!h) {
        fprintf(stderr, "libane: no ANE framework found at any known path\n");
        return -1;
    }

    // --- Step 2: Resolve classes (try alternatives) ---
    g_Desc    = resolve_class(DESC_CLASS_NAMES, g_desc_class_name, sizeof(g_desc_class_name));
    g_InMem   = resolve_class(MODEL_CLASS_NAMES, g_model_class_name, sizeof(g_model_class_name));
    g_Req     = resolve_class(REQ_CLASS_NAMES, NULL, 0);
    g_IO      = resolve_class(IO_CLASS_NAMES, NULL, 0);
    g_Client  = resolve_class(CLIENT_CLASS_NAMES, NULL, 0);
    g_DevInfo = resolve_class(DEVINFO_CLASS_NAMES, NULL, 0);

    g_api_info.has_descriptor    = (g_Desc != nil);
    g_api_info.has_model         = (g_InMem != nil);
    g_api_info.has_request       = (g_Req != nil);
    g_api_info.has_iosurface_obj = (g_IO != nil);
    g_api_info.has_client        = (g_Client != nil);
    g_api_info.has_device_info   = (g_DevInfo != nil);
    g_api_info.descriptor_class  = g_desc_class_name;
    g_api_info.model_class       = g_model_class_name;

    // Optional classes
    g_api_info.has_qos_mapper  = (NSClassFromString(@"_ANEQoSMapper") != nil);
    g_api_info.has_chaining    = (NSClassFromString(@"_ANEChainingRequest") != nil);
    g_api_info.has_perf_stats  = (NSClassFromString(@"_ANEPerformanceStats") != nil);
    g_api_info.has_buffer      = (NSClassFromString(@"_ANEBuffer") != nil);

    // Count all ANE classes
    g_api_info.classes_found = count_ane_classes();

    // --- Step 3: Resolve selectors (try alternatives) ---

    // Descriptor factory: modelWithMILText:weights:optionsPlist:
    if (g_Desc) {
        const char *desc_sels[] = {
            "modelWithMILText:weights:optionsPlist:",
            "modelWithMILText:weights:options:",
            "descriptorWithMILText:weights:optionsPlist:",
            "descriptorWithMILText:weights:",
            NULL
        };
        g_sel_modelWithMIL = resolve_selector(object_getClass(g_Desc), desc_sels);
        if (!g_sel_modelWithMIL) {
            fprintf(stderr, "libane: WARNING: descriptor factory selector not found, trying default\n");
            g_sel_modelWithMIL = @selector(modelWithMILText:weights:optionsPlist:);
        }
    }

    // Model factory: inMemoryModelWithDescriptor:
    if (g_InMem) {
        const char *model_sels[] = {
            "inMemoryModelWithDescriptor:",
            "modelWithDescriptor:",
            NULL
        };
        g_sel_inMemoryModel = resolve_selector(object_getClass(g_InMem), model_sels);
        if (!g_sel_inMemoryModel) g_sel_inMemoryModel = @selector(inMemoryModelWithDescriptor:);
    }

    // Compile/load/eval/unload
    if (g_InMem) {
        const char *compile_sels[] = { "compileWithQoS:options:error:", "compileWithOptions:error:", NULL };
        const char *load_sels[]    = { "loadWithQoS:options:error:", "loadWithOptions:error:", NULL };
        const char *eval_sels[]    = { "evaluateWithQoS:options:request:error:", "evaluateWithRequest:error:", NULL };
        const char *unload_sels[]  = { "unloadWithQoS:error:", "unloadWithError:", NULL };
        const char *hexid_sels[]   = { "hexStringIdentifier", "identifier", "modelIdentifier", NULL };

        g_sel_compile  = resolve_selector(g_InMem, compile_sels)  ?: @selector(compileWithQoS:options:error:);
        g_sel_load     = resolve_selector(g_InMem, load_sels)     ?: @selector(loadWithQoS:options:error:);
        g_sel_evaluate = resolve_selector(g_InMem, eval_sels)     ?: @selector(evaluateWithQoS:options:request:error:);
        g_sel_unload   = resolve_selector(g_InMem, unload_sels)   ?: @selector(unloadWithQoS:error:);
        g_sel_hexId    = resolve_selector(g_InMem, hexid_sels)    ?: @selector(hexStringIdentifier);
    }

    // IOSurface wrapper factory
    if (g_IO) {
        const char *io_sels[] = { "objectWithIOSurface:", "surfaceObjectWithIOSurface:", NULL };
        g_sel_objWithSurface = resolve_selector(object_getClass(g_IO), io_sels) ?: @selector(objectWithIOSurface:);
    }

    // Request factory
    if (g_Req) {
        const char *req_sels[] = {
            "requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:",
            "requestWithInputs:inputIndices:outputs:outputIndices:procedureIndex:",
            NULL
        };
        g_sel_reqWithInputs = resolve_selector(object_getClass(g_Req), req_sels)
            ?: @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:);
    }

    // --- Step 4: Get macOS build version ---
    static char build_buf[64];
    if (g_DevInfo) {
        @try {
            NSString *b = ((NSString*(*)(id,SEL))objc_msgSend)((id)g_DevInfo, sel_registerName("buildVersion"));
            if (b && [b isKindOfClass:[NSString class]]) {
                strncpy(build_buf, [b UTF8String], 63);
                g_api_info.macos_build = build_buf;
            }
        } @catch (NSException *ex) {}
    }

    // --- Step 5: Determine API version ---
    if (g_Desc && g_InMem && g_Req && g_IO) {
        g_api_info.api_version = 1;  // v1 = current known API (macOS 15-26)
    } else {
        g_api_info.api_version = 0;  // unknown / degraded
    }

    // --- Step 6: Check critical classes ---
    if (!g_Desc || !g_InMem || !g_Req || !g_IO) {
        fprintf(stderr, "libane: critical classes missing (desc=%s model=%s req=%s io=%s)\n",
            g_Desc ? "OK" : "MISSING", g_InMem ? "OK" : "MISSING",
            g_Req ? "OK" : "MISSING", g_IO ? "OK" : "MISSING");
        fprintf(stderr, "libane: found %d ANE classes total — API may have changed\n",
            g_api_info.classes_found);

        // List what ANE classes DO exist to help debug
        unsigned int count = 0;
        Class *classes = objc_copyClassList(&count);
        fprintf(stderr, "libane: discovered ANE classes:\n");
        for (unsigned int i = 0; i < count; i++) {
            NSString *name = NSStringFromClass(classes[i]);
            if ([name hasPrefix:@"_ANE"] || [name hasPrefix:@"ANE"])
                fprintf(stderr, "  - %s\n", [name UTF8String]);
        }
        free(classes);

        return -2;
    }

    g_init = true;
    g_compiles = 0;
    return 0;
}

ANEAPIInfo ane_api_info(void) {
    return g_api_info;
}

void ane_print_diagnostics(void) {
    ANEAPIInfo a = g_api_info;
    fprintf(stderr, "=== libane diagnostics ===\n");
    fprintf(stderr, "API version:     %d %s\n", a.api_version,
        a.api_version == 1 ? "(current)" : "(UNKNOWN - may need update)");
    fprintf(stderr, "macOS build:     %s\n", a.macos_build ? a.macos_build : "?");
    fprintf(stderr, "Framework:       %s\n", a.framework_path ? a.framework_path : "NOT LOADED");
    fprintf(stderr, "Classes found:   %d\n", a.classes_found);
    fprintf(stderr, "\nCritical classes:\n");
    fprintf(stderr, "  Descriptor:    %s (%s)\n", a.has_descriptor ? "OK" : "MISSING", a.descriptor_class ? a.descriptor_class : "?");
    fprintf(stderr, "  Model:         %s (%s)\n", a.has_model ? "OK" : "MISSING", a.model_class ? a.model_class : "?");
    fprintf(stderr, "  Request:       %s\n", a.has_request ? "OK" : "MISSING");
    fprintf(stderr, "  IOSurface:     %s\n", a.has_iosurface_obj ? "OK" : "MISSING");
    fprintf(stderr, "\nOptional classes:\n");
    fprintf(stderr, "  Client:        %s\n", a.has_client ? "OK" : "missing");
    fprintf(stderr, "  DeviceInfo:    %s\n", a.has_device_info ? "OK" : "missing");
    fprintf(stderr, "  QoSMapper:     %s\n", a.has_qos_mapper ? "OK" : "missing");
    fprintf(stderr, "  Chaining:      %s\n", a.has_chaining ? "OK" : "missing");
    fprintf(stderr, "  PerfStats:     %s\n", a.has_perf_stats ? "OK" : "missing");
    fprintf(stderr, "  Buffer:        %s\n", a.has_buffer ? "OK" : "missing");
    fprintf(stderr, "\nResolved selectors:\n");
    fprintf(stderr, "  desc factory:  %s\n", g_sel_modelWithMIL ? sel_getName(g_sel_modelWithMIL) : "NONE");
    fprintf(stderr, "  model factory: %s\n", g_sel_inMemoryModel ? sel_getName(g_sel_inMemoryModel) : "NONE");
    fprintf(stderr, "  compile:       %s\n", g_sel_compile ? sel_getName(g_sel_compile) : "NONE");
    fprintf(stderr, "  load:          %s\n", g_sel_load ? sel_getName(g_sel_load) : "NONE");
    fprintf(stderr, "  evaluate:      %s\n", g_sel_evaluate ? sel_getName(g_sel_evaluate) : "NONE");
    fprintf(stderr, "  unload:        %s\n", g_sel_unload ? sel_getName(g_sel_unload) : "NONE");
    fprintf(stderr, "  hexId:         %s\n", g_sel_hexId ? sel_getName(g_sel_hexId) : "NONE");
    fprintf(stderr, "  ioSurface:     %s\n", g_sel_objWithSurface ? sel_getName(g_sel_objWithSurface) : "NONE");
    fprintf(stderr, "  request:       %s\n", g_sel_reqWithInputs ? sel_getName(g_sel_reqWithInputs) : "NONE");

    if (a.api_version == 0) {
        fprintf(stderr, "\n*** API VERSION UNKNOWN ***\n");
        fprintf(stderr, "Apple may have changed the private API in this macOS version.\n");
        fprintf(stderr, "libane needs to be updated. Check for new class names above.\n");
    }
    fprintf(stderr, "==========================\n");
}

ANEDeviceInfo ane_device_info(void) {
    ANEDeviceInfo info = {0};
    if (!g_DevInfo) return info;

    info.has_ane = ((BOOL(*)(id,SEL))objc_msgSend)((id)g_DevInfo, sel_registerName("hasANE"));
    info.num_cores = (int)((long(*)(id,SEL))objc_msgSend)((id)g_DevInfo, sel_registerName("numANECores"));
    info.num_units = (int)((long(*)(id,SEL))objc_msgSend)((id)g_DevInfo, sel_registerName("numANEs"));
    info.board_type = (int)((long(*)(id,SEL))objc_msgSend)((id)g_DevInfo, sel_registerName("aneBoardType"));
    info.is_virtual = ((BOOL(*)(id,SEL))objc_msgSend)((id)g_DevInfo, sel_registerName("isVirtualMachine"));

    // String properties (static lifetime — safe to return)
    static char arch_buf[64], sub_buf[64], var_buf[64], prod_buf[64], build_buf[64];

    NSString *s;
    s = ((NSString*(*)(id,SEL))objc_msgSend)((id)g_DevInfo, sel_registerName("aneArchitectureType"));
    if (s && [s isKindOfClass:[NSString class]]) { strncpy(arch_buf, [s UTF8String], 63); info.arch = arch_buf; }

    s = ((NSString*(*)(id,SEL))objc_msgSend)((id)g_DevInfo, sel_registerName("aneSubType"));
    if (s && [s isKindOfClass:[NSString class]]) { strncpy(sub_buf, [s UTF8String], 63); info.sub_type = sub_buf; }

    s = ((NSString*(*)(id,SEL))objc_msgSend)((id)g_DevInfo, sel_registerName("aneSubTypeVariant"));
    if (s && [s isKindOfClass:[NSString class]]) { strncpy(var_buf, [s UTF8String], 63); info.variant = var_buf; }

    s = ((NSString*(*)(id,SEL))objc_msgSend)((id)g_DevInfo, sel_registerName("productName"));
    if (s && [s isKindOfClass:[NSString class]]) { strncpy(prod_buf, [s UTF8String], 63); info.product = prod_buf; }

    s = ((NSString*(*)(id,SEL))objc_msgSend)((id)g_DevInfo, sel_registerName("buildVersion"));
    if (s && [s isKindOfClass:[NSString class]]) { strncpy(build_buf, [s UTF8String], 63); info.build = build_buf; }

    return info;
}

ANEKernel *ane_compile(const char *mil, size_t mil_len,
                       const ANEWeight *weights, int n_weights,
                       int n_inputs, const size_t *input_sizes,
                       int n_outputs, const size_t *output_sizes,
                       ANEQoS qos) {
    @autoreleasepool {
        if (!g_init) return NULL;

        NSData *milData = [NSData dataWithBytes:mil length:mil_len];
        NSError *e = nil;

        // Build weight dictionary
        NSMutableDictionary *wdict = [NSMutableDictionary dictionary];
        for (int i = 0; i < n_weights; i++) {
            NSString *name = [NSString stringWithUTF8String:weights[i].name];
            NSData *data = [NSData dataWithBytes:weights[i].data length:weights[i].len];
            wdict[name] = @{@"offset": @0, @"data": data};
        }

        // Create descriptor (using resolved selector)
        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            g_Desc, g_sel_modelWithMIL,
            milData, wdict.count > 0 ? wdict : @{}, nil);
        if (!desc) return NULL;

        // Create model (using resolved selector)
        id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(
            g_InMem, g_sel_inMemoryModel, desc);
        if (!mdl) return NULL;

        // Pre-populate temp directory (required by ANE compiler)
        id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, g_sel_hexId);
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
            withIntermediateDirectories:YES attributes:nil error:nil];
        [milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];

        for (int i = 0; i < n_weights; i++) {
            NSString *name = [NSString stringWithUTF8String:weights[i].name];
            NSString *relPath = name;
            if ([name hasPrefix:@"@model_path/"]) relPath = [name substringFromIndex:12];
            NSString *fullPath = [td stringByAppendingPathComponent:relPath];
            [fm createDirectoryAtPath:[fullPath stringByDeletingLastPathComponent]
                withIntermediateDirectories:YES attributes:nil error:nil];
            [[NSData dataWithBytes:weights[i].data length:weights[i].len]
                writeToFile:fullPath atomically:YES];
        }

        // Compile (using resolved selector)
        unsigned int q = (unsigned int)qos;
        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, g_sel_compile, q, @{}, &e)) {
            fprintf(stderr, "libane: compile failed: %s\n",
                    e ? [[e description] UTF8String] : "unknown");
            [fm removeItemAtPath:td error:nil];
            return NULL;
        }

        // Load (retry once after 100ms, using resolved selector)
        BOOL loaded = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, g_sel_load, q, @{}, &e);
        if (!loaded) {
            usleep(100000);
            e = nil;
            loaded = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                    mdl, g_sel_load, q, @{}, &e);
        }
        if (!loaded) {
            fprintf(stderr, "libane: load failed: %s\n",
                    e ? [[e description] UTF8String] : "unknown");
            [fm removeItemAtPath:td error:nil];
            return NULL;
        }

        __sync_fetch_and_add(&g_compiles, 1);
        if (g_compiles >= ANE_COMPILE_SAFE_LIMIT) {
            fprintf(stderr, "libane: WARNING: %d/%d compile budget used — "
                    "save checkpoint and restart process soon\n",
                    g_compiles, ANE_COMPILE_BUDGET);
        }

        // SRAM budget check
        size_t total_io = 0;
        for (int i = 0; i < n_inputs; i++) total_io += input_sizes[i];
        for (int i = 0; i < n_outputs; i++) total_io += output_sizes[i];
        if (total_io > 32 * 1024 * 1024)
            fprintf(stderr, "libane: WARNING: total I/O %zuMB exceeds ANE SRAM (~32MB), "
                    "expect ~30%% throughput drop\n", total_io >> 20);

        // Build kernel handle
        ANEKernel *k = (ANEKernel *)calloc(1, sizeof(ANEKernel));
        k->model = mdl;
        k->tmpDir = td;
        k->milData = malloc(mil_len);
        memcpy(k->milData, mil, mil_len);
        k->milLen = mil_len;
        k->nIn = n_inputs;
        k->nOut = n_outputs;
        k->inBytes = (size_t *)malloc(n_inputs * sizeof(size_t));
        k->outBytes = (size_t *)malloc(n_outputs * sizeof(size_t));
        memcpy(k->inBytes, input_sizes, n_inputs * sizeof(size_t));
        memcpy(k->outBytes, output_sizes, n_outputs * sizeof(size_t));

        // Create IOSurfaces
        k->ioIn = (IOSurfaceRef *)malloc(n_inputs * sizeof(IOSurfaceRef));
        k->ioOut = (IOSurfaceRef *)malloc(n_outputs * sizeof(IOSurfaceRef));
        for (int i = 0; i < n_inputs; i++)  k->ioIn[i] = make_surface(input_sizes[i]);
        for (int i = 0; i < n_outputs; i++) k->ioOut[i] = make_surface(output_sizes[i]);

        // Build request
        NSMutableArray *wIns = [NSMutableArray arrayWithCapacity:n_inputs];
        NSMutableArray *iIdx = [NSMutableArray arrayWithCapacity:n_inputs];
        for (int i = 0; i < n_inputs; i++) {
            [wIns addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                g_IO, g_sel_objWithSurface, k->ioIn[i])];
            [iIdx addObject:@(i)];
        }
        NSMutableArray *wOuts = [NSMutableArray arrayWithCapacity:n_outputs];
        NSMutableArray *oIdx = [NSMutableArray arrayWithCapacity:n_outputs];
        for (int i = 0; i < n_outputs; i++) {
            [wOuts addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                g_IO, g_sel_objWithSurface, k->ioOut[i])];
            [oIdx addObject:@(i)];
        }
        k->request = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
            g_Req, g_sel_reqWithInputs,
            wIns, iIdx, wOuts, oIdx, nil, nil, @0);

        return k;
    }
}

bool ane_eval(ANEKernel *k, ANEQoS qos) {
    @autoreleasepool {
        if (!k || !k->model) return false;
        NSError *e = nil;
        return ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            k->model, g_sel_evaluate,
            (unsigned int)qos, @{}, k->request, &e);
    }
}

void ane_write(ANEKernel *k, int idx, const void *data, size_t bytes) {
    if (!k || idx < 0 || idx >= k->nIn) return;
    IOSurfaceLock(k->ioIn[idx], 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(k->ioIn[idx]), data, bytes);
    IOSurfaceUnlock(k->ioIn[idx], 0, NULL);
}

void ane_read(ANEKernel *k, int idx, void *data, size_t bytes) {
    if (!k || idx < 0 || idx >= k->nOut) return;
    IOSurfaceLock(k->ioOut[idx], kIOSurfaceLockReadOnly, NULL);
    memcpy(data, IOSurfaceGetBaseAddress(k->ioOut[idx]), bytes);
    IOSurfaceUnlock(k->ioOut[idx], kIOSurfaceLockReadOnly, NULL);
}

void *ane_input_ptr(ANEKernel *k, int idx) {
    if (!k || idx < 0 || idx >= k->nIn) return NULL;
    return IOSurfaceGetBaseAddress(k->ioIn[idx]);
}

void *ane_output_ptr(ANEKernel *k, int idx) {
    if (!k || idx < 0 || idx >= k->nOut) return NULL;
    return IOSurfaceGetBaseAddress(k->ioOut[idx]);
}

void ane_lock_input(ANEKernel *k, int idx)   { if (k && idx >= 0 && idx < k->nIn)  IOSurfaceLock(k->ioIn[idx], 0, NULL); }
void ane_unlock_input(ANEKernel *k, int idx) { if (k && idx >= 0 && idx < k->nIn)  IOSurfaceUnlock(k->ioIn[idx], 0, NULL); }
void ane_lock_output(ANEKernel *k, int idx)  { if (k && idx >= 0 && idx < k->nOut) IOSurfaceLock(k->ioOut[idx], kIOSurfaceLockReadOnly, NULL); }
void ane_unlock_output(ANEKernel *k, int idx){ if (k && idx >= 0 && idx < k->nOut) IOSurfaceUnlock(k->ioOut[idx], kIOSurfaceLockReadOnly, NULL); }

// ===== Weight Blob Builders =====

ANEWeight ane_weight_fp16(const char *name, const float *src, int rows, int cols) {
    int wsize = rows * cols * 2;
    int total = 128 + wsize;
    uint8_t *buf = (uint8_t *)calloc(total, 1);

    buf[0] = 0x01; buf[4] = 0x02;
    buf[64] = 0xEF; buf[65] = 0xBE; buf[66] = 0xAD; buf[67] = 0xDE;
    buf[68] = 0x01;
    *(uint32_t*)(buf + 72) = wsize;
    *(uint32_t*)(buf + 80) = 128;

    _Float16 *fp16 = (_Float16 *)(buf + 128);
    for (int i = 0; i < rows * cols; i++) fp16[i] = (_Float16)src[i];

    return (ANEWeight){.data = buf, .len = total, .name = name};
}

ANEWeight ane_weight_fp16_transposed(const char *name, const float *src, int rows, int cols) {
    int wsize = rows * cols * 2;
    int total = 128 + wsize;
    uint8_t *buf = (uint8_t *)calloc(total, 1);

    buf[0] = 0x01; buf[4] = 0x02;
    buf[64] = 0xEF; buf[65] = 0xBE; buf[66] = 0xAD; buf[67] = 0xDE;
    buf[68] = 0x01;
    *(uint32_t*)(buf + 72) = wsize;
    *(uint32_t*)(buf + 80) = 128;

    _Float16 *fp16 = (_Float16 *)(buf + 128);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            fp16[j * rows + i] = (_Float16)src[i * cols + j];

    return (ANEWeight){.data = buf, .len = total, .name = name};
}

ANEWeight ane_weight_int8(const char *name, const float *src, int rows, int cols, float *out_scale) {
    float max_abs = 0.0f;
    for (int i = 0; i < rows * cols; i++) {
        float a = src[i] < 0 ? -src[i] : src[i];
        if (a > max_abs) max_abs = a;
    }
    float scale = max_abs / 127.0f;
    if (scale == 0.0f) scale = 1.0f;

    int wsize = rows * cols;
    int total = 64 + wsize;
    uint8_t *buf = (uint8_t *)calloc(total, 1);
    buf[0] = 0xEF; buf[1] = 0xBE; buf[2] = 0xAD; buf[3] = 0xDE;
    buf[4] = 0x01; buf[10] = 0x08;

    int8_t *qdata = (int8_t *)(buf + 64);
    for (int i = 0; i < wsize; i++) {
        float v = src[i] / scale;
        if (v > 127.0f) v = 127.0f;
        if (v < -128.0f) v = -128.0f;
        qdata[i] = (int8_t)(v + (v >= 0 ? 0.5f : -0.5f));
    }

    *out_scale = scale;
    return (ANEWeight){.data = buf, .len = total, .name = name};
}

void ane_weight_free(ANEWeight *w) {
    if (w && w->data) { free(w->data); w->data = NULL; w->len = 0; }
}

// ===== Stacked Conv (for benchmarks) =====

ANEWeight ane_weight_stacked(const char *name, int ch, int depth) {
    size_t wsize = (size_t)ch * ch * 2;           // FP16 weights per layer
    size_t chunkSize = 64 + wsize;
    size_t total = 64 + chunkSize * depth;
    uint8_t *buf = (uint8_t *)calloc(total, 1);

    buf[0] = 0x01; buf[4] = 0x02;
    for (int i = 0; i < depth; i++) {
        uint8_t *chunk = buf + 64 + i * chunkSize;
        chunk[0] = 0xEF; chunk[1] = 0xBE; chunk[2] = 0xAD; chunk[3] = 0xDE;
        chunk[4] = 0x01; chunk[10] = 0x08;
        uint16_t *fp16 = (uint16_t *)(chunk + 64);
        for (size_t j = 0; j < wsize / 2; j++)
            fp16[j] = (arc4random() & 0x03FF) | 0x2000;  // small random FP16
    }
    return (ANEWeight){.data = buf, .len = total, .name = name};
}

char *ane_mil_stacked_conv(int ch, int sp, int depth, const char *weight_name) {
    size_t bufsize = 2048 + depth * 512;
    char *buf = (char *)malloc(bufsize);
    int pos = 0;

    pos += snprintf(buf + pos, bufsize - pos,
        "program(1.3)\n"
        "[buildInfo = dict<string, string>({"
        "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, "
        "{\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n{\n"
        "  func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
        "    string c_pt = const()[name=string(\"c_pt\"), val=string(\"valid\")];\n"
        "    tensor<int32, [2]> c_st = const()[name=string(\"c_st\"), val=tensor<int32, [2]>([1, 1])];\n"
        "    tensor<int32, [4]> c_pd = const()[name=string(\"c_pd\"), val=tensor<int32, [4]>([0, 0, 0, 0])];\n"
        "    tensor<int32, [2]> c_dl = const()[name=string(\"c_dl\"), val=tensor<int32, [2]>([1, 1])];\n"
        "    int32 c_gr = const()[name=string(\"c_gr\"), val=int32(1)];\n"
        "    string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];\n"
        "    tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype=to16, x=x)[name=string(\"cin\")];\n",
        ch, sp, ch, sp);

    size_t cs = 64 + (size_t)ch * ch * 2;
    for (int i = 0; i < depth; i++) {
        const char *prev = (i == 0) ? "x16" : NULL;
        char prev_buf[16];
        if (i > 0) { snprintf(prev_buf, sizeof(prev_buf), "c%d", i - 1); prev = prev_buf; }

        pos += snprintf(buf + pos, bufsize - pos,
            "    tensor<fp16, [%d, %d, 1, 1]> W%d = const()[name=string(\"W%d\"), "
            "val=tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path=string(\"%s\"), offset=uint64(%zu)))];\n"
            "    tensor<fp16, [1, %d, 1, %d]> c%d = conv(dilations=c_dl, groups=c_gr, "
            "pad=c_pd, pad_type=c_pt, strides=c_st, weight=W%d, x=%s)[name=string(\"c%d\")];\n",
            ch, ch, i, i, ch, ch, weight_name, (size_t)(64 + i * cs),
            ch, sp, i, i, prev, i);
    }

    char last[16];
    snprintf(last, sizeof(last), "c%d", depth - 1);
    pos += snprintf(buf + pos, bufsize - pos,
        "    string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];\n"
        "    tensor<fp32, [1, %d, 1, %d]> y = cast(dtype=to32, x=%s)[name=string(\"cout\")];\n"
        "  } -> (y);\n}\n",
        ch, sp, last);

    return buf;
}

// ===== MIL Generation =====

char *ane_mil_header(void) {
    const char *hdr =
        "program(1.3)\n"
        "[buildInfo = dict<string, string>({"
        "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, "
        "{\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n{\n";
    return strdup(hdr);
}

char *ane_mil_linear(int in_ch, int out_ch, int seq, const char *weight_name) {
    char *buf = (char *)malloc(4096);
    snprintf(buf, 4096,
        "program(1.3)\n"
        "[buildInfo = dict<string, string>({"
        "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, "
        "{\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n{\n"
        "  func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
        "    string c_pt = const()[name=string(\"c_pt\"), val=string(\"valid\")];\n"
        "    tensor<int32, [2]> c_st = const()[name=string(\"c_st\"), val=tensor<int32, [2]>([1, 1])];\n"
        "    tensor<int32, [4]> c_pd = const()[name=string(\"c_pd\"), val=tensor<int32, [4]>([0, 0, 0, 0])];\n"
        "    tensor<int32, [2]> c_dl = const()[name=string(\"c_dl\"), val=tensor<int32, [2]>([1, 1])];\n"
        "    int32 c_gr = const()[name=string(\"c_gr\"), val=int32(1)];\n"
        "    string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];\n"
        "    tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype=to16, x=x)[name=string(\"cin\")];\n"
        "    tensor<fp16, [%d, %d, 1, 1]> W = const()[name=string(\"W\"), "
        "val=tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path=string(\"%s\"), offset=uint64(64)))];\n"
        "    tensor<fp16, [1, %d, 1, %d]> y16 = conv(dilations=c_dl, groups=c_gr, "
        "pad=c_pd, pad_type=c_pt, strides=c_st, weight=W, x=x16)[name=string(\"conv\")];\n"
        "    string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];\n"
        "    tensor<fp32, [1, %d, 1, %d]> y = cast(dtype=to32, x=y16)[name=string(\"cout\")];\n"
        "  } -> (y);\n}\n",
        in_ch, seq,
        in_ch, seq,
        out_ch, in_ch, out_ch, in_ch, weight_name,
        out_ch, seq,
        out_ch, seq);
    return buf;
}

// ===== Lifecycle =====

int ane_compile_count(void) { return g_compiles; }

bool ane_reload_weights(ANEKernel *k, const ANEWeight *weights, int n_weights, ANEQoS qos) {
    @autoreleasepool {
        if (!k || !k->model || !k->tmpDir) return false;
        NSError *e = nil;
        unsigned int q = (unsigned int)qos;
        NSFileManager *fm = [NSFileManager defaultManager];

        // Step 1: Unload model from ANE
        if (g_sel_unload) {
            BOOL ok = ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
                k->model, g_sel_unload, q, &e);
            if (!ok) {
                fprintf(stderr, "libane: delta unload failed: %s\n",
                        e ? [[e description] UTF8String] : "unknown");
                return false;
            }
        }

        // Step 2: Reconstruct tmpDir (ANE cleans it on unload)
        [fm createDirectoryAtPath:[k->tmpDir stringByAppendingPathComponent:@"weights"]
            withIntermediateDirectories:YES attributes:nil error:nil];

        // Re-write MIL text
        if (k->milData && k->milLen > 0) {
            NSData *milNS = [NSData dataWithBytes:k->milData length:k->milLen];
            [milNS writeToFile:[k->tmpDir stringByAppendingPathComponent:@"model.mil"]
                    atomically:YES];
        }

        // Write updated weight files (source + compiled 'data' blob)
        for (int i = 0; i < n_weights; i++) {
            NSString *name = [NSString stringWithUTF8String:weights[i].name];
            NSString *relPath = name;
            if ([name hasPrefix:@"@model_path/"]) relPath = [name substringFromIndex:12];
            NSString *fullPath = [k->tmpDir stringByAppendingPathComponent:relPath];

            [fm createDirectoryAtPath:[fullPath stringByDeletingLastPathComponent]
                withIntermediateDirectories:YES attributes:nil error:nil];
            NSData *wdata = [NSData dataWithBytes:weights[i].data length:weights[i].len];
            [wdata writeToFile:fullPath atomically:NO];

            // Also patch the compiled 'data' blob (same format)
            NSString *dataPath = [k->tmpDir stringByAppendingPathComponent:@"data"];
            [wdata writeToFile:dataPath atomically:NO];
        }

        // Step 3: Reload onto ANE (program identity preserved)
        e = nil;
        BOOL loaded = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            k->model, g_sel_load, q, @{}, &e);
        if (!loaded) {
            usleep(100000);
            e = nil;
            loaded = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                k->model, g_sel_load, q, @{}, &e);
        }
        if (!loaded) {
            fprintf(stderr, "libane: delta reload failed: %s\n",
                    e ? [[e description] UTF8String] : "unknown");
            return false;
        }

        return true;
    }
}

void ane_free(ANEKernel *k) {
    @autoreleasepool {
        if (!k) return;
        NSError *e = nil;
        if (k->model && g_sel_unload) {
            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
                k->model, g_sel_unload, 21, &e);
        }
        for (int i = 0; i < k->nIn; i++)  if (k->ioIn[i])  CFRelease(k->ioIn[i]);
        for (int i = 0; i < k->nOut; i++) if (k->ioOut[i]) CFRelease(k->ioOut[i]);
        if (k->tmpDir)
            [[NSFileManager defaultManager] removeItemAtPath:k->tmpDir error:nil];
        free(k->ioIn); free(k->ioOut);
        free(k->inBytes); free(k->outBytes);
        if (k->milData) free(k->milData);
        k->model = nil; k->request = nil; k->tmpDir = nil;
        free(k);
    }
}
