// test_analytics_direct.m — Bypass daemon: extract/construct compiler analytics buffer
// Probes ANECompiler.framework directly, searches compiled model caches,
// inspects _ANEProgramForEvaluation, and constructs fake analytics buffers.
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <dirent.h>
#import <sys/stat.h>
#import <sys/wait.h>

#pragma mark — Utility

static void dump_class_brief(const char *name) {
    Class cls = NSClassFromString([NSString stringWithUTF8String:name]);
    if (!cls) { printf("    [%s] NOT FOUND\n", name); return; }
    printf("    [%s]\n", name);
    unsigned int count = 0;
    Method *methods = class_copyMethodList(cls, &count);
    for (unsigned int i = 0; i < count; i++) {
        SEL sel = method_getName(methods[i]);
        printf("      - %s\n", sel_getName(sel));
    }
    free(methods);
    Class meta = object_getClass((id)cls);
    methods = class_copyMethodList(meta, &count);
    for (unsigned int i = 0; i < count; i++) {
        SEL sel = method_getName(methods[i]);
        printf("      + %s\n", sel_getName(sel));
    }
    free(methods);
    unsigned int ic = 0;
    Ivar *ivars = class_copyIvarList(cls, &ic);
    for (unsigned int i = 0; i < ic; i++)
        printf("      @ %s (%s)\n", ivar_getName(ivars[i]), ivar_getTypeEncoding(ivars[i]) ?: "?");
    free(ivars);
}

// Recursively list files under a directory
static void list_files_recursive(const char *basepath, int depth) {
    if (depth > 4) return;
    DIR *dir = opendir(basepath);
    if (!dir) return;
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_name[0] == '.') continue;
        char fullpath[PATH_MAX];
        snprintf(fullpath, sizeof(fullpath), "%s/%s", basepath, entry->d_name);
        struct stat st;
        if (stat(fullpath, &st) != 0) continue;
        if (S_ISDIR(st.st_mode)) {
            printf("      [DIR] %s\n", fullpath);
            list_files_recursive(fullpath, depth + 1);
        } else {
            printf("      %s (%lld bytes)\n", fullpath, (long long)st.st_size);
        }
    }
    closedir(dir);
}

#pragma mark — Compile a kernel and return the model + temp dir

static id compile_kernel(Class g_D, Class g_I, int CH, int SP, NSString **outTempDir) {
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

    *outTempDir = td;

    NSError *e = nil;
    BOOL compOk = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
        mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e);
    if (!compOk) {
        printf("    compile FAILED: %s\n", e ? [[e description] UTF8String] : "?");
        return nil;
    }
    e = nil;
    ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
        mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
    return mdl;
}

int main() {
    @autoreleasepool {
        setbuf(stdout, NULL);

        printf("\n");
        printf("  ╔══════════════════════════════════════════════════════════════╗\n");
        printf("  ║     DIRECT ANALYTICS EXTRACTION — BYPASS DAEMON             ║\n");
        printf("  ╚══════════════════════════════════════════════════════════════╝\n\n");

        // Load frameworks
        void *ane = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
        void *esp = dlopen("/System/Library/PrivateFrameworks/Espresso.framework/Espresso", RTLD_NOW);
        void *compiler = dlopen("/System/Library/PrivateFrameworks/ANECompiler.framework/ANECompiler", RTLD_NOW);
        printf("  Frameworks loaded:\n");
        printf("    AppleNeuralEngine: %s\n", ane ? "YES" : "NO");
        printf("    Espresso:          %s\n", esp ? "YES" : "NO");
        printf("    ANECompiler:       %s\n\n", compiler ? "YES" : "NO");

        // ════════════════════════════════════════════════════════════════
        // SECTION A: Probe ANECompiler.framework symbols directly
        // ════════════════════════════════════════════════════════════════
        printf("  ══════════════════════════════════════════════════════════════\n");
        printf("  SECTION A: ANECompiler.framework symbol probe\n");
        printf("  ══════════════════════════════════════════════════════════════\n\n");

        const char *symbols[] = {
            "ANECCompile",
            "ANECCompileJIT",
            "ANECCompileOnline",
            "ANECCompileOffline",
            "ANECCreateCompilerOptionDictionary",
            "ANECCreateCompilerOptionsCFString",
            "ANECCreateCompilerInputDictionary",
            "ANECCreateCompilerInputCFString",
            "ANECCreateCompilerInputsCFString",
            "ANECGetAnalyticsBufferSize",
            "ANECGetCompilerFileFormat",
            "ANECCreateDeviceProperty",
            "ANECGetDeviceProperty",
            "ANECCreateModelDictionary",
            "ANECValidate",
            "ANECGetMutableOperationInfo",
            "ANECGetMutableOperationInfoSize",
            "ANECGetMutableProcedureInfoSize",
            "ANECGetMutableWeight",
            "ANECGetMutableWeightInfo",
            "ANECPreferredInterleaveFactors",
            "ANECValidateMutableProcedureInfo",
            "ANECUnitValidatorCreate",
            "ANECUnitValidatorCreateWithParams",
            "ANECUnitValidatorDelete",
            "ANECValidateNetworkCreate",
            "ANECRegisterTunnelingMILOpsets",
            "ANECGetValidateNetworkSupportedVersion",
            "ANECGetMPSDialectSupportedVersion",
            "ANECGetMPSSPIDialectSupportedVersion",
            // Speculative names
            "ANECGetAnalytics",
            "ANECGetAnalyticsBuffer",
            "ANECSetAnalytics",
            "ANECCompileWithAnalytics",
            "ANECGetCompilerAnalytics",
            "ANECCompileAndGetAnalytics",
            "ANECGetStaticAnalytics",
            "ANECGetPerformanceStats",
            NULL
        };

        printf("  Symbol resolution (dlsym from ANECompiler.framework):\n");
        for (int i = 0; symbols[i]; i++) {
            // Try with underscore prefix removed (C symbols have _ prefix in Mach-O)
            void *sym = dlsym(compiler ?: RTLD_DEFAULT, symbols[i]);
            printf("    %-48s %s\n", symbols[i], sym ? "FOUND" : "---");
        }
        printf("\n");

        // ── Try ANECGetAnalyticsBufferSize ──
        printf("  ── ANECGetAnalyticsBufferSize probe ──\n\n");
        typedef uint64_t (*ANECGetAnalyticsBufferSize_fn)(void *);
        typedef uint64_t (*ANECGetAnalyticsBufferSize_fn0)(void);
        typedef uint64_t (*ANECGetAnalyticsBufferSize_fn2)(void *, void *);

        void *getAbsSym = dlsym(compiler ?: RTLD_DEFAULT, "ANECGetAnalyticsBufferSize");
        if (getAbsSym) {
            printf("    ANECGetAnalyticsBufferSize found at %p\n", getAbsSym);

            // Try calling with NULL
            @try {
                uint64_t sz0 = ((ANECGetAnalyticsBufferSize_fn0)getAbsSym)();
                printf("    ANECGetAnalyticsBufferSize() = %llu\n", sz0);
            } @catch (NSException *ex) {
                printf("    ANECGetAnalyticsBufferSize() threw: %s\n", [[ex reason] UTF8String]);
            }

            // Try calling with NULL arg
            @try {
                uint64_t sz1 = ((ANECGetAnalyticsBufferSize_fn)getAbsSym)(NULL);
                printf("    ANECGetAnalyticsBufferSize(NULL) = %llu\n", sz1);
            } @catch (NSException *ex) {
                printf("    ANECGetAnalyticsBufferSize(NULL) threw: %s\n", [[ex reason] UTF8String]);
            }
        } else {
            printf("    ANECGetAnalyticsBufferSize NOT FOUND\n");
        }
        printf("\n");

        // For C function calls that might crash, use fork() to test safely
        // The child process can crash without killing us
        typedef void (^SafeBlock)(void);
        void (^safe_call)(const char *, SafeBlock) = ^(const char *label, SafeBlock block) {
            printf("    %s: ", label);
            fflush(stdout);
            pid_t pid = fork();
            if (pid == 0) {
                // Child — run the block, print result, exit
                block();
                fflush(stdout);
                _exit(0);
            } else if (pid > 0) {
                int status;
                waitpid(pid, &status, 0);
                if (WIFSIGNALED(status)) {
                    printf("CRASHED (signal %d)\n", WTERMSIG(status));
                }
            } else {
                printf("fork failed\n");
            }
        };

        // ── Try ANECCreateCompilerOptionDictionary ──
        printf("  ── Compiler options probe ──\n\n");
        void *optDictSym = dlsym(compiler ?: RTLD_DEFAULT, "ANECCreateCompilerOptionDictionary");
        void *optCFStrSym = dlsym(compiler ?: RTLD_DEFAULT, "ANECCreateCompilerOptionsCFString");

        if (optDictSym) {
            safe_call("ANECCreateCompilerOptionDictionary()", ^{
                typedef void* (*Fn)(void);
                void *dict = ((Fn)optDictSym)();
                if (dict) {
                    id cfDict = (__bridge id)dict;
                    printf("class=%s ", class_getName(object_getClass(cfDict)));
                    if ([cfDict isKindOfClass:[NSDictionary class]]) {
                        NSDictionary *d = (NSDictionary *)cfDict;
                        printf("keys=%lu\n", (unsigned long)[d count]);
                        for (id key in [[d allKeys] sortedArrayUsingSelector:@selector(compare:)]) {
                            id val = d[key];
                            printf("        %s = %s (%s)\n",
                                [[key description] UTF8String],
                                [[val description] UTF8String],
                                class_getName(object_getClass(val)));
                        }
                    } else {
                        printf("value=%s\n", [[cfDict description] UTF8String]);
                    }
                } else {
                    printf("NULL\n");
                }
            });
        }

        if (optCFStrSym) {
            safe_call("ANECCreateCompilerOptionsCFString()", ^{
                typedef void* (*Fn)(void);
                void *cfstr = ((Fn)optCFStrSym)();
                if (cfstr) {
                    id str = (__bridge id)cfstr;
                    printf("class=%s ", class_getName(object_getClass(str)));
                    NSString *s = (NSString *)str;
                    if ([s length] > 500) {
                        printf("(len=%lu) first 500: %.500s...\n", (unsigned long)[s length], [s UTF8String]);
                    } else {
                        printf("value=%s\n", [s UTF8String]);
                    }
                } else {
                    printf("NULL\n");
                }
            });
        }
        printf("\n");

        // ── Try ANECCreateCompilerInputDictionary ──
        printf("  ── Compiler input dictionary probe ──\n\n");
        void *inputDictSym = dlsym(compiler ?: RTLD_DEFAULT, "ANECCreateCompilerInputDictionary");
        if (inputDictSym) {
            safe_call("ANECCreateCompilerInputDictionary()", ^{
                typedef void* (*Fn)(void);
                void *dict = ((Fn)inputDictSym)();
                if (dict) {
                    id cfDict = (__bridge id)dict;
                    printf("class=%s ", class_getName(object_getClass(cfDict)));
                    if ([cfDict isKindOfClass:[NSDictionary class]]) {
                        NSDictionary *d = (NSDictionary *)cfDict;
                        printf("keys=%lu\n", (unsigned long)[d count]);
                        for (id key in [[d allKeys] sortedArrayUsingSelector:@selector(compare:)]) {
                            id val = d[key];
                            printf("        %s = %s (%s)\n",
                                [[key description] UTF8String],
                                [[val description] UTF8String],
                                class_getName(object_getClass(val)));
                        }
                    }
                } else {
                    printf("NULL\n");
                }
            });
        }
        printf("\n");

        // ── Try ANECCreateDeviceProperty ──
        printf("  ── Device property probe ──\n\n");
        void *createDevSym = dlsym(compiler ?: RTLD_DEFAULT, "ANECCreateDeviceProperty");

        if (createDevSym) {
            safe_call("ANECCreateDeviceProperty()", ^{
                typedef void* (*Fn)(void);
                void *dp = ((Fn)createDevSym)();
                printf("ptr=%p ", dp);
                if (dp) {
                    id obj = (__bridge id)dp;
                    printf("class=%s\n", class_getName(object_getClass(obj)));
                    if ([obj isKindOfClass:[NSDictionary class]]) {
                        NSDictionary *d = (NSDictionary *)obj;
                        for (id key in [[d allKeys] sortedArrayUsingSelector:@selector(compare:)]) {
                            printf("        %s = %s\n",
                                [[key description] UTF8String],
                                [[d[key] description] UTF8String]);
                        }
                    } else {
                        printf("        desc: %s\n", [[obj description] UTF8String]);
                    }
                } else {
                    printf("\n");
                }
            });
        }
        printf("\n");

        // ── Try ANECGetCompilerFileFormat ──
        printf("  ── Compiler file format ──\n\n");
        void *fileFmtSym = dlsym(compiler ?: RTLD_DEFAULT, "ANECGetCompilerFileFormat");
        if (fileFmtSym) {
            safe_call("ANECGetCompilerFileFormat()", ^{
                typedef uint32_t (*Fn)(void);
                uint32_t fmt = ((Fn)fileFmtSym)();
                printf("%u (0x%X)\n", fmt, fmt);
            });
        }
        printf("\n");

        // ── Try ANECPreferredInterleaveFactors ──
        printf("  ── Preferred interleave factors ──\n\n");
        void *interleaveSym = dlsym(compiler ?: RTLD_DEFAULT, "ANECPreferredInterleaveFactors");
        if (interleaveSym) {
            safe_call("ANECPreferredInterleaveFactors()", ^{
                typedef void* (*Fn)(void);
                void *result = ((Fn)interleaveSym)();
                printf("ptr=%p ", result);
                if (result) {
                    id obj = (__bridge id)result;
                    printf("class=%s desc=%s\n", class_getName(object_getClass(obj)),
                        [[obj description] UTF8String]);
                } else {
                    printf("\n");
                }
            });
        }
        printf("\n");

        // ── Try ANECGetMutableProcedureInfoSize / OperationInfoSize ──
        printf("  ── Mutable info sizes ──\n\n");
        void *procInfoSzSym = dlsym(compiler ?: RTLD_DEFAULT, "ANECGetMutableProcedureInfoSize");
        void *opInfoSzSym = dlsym(compiler ?: RTLD_DEFAULT, "ANECGetMutableOperationInfoSize");
        if (procInfoSzSym) {
            safe_call("ANECGetMutableProcedureInfoSize()", ^{
                typedef uint64_t (*Fn)(void);
                uint64_t sz = ((Fn)procInfoSzSym)();
                printf("%llu\n", sz);
            });
        }
        if (opInfoSzSym) {
            safe_call("ANECGetMutableOperationInfoSize()", ^{
                typedef uint64_t (*Fn)(void);
                uint64_t sz = ((Fn)opInfoSzSym)();
                printf("%llu\n", sz);
            });
        }
        printf("\n");

        // ════════════════════════════════════════════════════════════════
        // SECTION B: Compile a model and search for analytics files
        // ════════════════════════════════════════════════════════════════
        printf("  ══════════════════════════════════════════════════════════════\n");
        printf("  SECTION B: Compile model + search for analytics files\n");
        printf("  ══════════════════════════════════════════════════════════════\n\n");

        Class g_D = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class g_I = NSClassFromString(@"_ANEInMemoryModel");

        // Snapshot /tmp before compilation (only track model-related dirs)
        printf("  Snapshotting /tmp before compile...\n");
        NSMutableSet *beforeFiles = [NSMutableSet set];
        {
            NSArray *items = [[NSFileManager defaultManager] contentsOfDirectoryAtPath:NSTemporaryDirectory() error:nil];
            for (NSString *f in items) {
                [beforeFiles addObject:f];
            }
        }
        printf("    %lu top-level items in tmp before\n\n", (unsigned long)[beforeFiles count]);

        // Compile
        NSString *tempDir = nil;
        id mdl = compile_kernel(g_D, g_I, 512, 64, &tempDir);
        printf("    Model: %s\n", mdl ? "compiled+loaded" : "FAILED");
        printf("    Temp dir: %s\n\n", tempDir ? [tempDir UTF8String] : "nil");

        // Snapshot /tmp after compilation (top-level only)
        printf("  New top-level dirs created during compilation:\n");
        {
            NSArray *items = [[NSFileManager defaultManager] contentsOfDirectoryAtPath:NSTemporaryDirectory() error:nil];
            int newCount = 0;
            for (NSString *f in items) {
                if (![beforeFiles containsObject:f]) {
                    printf("    [NEW] %s\n", [f UTF8String]);
                    newCount++;
                    if (newCount > 10) { printf("    ... (truncated)\n"); break; }
                }
            }
            if (newCount == 0) printf("    (none — daemon handles compilation remotely)\n");
        }
        printf("\n");

        // Check model temp dir
        if (tempDir) {
            printf("  Model temp directory contents:\n");
            list_files_recursive([tempDir UTF8String], 0);
            printf("\n");
        }

        // Check daemon cache paths
        printf("  Daemon/system cache paths:\n");
        const char *cachePaths[] = {
            "/private/var/db/anedaemon",
            "/private/var/db/com.apple.aned",
            "/Library/Caches/com.apple.aned",
            "/private/var/folders",
            NULL
        };
        for (int i = 0; cachePaths[i]; i++) {
            DIR *d = opendir(cachePaths[i]);
            if (d) {
                printf("    [EXISTS] %s\n", cachePaths[i]);
                closedir(d);
            } else {
                printf("    [NO]     %s\n", cachePaths[i]);
            }
        }
        printf("\n");

        // Look for .hwx files in common temp locations
        printf("  Searching for .hwx/.analytics files in tmp:\n");
        {
            NSDirectoryEnumerator *en = [[NSFileManager defaultManager]
                enumeratorAtPath:NSTemporaryDirectory()];
            NSString *f;
            int found = 0;
            while ((f = [en nextObject])) {
                if ([[f pathExtension] isEqualToString:@"hwx"] ||
                    [[f pathExtension] isEqualToString:@"analytics"]) {
                    NSString *full = [NSTemporaryDirectory() stringByAppendingPathComponent:f];
                    NSDictionary *attrs = [[NSFileManager defaultManager] attributesOfItemAtPath:full error:nil];
                    printf("    %s (%llu bytes)\n", [f UTF8String],
                        [attrs[NSFileSize] unsignedLongLongValue]);
                    found++;
                    if (found > 20) { printf("    ...\n"); break; }
                }
            }
            if (found == 0) printf("    (none found)\n");
        }

        // Count net.plist files (these are the compiled model outputs)
        printf("  Compiled model plists (net.plist) in tmp:\n");
        {
            NSDirectoryEnumerator *en = [[NSFileManager defaultManager]
                enumeratorAtPath:NSTemporaryDirectory()];
            NSString *f;
            int plistCount = 0;
            NSString *firstPlist = nil;
            while ((f = [en nextObject])) {
                if ([[f lastPathComponent] isEqualToString:@"net.plist"]) {
                    plistCount++;
                    if (!firstPlist) firstPlist = [NSTemporaryDirectory() stringByAppendingPathComponent:f];
                }
            }
            printf("    Found %d net.plist files\n", plistCount);
            if (firstPlist) {
                printf("    Sample: %s\n", [firstPlist UTF8String]);
                // Read and dump the plist
                NSDictionary *plist = [NSDictionary dictionaryWithContentsOfFile:firstPlist];
                if (plist) {
                    printf("    net.plist keys:\n");
                    for (NSString *key in [[plist allKeys] sortedArrayUsingSelector:@selector(compare:)]) {
                        id val = plist[key];
                        NSString *desc = [val description];
                        if ([desc length] > 100) desc = [[desc substringToIndex:100] stringByAppendingString:@"..."];
                        printf("      %s = %s\n", [key UTF8String], [desc UTF8String]);
                    }
                }
            }
        }
        printf("\n");

        // ════════════════════════════════════════════════════════════════
        // SECTION C: Extract analytics from _ANEProgramForEvaluation
        // ════════════════════════════════════════════════════════════════
        printf("  ══════════════════════════════════════════════════════════════\n");
        printf("  SECTION C: _ANEProgramForEvaluation introspection\n");
        printf("  ══════════════════════════════════════════════════════════════\n\n");

        if (mdl) {
            // Dump _ANEProgramForEvaluation class
            dump_class_brief("_ANEProgramForEvaluation");
            printf("\n");

            // Get program from model
            @try {
                // _ANEInMemoryModel._model -> _ANEModel
                Ivar modelIvar = class_getInstanceVariable([mdl class], "_model");
                id aneModel = modelIvar ? object_getIvar(mdl, modelIvar) : nil;
                printf("    _ANEModel: %s (%s)\n",
                    aneModel ? "found" : "nil",
                    aneModel ? class_getName(object_getClass(aneModel)) : "?");

                if (aneModel) {
                    // Dump _ANEModel class
                    printf("\n");
                    dump_class_brief(class_getName(object_getClass(aneModel)));
                    printf("\n");

                    // Try to get all properties from the model
                    printf("    _ANEModel properties (object-returning only):\n");
                    unsigned int mc = 0;
                    Method *meths = class_copyMethodList(object_getClass(aneModel), &mc);
                    for (unsigned int m = 0; m < mc; m++) {
                        SEL sel = method_getName(meths[m]);
                        const char *sn = sel_getName(sel);
                        if (strncmp(sn, "set", 3) == 0 || strcmp(sn, ".cxx_destruct") == 0 ||
                            strcmp(sn, "dealloc") == 0 || strncmp(sn, "init", 4) == 0) continue;
                        int colons = 0;
                        for (const char *p = sn; *p; p++) if (*p == ':') colons++;
                        if (colons > 0) continue;
                        // Only call methods that return objects (type encoding starts with '@')
                        const char *retType = method_getTypeEncoding(meths[m]);
                        if (!retType || retType[0] != '@') {
                            // Print scalar methods with their type
                            if (retType && (retType[0] == 'Q' || retType[0] == 'q' || retType[0] == 'I' || retType[0] == 'i')) {
                                uint64_t val = ((uint64_t(*)(id,SEL))objc_msgSend)(aneModel, sel);
                                printf("      .%s = %llu (scalar)\n", sn, val);
                            } else {
                                printf("      .%s (type: %s) [skipped]\n", sn, retType ?: "?");
                            }
                            continue;
                        }
                        @try {
                            id val = ((id(*)(id,SEL))objc_msgSend)(aneModel, sel);
                            if (val) {
                                NSString *desc = [val description];
                                if ([desc length] > 200) desc = [[desc substringToIndex:200] stringByAppendingString:@"..."];
                                printf("      .%s = %s (%s)\n", sn, [desc UTF8String], class_getName(object_getClass(val)));
                            } else {
                                printf("      .%s = nil\n", sn);
                            }
                        } @catch (NSException *ex) {
                            printf("      .%s -> EXC: %s\n", sn, [[ex reason] UTF8String]);
                        }
                    }
                    free(meths);
                    printf("\n");

                    // Get programHandle -> look for _ANEProgramForEvaluation
                    id program = nil;
                    @try {
                        program = ((id(*)(id,SEL))objc_msgSend)(aneModel, @selector(program));
                    } @catch (NSException *ex) {}
                    if (!program) {
                        // Try through ivars
                        unsigned int ic = 0;
                        Ivar *ivars = class_copyIvarList(object_getClass(aneModel), &ic);
                        for (unsigned int i = 0; i < ic; i++) {
                            const char *name = ivar_getName(ivars[i]);
                            const char *type = ivar_getTypeEncoding(ivars[i]);
                            if (type && type[0] == '@') {
                                id val = object_getIvar(aneModel, ivars[i]);
                                if (val && [val isKindOfClass:NSClassFromString(@"_ANEProgramForEvaluation")]) {
                                    program = val;
                                    printf("    Found _ANEProgramForEvaluation via ivar '%s'\n", name);
                                    break;
                                }
                            }
                        }
                        free(ivars);
                    }

                    if (program) {
                        printf("    _ANEProgramForEvaluation: %s\n", [[program description] UTF8String]);
                        printf("\n    _ANEProgramForEvaluation properties:\n");
                        Class progCls = object_getClass(program);
                        mc = 0;
                        meths = class_copyMethodList(progCls, &mc);
                        for (unsigned int m = 0; m < mc; m++) {
                            SEL sel = method_getName(meths[m]);
                            const char *sn = sel_getName(sel);
                            if (strncmp(sn, "set", 3) == 0 || strcmp(sn, ".cxx_destruct") == 0 ||
                                strcmp(sn, "dealloc") == 0 || strncmp(sn, "init", 4) == 0) continue;
                            int colons = 0;
                            for (const char *p = sn; *p; p++) if (*p == ':') colons++;
                            if (colons > 0) continue;
                            const char *retType = method_getTypeEncoding(meths[m]);
                            if (!retType || retType[0] != '@') {
                                if (retType && (retType[0] == 'Q' || retType[0] == 'q' || retType[0] == 'I')) {
                                    uint64_t val = ((uint64_t(*)(id,SEL))objc_msgSend)(program, sel);
                                    printf("      .%s = %llu (scalar)\n", sn, val);
                                } else {
                                    printf("      .%s (type: %s) [skipped]\n", sn, retType ?: "?");
                                }
                                continue;
                            }
                            @try {
                                id val = ((id(*)(id,SEL))objc_msgSend)(program, sel);
                                if (val) {
                                    NSString *desc = [val description];
                                    if ([desc length] > 200) desc = [[desc substringToIndex:200] stringByAppendingString:@"..."];
                                    printf("      .%s = %s (%s)\n", sn, [desc UTF8String], class_getName(object_getClass(val)));
                                } else {
                                    printf("      .%s = nil\n", sn);
                                }
                            } @catch (NSException *ex) {
                                printf("      .%s -> EXC: %s\n", sn, [[ex reason] UTF8String]);
                            }
                        }
                        free(meths);

                        // Also try getting raw ivars
                        printf("\n    _ANEProgramForEvaluation ivars:\n");
                        unsigned int ic2 = 0;
                        Ivar *ivars2 = class_copyIvarList(progCls, &ic2);
                        for (unsigned int i = 0; i < ic2; i++) {
                            const char *name = ivar_getName(ivars2[i]);
                            const char *type = ivar_getTypeEncoding(ivars2[i]);
                            printf("      @ %s (%s)\n", name, type ?: "?");
                            if (type && type[0] == '@') {
                                @try {
                                    id val = object_getIvar(program, ivars2[i]);
                                    if (val) {
                                        NSString *desc = [val description];
                                        if ([desc length] > 300)
                                            desc = [[desc substringToIndex:300] stringByAppendingString:@"..."];
                                        printf("        = %s (%s)\n", [desc UTF8String], class_getName(object_getClass(val)));
                                    } else {
                                        printf("        = nil\n");
                                    }
                                } @catch (NSException *ex) {
                                    printf("        = EXC\n");
                                }
                            }
                        }
                        free(ivars2);
                    } else {
                        printf("    _ANEProgramForEvaluation: NOT FOUND on model\n");
                    }

                    // Also check _ANEClient ivars for connection details
                    printf("\n    _ANEClient introspection:\n");
                    Ivar connIvar = class_getInstanceVariable([mdl class], "_sharedConnection");
                    id client = connIvar ? object_getIvar(mdl, connIvar) : nil;
                    if (client) {
                        printf("      class: %s\n", class_getName(object_getClass(client)));
                        unsigned int ic3 = 0;
                        Ivar *ivars3 = class_copyIvarList(object_getClass(client), &ic3);
                        for (unsigned int i = 0; i < ic3; i++) {
                            const char *name = ivar_getName(ivars3[i]);
                            const char *type = ivar_getTypeEncoding(ivars3[i]);
                            printf("      @ %s (%s)\n", name, type ?: "?");
                            if (type && type[0] == '@') {
                                @try {
                                    id val = object_getIvar(client, ivars3[i]);
                                    if (val) {
                                        NSString *desc = [val description];
                                        if ([desc length] > 300)
                                            desc = [[desc substringToIndex:300] stringByAppendingString:@"..."];
                                        printf("        = %s (%s)\n", [desc UTF8String], class_getName(object_getClass(val)));
                                    }
                                } @catch (NSException *ex) {}
                            }
                        }
                        free(ivars3);
                    }
                }
            } @catch (NSException *ex) {
                printf("    EXC: %s\n", [[ex reason] UTF8String]);
            }
        }
        printf("\n");

        // ════════════════════════════════════════════════════════════════
        // SECTION D: Construct fake analytics buffers
        // ════════════════════════════════════════════════════════════════
        printf("  ══════════════════════════════════════════════════════════════\n");
        printf("  SECTION D: Construct fake analytics buffers\n");
        printf("  ══════════════════════════════════════════════════════════════\n\n");

        Class analyticsCls = NSClassFromString(@"_ANECompilerAnalytics");
        if (!analyticsCls) {
            printf("    _ANECompilerAnalytics not available\n\n");
        } else {
            // Known struct sizes
            // _AnalyticsProcedureInfo: 48 bytes {IIIIIQIQ}
            // _AnalyticsLayerInfo: 132 bytes {[64c][64c]f}
            // _AnalyticsTaskInfo: 16 bytes {IQ}
            // _AnalyticsGroupInfo: 32 bytes {IQIQ}
            // _AnalyticsData: {II[0c]} — variable-length: header(8 bytes) + data

            // Strategy: The analytics buffer likely has a header followed by procedure info,
            // which references layers, groups, and tasks. The _AnalyticsData struct suggests
            // a format: [uint32 type][uint32 size][data...].

            // First, let's try to understand the buffer format from the parser.
            // _ANECompilerAnalytics has: analyticsBuffer (NSData), populateAnalytics, procedureAnalytics
            // populateAnalytics parses analyticsBuffer into procedureAnalytics array

            // Let's check what stringForAnalyticsType returns for all types
            printf("  ── stringForAnalyticsType (class method) ──\n\n");
            for (unsigned int t = 0; t < 32; t++) {
                @try {
                    id s = ((id(*)(Class,SEL,unsigned int))objc_msgSend)(
                        analyticsCls, @selector(stringForAnalyticsType:), t);
                    if (s && [(NSString *)s length] > 0)
                        printf("    [%2u] = \"%s\"\n", t, [(NSString *)s UTF8String]);
                } @catch (NSException *ex) {}
            }
            printf("\n");

            // Test 1: Minimal buffer — just a header
            printf("  ── Test 1: Empty buffer (0 bytes) ──\n");
            @try {
                id a = ((id(*)(Class,SEL,id))objc_msgSend)(analyticsCls, @selector(objectWithBuffer:), [NSData data]);
                printf("    Result: %s\n", a ? "object created" : "nil");
                if (a) {
                    BOOL pop = ((BOOL(*)(id,SEL))objc_msgSend)(a, @selector(populateAnalytics));
                    printf("    populateAnalytics: %s\n", pop ? "YES" : "NO");
                    id pa = ((id(*)(id,SEL))objc_msgSend)(a, @selector(procedureAnalytics));
                    printf("    procedureAnalytics: %s (count=%lu)\n",
                        pa ? "exists" : "nil",
                        pa ? (unsigned long)[(NSArray *)pa count] : 0UL);
                }
            } @catch (NSException *ex) { printf("    EXC: %s\n", [[ex reason] UTF8String]); }
            printf("\n");

            // Test 2: Buffer with _AnalyticsData header: {type=0, size=0}
            printf("  ── Test 2: Minimal _AnalyticsData header (8 bytes) ──\n");
            @try {
                uint8_t buf[8] = {0};
                // type=1 (Start Time Stamp), size=0
                ((uint32_t*)buf)[0] = 1;
                ((uint32_t*)buf)[1] = 0;
                NSData *data = [NSData dataWithBytes:buf length:8];
                id a = ((id(*)(Class,SEL,id))objc_msgSend)(analyticsCls, @selector(objectWithBuffer:), data);
                printf("    Result: %s\n", a ? "object created" : "nil");
                if (a) {
                    BOOL pop = ((BOOL(*)(id,SEL))objc_msgSend)(a, @selector(populateAnalytics));
                    printf("    populateAnalytics: %s\n", pop ? "YES" : "NO");
                    id pa = ((id(*)(id,SEL))objc_msgSend)(a, @selector(procedureAnalytics));
                    printf("    procedureAnalytics count: %lu\n",
                        pa ? (unsigned long)[(NSArray *)pa count] : 0UL);
                }
            } @catch (NSException *ex) { printf("    EXC: %s\n", [[ex reason] UTF8String]); }
            printf("\n");

            // Test 3: Buffer starting with procedure count
            // Hypothesis: buffer starts with [uint32 numProcedures], then procedure data
            printf("  ── Test 3: Buffer with 1 procedure header ──\n");
            @try {
                // _AnalyticsProcedureInfo = 48 bytes
                // Try: [numProcs=1] [48 bytes of procedure info]
                uint8_t buf[256];
                memset(buf, 0, sizeof(buf));
                ((uint32_t*)buf)[0] = 1;  // 1 procedure
                // Procedure info at offset 4:
                // field0 = proc index = 0
                // field1 = layer count = 1
                // field2 = group count = 1
                // field3 = task count = 1
                ((uint32_t*)(buf+4))[0] = 0;   // proc index
                ((uint32_t*)(buf+4))[1] = 1;   // layer count
                ((uint32_t*)(buf+4))[2] = 1;   // group count
                ((uint32_t*)(buf+4))[3] = 1;   // task count
                // Layer info at offset 52 (4 + 48):
                // 132 bytes: name[64] + type[64] + float
                strcpy((char*)(buf+52), "conv_layer");
                strcpy((char*)(buf+52+64), "conv");
                *(float*)(buf+52+128) = 42.5f;

                NSData *data = [NSData dataWithBytes:buf length:sizeof(buf)];
                id a = ((id(*)(Class,SEL,id))objc_msgSend)(analyticsCls, @selector(objectWithBuffer:), data);
                printf("    Result: %s\n", a ? "object created" : "nil");
                if (a) {
                    BOOL pop = ((BOOL(*)(id,SEL))objc_msgSend)(a, @selector(populateAnalytics));
                    printf("    populateAnalytics: %s\n", pop ? "YES" : "NO");
                    id pa = ((id(*)(id,SEL))objc_msgSend)(a, @selector(procedureAnalytics));
                    printf("    procedureAnalytics count: %lu\n",
                        pa ? (unsigned long)[(NSArray *)pa count] : 0UL);
                    if (pa && [(NSArray *)pa count] > 0) {
                        for (id proc in (NSArray *)pa) {
                            printf("    Procedure: %s\n", [[proc description] UTF8String]);
                            @try {
                                id ident = ((id(*)(id,SEL))objc_msgSend)(proc, @selector(identifier));
                                printf("      .identifier = %s\n", ident ? [[ident description] UTF8String] : "nil");
                                id metrics = ((id(*)(id,SEL))objc_msgSend)(proc, @selector(procedureMetrics));
                                printf("      .procedureMetrics = %s\n", metrics ? [[metrics description] UTF8String] : "nil");
                                id groups = ((id(*)(id,SEL))objc_msgSend)(proc, @selector(groupInfo));
                                printf("      .groupInfo count = %lu\n",
                                    groups ? (unsigned long)[(NSArray *)groups count] : 0UL);
                                if (groups) {
                                    for (id grp in (NSArray *)groups) {
                                        id layers = ((id(*)(id,SEL))objc_msgSend)(grp, @selector(layerInfo));
                                        printf("        Group layers count = %lu\n",
                                            layers ? (unsigned long)[(NSArray *)layers count] : 0UL);
                                        if (layers) {
                                            for (id layer in (NSArray *)layers) {
                                                id ln = ((id(*)(id,SEL))objc_msgSend)(layer, @selector(layerName));
                                                // weight is a float
                                                float wt = ((float(*)(id,SEL))objc_msgSend)(layer, @selector(weight));
                                                printf("          layer: %s  weight=%.3f\n",
                                                    ln ? [[ln description] UTF8String] : "nil", wt);
                                            }
                                        }
                                    }
                                }
                            } @catch (NSException *ex) {
                                printf("      EXC: %s\n", [[ex reason] UTF8String]);
                            }
                        }
                    }
                }
            } @catch (NSException *ex) { printf("    EXC: %s\n", [[ex reason] UTF8String]); }
            printf("\n");

            // Test 4: Try different buffer format — maybe it starts with _AnalyticsData structs
            // {II[0c]} means: [uint32 analyticsType] [uint32 dataSize] [char data[]]
            // The buffer might be a sequence of these entries
            printf("  ── Test 4: Buffer as sequence of _AnalyticsData entries ──\n");
            @try {
                NSMutableData *buf = [NSMutableData data];

                // Entry: DRAMTraffic (type 3), value = 1234.0
                {
                    uint32_t type = 3; // DRAMTraffic
                    uint32_t size = sizeof(float);
                    float val = 1234.0f;
                    [buf appendBytes:&type length:4];
                    [buf appendBytes:&size length:4];
                    [buf appendBytes:&val length:sizeof(float)];
                }
                // Entry: L2Traffic (type 4)
                {
                    uint32_t type = 4;
                    uint32_t size = sizeof(float);
                    float val = 567.0f;
                    [buf appendBytes:&type length:4];
                    [buf appendBytes:&size length:4];
                    [buf appendBytes:&val length:sizeof(float)];
                }
                // Entry: ViolatesMaxLatency (type 11)
                {
                    uint32_t type = 11;
                    uint32_t size = sizeof(uint32_t);
                    uint32_t val = 1;
                    [buf appendBytes:&type length:4];
                    [buf appendBytes:&size length:4];
                    [buf appendBytes:&val length:4];
                }

                id a = ((id(*)(Class,SEL,id))objc_msgSend)(analyticsCls, @selector(objectWithBuffer:), buf);
                printf("    Result: %s\n", a ? "object created" : "nil");
                if (a) {
                    BOOL pop = ((BOOL(*)(id,SEL))objc_msgSend)(a, @selector(populateAnalytics));
                    printf("    populateAnalytics: %s\n", pop ? "YES" : "NO");
                    id pa = ((id(*)(id,SEL))objc_msgSend)(a, @selector(procedureAnalytics));
                    printf("    procedureAnalytics count: %lu\n",
                        pa ? (unsigned long)[(NSArray *)pa count] : 0UL);
                    if (pa && [(NSArray *)pa count] > 0) {
                        printf("    GOT PROCEDURES! Dumping...\n");
                        for (id proc in (NSArray *)pa) {
                            printf("      %s\n", [[proc description] UTF8String]);
                        }
                    }
                    // Check serialize
                    id ser = ((id(*)(id,SEL))objc_msgSend)(a, @selector(serialize));
                    printf("    serialize: %s\n", ser ? [[ser description] UTF8String] : "nil");
                }
            } @catch (NSException *ex) { printf("    EXC: %s\n", [[ex reason] UTF8String]); }
            printf("\n");

            // Test 5: Large buffer — maybe it needs a minimum size
            // ANECGetAnalyticsBufferSize might tell us
            printf("  ── Test 5: Large zero-filled buffer (16KB) ──\n");
            @try {
                NSMutableData *buf = [NSMutableData dataWithLength:16384];
                id a = ((id(*)(Class,SEL,id))objc_msgSend)(analyticsCls, @selector(objectWithBuffer:), buf);
                printf("    Result: %s\n", a ? "object created" : "nil");
                if (a) {
                    BOOL pop = ((BOOL(*)(id,SEL))objc_msgSend)(a, @selector(populateAnalytics));
                    printf("    populateAnalytics: %s\n", pop ? "YES" : "NO");
                    id pa = ((id(*)(id,SEL))objc_msgSend)(a, @selector(procedureAnalytics));
                    printf("    procedureAnalytics count: %lu\n",
                        pa ? (unsigned long)[(NSArray *)pa count] : 0UL);
                }
            } @catch (NSException *ex) { printf("    EXC: %s\n", [[ex reason] UTF8String]); }
            printf("\n");

            // Test 6: Buffer with procedure info that has proper fields
            // Hypothesis: buffer = [numProcedures:uint32] [_AnalyticsProcedureInfo * N]
            //   followed by layer/group/task data referenced by counts in procedure info
            printf("  ── Test 6: Structured buffer (1 proc, 1 layer, 1 group, 1 task) ──\n");
            @try {
                NSMutableData *buf = [NSMutableData data];

                // Header: number of procedures
                uint32_t numProcs = 1;
                [buf appendBytes:&numProcs length:4];

                // _AnalyticsProcedureInfo (48 bytes): {IIIIIQIQ}
                // Padding: I=4, Q=8 → {I,I,I,I,I,pad4,Q,I,pad4,Q} = 4+4+4+4+4+4+8+4+4+8 = 48
                uint32_t procBuf[12]; // 48 bytes
                memset(procBuf, 0, 48);
                procBuf[0] = 0;   // proc index
                procBuf[1] = 1;   // layer count
                procBuf[2] = 1;   // group count
                procBuf[3] = 1;   // task count
                procBuf[4] = 0;   // field4
                // field5 (uint64) at offset 24: procBuf[6..7]
                // field6 (uint32) at offset 32: procBuf[8]
                // field7 (uint64) at offset 40: procBuf[10..11]
                [buf appendBytes:procBuf length:48];

                // _AnalyticsLayerInfo (132 bytes): name[64] + type[64] + float
                uint8_t layerBuf[132];
                memset(layerBuf, 0, 132);
                strcpy((char*)layerBuf, "test_conv");
                strcpy((char*)(layerBuf+64), "convolution");
                *(float*)(layerBuf+128) = 99.5f;
                [buf appendBytes:layerBuf length:132];

                // _AnalyticsGroupInfo (32 bytes): {IQIQ}
                // {I,pad4,Q,I,pad4,Q} = 4+4+8+4+4+8 = 32
                uint8_t groupBuf[32];
                memset(groupBuf, 0, 32);
                *(uint32_t*)(groupBuf+0) = 0;  // groupId
                *(uint64_t*)(groupBuf+8) = 100; // field1
                *(uint32_t*)(groupBuf+16) = 1;  // field2 (num layers in group?)
                *(uint64_t*)(groupBuf+24) = 0;  // field3
                [buf appendBytes:groupBuf length:32];

                // _AnalyticsTaskInfo (16 bytes): {IQ}
                // {I,pad4,Q} = 4+4+8 = 16
                uint8_t taskBuf[16];
                memset(taskBuf, 0, 16);
                *(uint32_t*)(taskBuf+0) = 42;  // taskId
                *(uint64_t*)(taskBuf+8) = 1000; // metric
                [buf appendBytes:taskBuf length:16];

                printf("    Buffer size: %lu bytes\n", (unsigned long)[buf length]);
                id a = ((id(*)(Class,SEL,id))objc_msgSend)(analyticsCls, @selector(objectWithBuffer:), buf);
                printf("    Result: %s\n", a ? "object created" : "nil");
                if (a) {
                    BOOL pop = ((BOOL(*)(id,SEL))objc_msgSend)(a, @selector(populateAnalytics));
                    printf("    populateAnalytics: %s\n", pop ? "YES" : "NO");
                    id pa = ((id(*)(id,SEL))objc_msgSend)(a, @selector(procedureAnalytics));
                    printf("    procedureAnalytics: %s (count=%lu)\n",
                        pa ? "exists" : "nil",
                        pa ? (unsigned long)[(NSArray *)pa count] : 0UL);
                    if (pa && [(NSArray *)pa count] > 0) {
                        printf("    === BUFFER FORMAT DECODED! ===\n");
                        for (id proc in (NSArray *)pa) {
                            @try {
                                id ident = ((id(*)(id,SEL))objc_msgSend)(proc, @selector(identifier));
                                id metrics = ((id(*)(id,SEL))objc_msgSend)(proc, @selector(procedureMetrics));
                                id groups = ((id(*)(id,SEL))objc_msgSend)(proc, @selector(groupInfo));
                                printf("      ident: %s\n", ident ? [[ident description] UTF8String] : "nil");
                                printf("      metrics: %s\n", metrics ? [[metrics description] UTF8String] : "nil");
                                printf("      groups: %lu\n", groups ? (unsigned long)[(NSArray *)groups count] : 0UL);
                                if (groups) {
                                    for (id grp in (NSArray *)groups) {
                                        id gid = ((id(*)(id,SEL))objc_msgSend)(grp, @selector(groupID));
                                        id layers = ((id(*)(id,SEL))objc_msgSend)(grp, @selector(layerInfo));
                                        id tasks = ((id(*)(id,SEL))objc_msgSend)(grp, @selector(taskInfo));
                                        printf("        groupID: %s  layers: %lu  tasks: %lu\n",
                                            gid ? [[gid description] UTF8String] : "nil",
                                            layers ? (unsigned long)[(NSArray *)layers count] : 0UL,
                                            tasks ? (unsigned long)[(NSArray *)tasks count] : 0UL);
                                        if (layers) {
                                            for (id layer in (NSArray *)layers) {
                                                id ln = ((id(*)(id,SEL))objc_msgSend)(layer, @selector(layerName));
                                                float wt = ((float(*)(id,SEL))objc_msgSend)(layer, @selector(weight));
                                                printf("          layer: '%s'  weight=%.3f\n",
                                                    ln ? [[ln description] UTF8String] : "nil", wt);
                                            }
                                        }
                                        if (tasks) {
                                            for (id task in (NSArray *)tasks) {
                                                id tm = ((id(*)(id,SEL))objc_msgSend)(task, @selector(metrics));
                                                printf("          task metrics: %s\n",
                                                    tm ? [[tm description] UTF8String] : "nil");
                                            }
                                        }
                                    }
                                }
                            } @catch (NSException *ex) {
                                printf("      EXC: %s\n", [[ex reason] UTF8String]);
                            }
                        }
                    }

                    // Try serialize to see what the canonical format looks like
                    id ser = ((id(*)(id,SEL))objc_msgSend)(a, @selector(serialize));
                    if (ser && [ser isKindOfClass:[NSData class]]) {
                        NSData *serData = (NSData *)ser;
                        printf("    serialize: %lu bytes\n", (unsigned long)[serData length]);
                        // Hex dump first 128 bytes
                        const uint8_t *bytes = [serData bytes];
                        printf("    hex dump:\n      ");
                        for (NSUInteger i = 0; i < MIN([serData length], 128); i++) {
                            printf("%02x ", bytes[i]);
                            if ((i + 1) % 16 == 0) printf("\n      ");
                        }
                        printf("\n");
                    } else {
                        printf("    serialize: %s\n", ser ? [[ser description] UTF8String] : "nil");
                    }
                }
            } @catch (NSException *ex) { printf("    EXC: %s\n", [[ex reason] UTF8String]); }
            printf("\n");

            // Test 7: Try alternate format — buffer might start with a magic/version
            printf("  ── Test 7: Buffer with magic/version header ──\n");
            @try {
                NSMutableData *buf = [NSMutableData data];

                // Maybe: [magic:uint32] [version:uint32] [numProcedures:uint32] [data...]
                uint32_t magic = 0x414E4543; // "ANEC"
                uint32_t version = 1;
                uint32_t numProcs = 1;
                [buf appendBytes:&magic length:4];
                [buf appendBytes:&version length:4];
                [buf appendBytes:&numProcs length:4];

                // Procedure info (48 bytes)
                uint8_t procBuf[48];
                memset(procBuf, 0, 48);
                ((uint32_t*)procBuf)[1] = 1; // 1 layer
                ((uint32_t*)procBuf)[2] = 1; // 1 group
                ((uint32_t*)procBuf)[3] = 1; // 1 task
                [buf appendBytes:procBuf length:48];

                // Layer (132)
                uint8_t layerBuf[132];
                memset(layerBuf, 0, 132);
                strcpy((char*)layerBuf, "magic_conv");
                *(float*)(layerBuf+128) = 77.0f;
                [buf appendBytes:layerBuf length:132];

                // Group (32)
                uint8_t groupBuf[32];
                memset(groupBuf, 0, 32);
                [buf appendBytes:groupBuf length:32];

                // Task (16)
                uint8_t taskBuf[16];
                memset(taskBuf, 0, 16);
                [buf appendBytes:taskBuf length:16];

                id a = ((id(*)(Class,SEL,id))objc_msgSend)(analyticsCls, @selector(objectWithBuffer:), buf);
                printf("    Result: %s\n", a ? "object created" : "nil");
                if (a) {
                    BOOL pop = ((BOOL(*)(id,SEL))objc_msgSend)(a, @selector(populateAnalytics));
                    printf("    populateAnalytics: %s\n", pop ? "YES" : "NO");
                    id pa = ((id(*)(id,SEL))objc_msgSend)(a, @selector(procedureAnalytics));
                    printf("    procedureAnalytics count: %lu\n",
                        pa ? (unsigned long)[(NSArray *)pa count] : 0UL);
                    if (pa && [(NSArray *)pa count] > 0) {
                        printf("    === MAGIC HEADER FORMAT WORKS! ===\n");
                    }
                }
            } @catch (NSException *ex) { printf("    EXC: %s\n", [[ex reason] UTF8String]); }
            printf("\n");

            // Test 8: Try NSKeyedArchiver format — maybe it's a serialized plist
            printf("  ── Test 8: Try NSKeyedArchiver serialization ──\n");
            @try {
                // Create a proper analytics object and see if we can set its buffer
                id a = [[analyticsCls alloc] init];
                if (a) {
                    printf("    Created _ANECompilerAnalytics via alloc/init\n");

                    // Try to set analyticsBuffer directly via setValue:forKey:
                    NSMutableData *fakeBuf = [NSMutableData dataWithLength:1024];
                    @try {
                        [a setValue:fakeBuf forKey:@"analyticsBuffer"];
                        printf("    Set analyticsBuffer via KVC: OK\n");
                        id gotBuf = [a valueForKey:@"analyticsBuffer"];
                        printf("    Get analyticsBuffer: %lu bytes\n",
                            gotBuf ? (unsigned long)[(NSData *)gotBuf length] : 0UL);

                        BOOL pop = ((BOOL(*)(id,SEL))objc_msgSend)(a, @selector(populateAnalytics));
                        printf("    populateAnalytics after setting buffer: %s\n", pop ? "YES" : "NO");
                    } @catch (NSException *ex) {
                        printf("    KVC set analyticsBuffer EXC: %s\n", [[ex reason] UTF8String]);
                    }

                    // Check all ivars of the analytics object
                    printf("    Ivar values after init:\n");
                    unsigned int ic = 0;
                    Ivar *ivars = class_copyIvarList(analyticsCls, &ic);
                    for (unsigned int i = 0; i < ic; i++) {
                        const char *name = ivar_getName(ivars[i]);
                        const char *type = ivar_getTypeEncoding(ivars[i]);
                        printf("      @ %s (%s)", name, type ?: "?");
                        if (type && type[0] == '@') {
                            @try {
                                id val = object_getIvar(a, ivars[i]);
                                printf(" = %s", val ? [[val description] UTF8String] : "nil");
                            } @catch (NSException *ex) { printf(" = EXC"); }
                        }
                        printf("\n");
                    }
                    free(ivars);
                }
            } @catch (NSException *ex) { printf("    EXC: %s\n", [[ex reason] UTF8String]); }
            printf("\n");
        }

        // ════════════════════════════════════════════════════════════════
        // SECTION E: Try compiling with analytics-triggering options
        // ════════════════════════════════════════════════════════════════
        printf("  ══════════════════════════════════════════════════════════════\n");
        printf("  SECTION E: Compile with kANEFPerformanceStatsMask + options\n");
        printf("  ══════════════════════════════════════════════════════════════\n\n");

        if (g_D && g_I) {
            // Try compiling with various option keys that might trigger analytics
            NSDictionary *optionSets[] = {
                @{@"kANEFPerformanceStatsMask": @(0xFFFFFFFF)},
                @{@"PerformanceStatsMask": @(0xFFFFFFFF)},
                @{@"ANECAnalytics": @YES},
                @{@"CompilerAnalytics": @YES},
                @{@"EnableAnalytics": @YES, @"PerformanceStatsMask": @(0xFFFF)},
            };
            const char *optionNames[] = {
                "kANEFPerformanceStatsMask=0xFFFFFFFF",
                "PerformanceStatsMask=0xFFFFFFFF",
                "ANECAnalytics=YES",
                "CompilerAnalytics=YES",
                "EnableAnalytics+PerformanceStatsMask",
            };

            for (int o = 0; o < 5; o++) {
                printf("  ── Options: %s ──\n", optionNames[o]);
                @try {
                    _Float16 *w = (_Float16*)calloc(256*256, sizeof(_Float16));
                    for (int i = 0; i < 256; i++) w[i*256+i] = (_Float16)1.0f;
                    int ws = 256*256*2, tot = 128+ws;
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
                        "    func main<ios18>(tensor<fp32, [1, 256, 1, 32]> x) {\n"
                        "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];\n"
                        "        tensor<fp16, [1,256,1,32]> x16 = cast(dtype=to16,x=x)[name=string(\"cin\")];\n"
                        "        tensor<fp16, [256,256,1,1]> W = const()[name=string(\"W\"), "
                        "val=tensor<fp16, [256,256,1,1]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(64)))];\n"
                        "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
                        "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
                        "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
                        "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
                        "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
                        "        tensor<fp16, [1,256,1,32]> y16 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W,x=x16)"
                        "[name=string(\"conv\")];\n"
                        "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];\n"
                        "        tensor<fp32, [1,256,1,32]> y = cast(dtype=to32,x=y16)[name=string(\"cout\")];\n"
                        "    } -> (y);\n"
                        "}\n"];

                    NSData *md = [mil dataUsingEncoding:NSUTF8StringEncoding];
                    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D, @selector(modelWithMILText:weights:optionsPlist:),
                        md, @{@"@model_path/weights/weight.bin": @{@"offset":@0, @"data":wdata}}, nil);
                    id mdl2 = ((id(*)(Class,SEL,id))objc_msgSend)(g_I, @selector(inMemoryModelWithDescriptor:), desc);
                    id hx = ((id(*)(id,SEL))objc_msgSend)(mdl2, @selector(hexStringIdentifier));
                    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
                    [[NSFileManager defaultManager] createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
                        withIntermediateDirectories:YES attributes:nil error:nil];
                    [md writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
                    [wdata writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

                    NSError *e = nil;
                    BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                        mdl2, @selector(compileWithQoS:options:error:), 21, optionSets[o], &e);
                    printf("    compile: %s\n", ok ? "OK" : "FAILED");
                    if (!ok && e) printf("    error: %s\n", [[e localizedDescription] UTF8String]);

                    if (ok) {
                        e = nil;
                        BOOL loadOk = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                            mdl2, @selector(loadWithQoS:options:error:), 21, @{}, &e);
                        printf("    load: %s\n", loadOk ? "OK" : "FAILED");

                        if (loadOk) {
                            // Check for analytics in modelAttributes
                            Ivar mi = class_getInstanceVariable([mdl2 class], "_model");
                            id am = mi ? object_getIvar(mdl2, mi) : nil;
                            if (am) {
                                NSDictionary *attrs = ((id(*)(id,SEL))objc_msgSend)(am, @selector(modelAttributes));
                                // Look for any analytics-related keys
                                for (NSString *key in [attrs allKeys]) {
                                    NSString *lk = [key lowercaseString];
                                    if ([lk containsString:@"analytic"] || [lk containsString:@"perf"] ||
                                        [lk containsString:@"stat"] || [lk containsString:@"latenc"] ||
                                        [lk containsString:@"traffic"] || [lk containsString:@"dram"] ||
                                        [lk containsString:@"sram"] || [lk containsString:@"spill"]) {
                                        printf("    ANALYTICS KEY: %s = %s\n",
                                            [key UTF8String], [[attrs[key] description] UTF8String]);
                                    }
                                }
                                // Show all keys for this option set
                                printf("    modelAttributes keys: ");
                                for (NSString *key in [[attrs allKeys] sortedArrayUsingSelector:@selector(compare:)]) {
                                    printf("%s ", [key UTF8String]);
                                }
                                printf("\n");
                            }

                            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
                                mdl2, @selector(unloadWithQoS:error:), 21, &e);
                        }
                    }
                } @catch (NSException *ex) {
                    printf("    EXC: %s\n", [[ex reason] UTF8String]);
                }
                printf("\n");
            }
        }

        // ════════════════════════════════════════════════════════════════
        // SECTION F: Espresso profiling — compiler_analytics_file_names
        // ════════════════════════════════════════════════════════════════
        printf("  ══════════════════════════════════════════════════════════════\n");
        printf("  SECTION F: Espresso profiling analytics\n");
        printf("  ══════════════════════════════════════════════════════════════\n\n");

        Class espAnaCls = NSClassFromString(@"EspressoProfilingANEcompilerAnalytics");
        if (espAnaCls) {
            dump_class_brief("EspressoProfilingANEcompilerAnalytics");
            printf("\n");

            @try {
                id inst = [[espAnaCls alloc] init];
                if (inst) {
                    printf("    Created instance\n");
                    // Get all property values
                    @try {
                        id fnames = ((id(*)(id,SEL))objc_msgSend)(inst, @selector(compiler_analytics_file_names));
                        printf("    .compiler_analytics_file_names = %s\n",
                            fnames ? [[fnames description] UTF8String] : "nil");
                    } @catch (NSException *ex) {}

                    @try {
                        id aneData = ((id(*)(id,SEL))objc_msgSend)(inst, @selector(ane_compiler_analytics_data));
                        printf("    .ane_compiler_analytics_data = %s\n",
                            aneData ? [[aneData description] UTF8String] : "nil");
                    } @catch (NSException *ex) {}

                    // Try all zero-arg getters
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
                            printf("    .%s = %s\n", sn,
                                val ? [[val description] UTF8String] : "nil");
                        } @catch (NSException *ex) {
                            printf("    .%s = EXC: %s\n", sn, [[ex reason] UTF8String]);
                        }
                    }
                    free(meths);
                }
            } @catch (NSException *ex) {
                printf("    EXC: %s\n", [[ex reason] UTF8String]);
            }
        }

        // Also check EspressoProfilingNetworkANEInfo
        printf("\n");
        Class espNetAneCls = NSClassFromString(@"EspressoProfilingNetworkANEInfo");
        if (espNetAneCls) {
            dump_class_brief("EspressoProfilingNetworkANEInfo");
            printf("\n");
            @try {
                id inst = [[espNetAneCls alloc] init];
                if (inst) {
                    unsigned int mc = 0;
                    Method *meths = class_copyMethodList(espNetAneCls, &mc);
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
                            printf("    .%s = %s\n", sn,
                                val ? [[val description] UTF8String] : "nil");
                        } @catch (NSException *ex) {
                            printf("    .%s = EXC: %s\n", sn, [[ex reason] UTF8String]);
                        }
                    }
                    free(meths);
                }
            } @catch (NSException *ex) {}
        }
        printf("\n");

        // ════════════════════════════════════════════════════════════════
        // SECTION G: Try calling ANECCompile directly
        // ════════════════════════════════════════════════════════════════
        printf("  ══════════════════════════════════════════════════════════════\n");
        printf("  SECTION G: Direct ANECCompile call attempt\n");
        printf("  ══════════════════════════════════════════════════════════════\n\n");

        // ANECCompile signature (from reverse engineering):
        // int ANECCompile(CFDictionaryRef model, CFDictionaryRef options, CFDictionaryRef *output)
        // or possibly: int ANECCompile(const void *inputData, size_t inputSize,
        //                              void **outputData, size_t *outputSize,
        //                              void *analyticsBuffer, size_t *analyticsSize,
        //                              CFDictionaryRef options)

        void *compileSym = dlsym(compiler ?: RTLD_DEFAULT, "ANECCompile");
        if (compileSym) {
            printf("    ANECCompile found at %p\n\n", compileSym);

            // Try signature 1: int ANECCompile(CFDictionaryRef model, CFDictionaryRef options, CFDictionaryRef *output)
            printf("    Trying: ANECCompile(modelDict, optionsDict, &output)\n");
            @try {
                void *modelDictSym = dlsym(compiler ?: RTLD_DEFAULT, "ANECCreateModelDictionary");
                if (modelDictSym) {
                    typedef void* (*CreateModelDict_fn)(void);
                    void *modelDict = ((CreateModelDict_fn)modelDictSym)();
                    printf("    ANECCreateModelDictionary() = %p\n", modelDict);
                    if (modelDict) {
                        id md = (__bridge id)modelDict;
                        printf("      class: %s\n", class_getName(object_getClass(md)));
                        if ([md isKindOfClass:[NSDictionary class]]) {
                            printf("      keys: ");
                            for (id key in [(NSDictionary *)md allKeys]) {
                                printf("%s ", [[key description] UTF8String]);
                            }
                            printf("\n");
                        }
                    }
                }
            } @catch (NSException *ex) {
                printf("    ANECCreateModelDictionary EXC: %s\n", [[ex reason] UTF8String]);
            }
        } else {
            printf("    ANECCompile NOT FOUND\n");
        }
        printf("\n");

        // ════════════════════════════════════════════════════════════════
        // SECTION H: Check _ANEDataReporter and _ANEProcedureData
        // ════════════════════════════════════════════════════════════════
        printf("  ══════════════════════════════════════════════════════════════\n");
        printf("  SECTION H: _ANEDataReporter and _ANEProcedureData\n");
        printf("  ══════════════════════════════════════════════════════════════\n\n");

        dump_class_brief("_ANEDataReporter");
        printf("\n");
        dump_class_brief("_ANEProcedureData");
        printf("\n");

        // Try to instantiate _ANEDataReporter
        Class drCls = NSClassFromString(@"_ANEDataReporter");
        if (drCls) {
            @try {
                id dr = [[drCls alloc] init];
                printf("    _ANEDataReporter created: %s\n", dr ? "YES" : "nil");
                if (dr) {
                    unsigned int mc = 0;
                    Method *meths = class_copyMethodList(drCls, &mc);
                    for (unsigned int m = 0; m < mc; m++) {
                        SEL sel = method_getName(meths[m]);
                        const char *sn = sel_getName(sel);
                        if (strncmp(sn, "set", 3) == 0 || strcmp(sn, ".cxx_destruct") == 0 ||
                            strcmp(sn, "dealloc") == 0 || strncmp(sn, "init", 4) == 0) continue;
                        int colons = 0;
                        for (const char *p = sn; *p; p++) if (*p == ':') colons++;
                        if (colons > 0) continue;
                        @try {
                            id val = ((id(*)(id,SEL))objc_msgSend)(dr, sel);
                            printf("      .%s = %s\n", sn,
                                val ? [[val description] UTF8String] : "nil");
                        } @catch (NSException *ex) {
                            printf("      .%s = EXC\n", sn);
                        }
                    }
                    free(meths);
                }
            } @catch (NSException *ex) {}
        }
        printf("\n");

        // Unload the test model
        if (mdl) {
            NSError *e = nil;
            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
                mdl, @selector(unloadWithQoS:error:), 21, &e);
        }

        // ════════════════════════════════════════════════════════════════
        // SECTION I: Scan all _ANE* classes for analytics/buffer/data references
        // ════════════════════════════════════════════════════════════════
        printf("  ══════════════════════════════════════════════════════════════\n");
        printf("  SECTION I: Scan ALL _ANE* classes for analytics methods\n");
        printf("  ══════════════════════════════════════════════════════════════\n\n");

        {
            unsigned int cc = 0;
            Class *classes = objc_copyClassList(&cc);
            for (unsigned int i = 0; i < cc; i++) {
                const char *cn = class_getName(classes[i]);
                if (strncmp(cn, "_ANE", 4) != 0) continue;

                unsigned int mc = 0;
                Method *meths = class_copyMethodList(classes[i], &mc);
                for (unsigned int m = 0; m < mc; m++) {
                    const char *sn = sel_getName(method_getName(meths[m]));
                    // Search for analytics/buffer/stats/perf in method names
                    if (strcasestr(sn, "analytic") || strcasestr(sn, "buffer") ||
                        strcasestr(sn, "stats") || strcasestr(sn, "perf") ||
                        strcasestr(sn, "latenc") || strcasestr(sn, "traffic") ||
                        strcasestr(sn, "spill") || strcasestr(sn, "sram")) {
                        printf("    %s - %s\n", cn, sn);
                    }
                }
                free(meths);

                // Also check class methods
                Class meta = object_getClass((id)classes[i]);
                meths = class_copyMethodList(meta, &mc);
                for (unsigned int m = 0; m < mc; m++) {
                    const char *sn = sel_getName(method_getName(meths[m]));
                    if (strcasestr(sn, "analytic") || strcasestr(sn, "buffer") ||
                        strcasestr(sn, "stats") || strcasestr(sn, "perf") ||
                        strcasestr(sn, "latenc") || strcasestr(sn, "traffic") ||
                        strcasestr(sn, "spill") || strcasestr(sn, "sram")) {
                        printf("    %s + %s\n", cn, sn);
                    }
                }
                free(meths);
            }
            free(classes);
        }
        printf("\n");

        printf("  ╔══════════════════════════════════════════════════════════════╗\n");
        printf("  ║     PROBE COMPLETE                                          ║\n");
        printf("  ╚══════════════════════════════════════════════════════════════╝\n\n");
    }
    return 0;
}
