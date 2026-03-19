// test_analytics_profiling.m — Probe ANE analytics via env vars, profiling flags, compile options
// Goal: Find a way to trigger analytics output without XPC swizzling
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <mach-o/dyld.h>
#import <mach-o/getsect.h>
#import <sys/stat.h>
#import <dirent.h>

// ── Utility: dump all methods/properties of a class ──
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
    objc_property_t *props = class_copyPropertyList(cls, &count);
    for (unsigned int i = 0; i < count; i++) {
        const char *pname = property_getName(props[i]);
        const char *pattrs = property_getAttributes(props[i]);
        printf("    P %s  (%s)\n", pname, pattrs ?: "?");
    }
    free(props);
    printf("\n");
}

// ── List files in a directory matching a pattern ──
static NSArray<NSString*> *list_files(NSString *dir, NSString *pattern) {
    NSMutableArray *results = [NSMutableArray array];
    NSFileManager *fm = [NSFileManager defaultManager];
    NSError *err = nil;
    NSArray *contents = [fm contentsOfDirectoryAtPath:dir error:&err];
    if (!contents) return results;
    for (NSString *f in contents) {
        if (!pattern || [f rangeOfString:pattern options:NSCaseInsensitiveSearch].location != NSNotFound) {
            [results addObject:[dir stringByAppendingPathComponent:f]];
        }
    }
    return results;
}

// ── Snapshot directory for change detection ──
static NSDictionary<NSString*,NSDate*> *snapshot_dir(NSString *dir) {
    NSMutableDictionary *snap = [NSMutableDictionary dictionary];
    NSFileManager *fm = [NSFileManager defaultManager];
    NSError *err = nil;
    NSArray *contents = [fm contentsOfDirectoryAtPath:dir error:&err];
    for (NSString *f in contents) {
        NSString *path = [dir stringByAppendingPathComponent:f];
        NSDictionary *attrs = [fm attributesOfItemAtPath:path error:nil];
        if (attrs) snap[path] = attrs[NSFileModificationDate];
    }
    return snap;
}

// ── Diff two snapshots ──
static NSArray<NSString*> *diff_snapshots(NSDictionary *before, NSDictionary *after) {
    NSMutableArray *newFiles = [NSMutableArray array];
    for (NSString *path in after) {
        if (!before[path]) {
            [newFiles addObject:path];
        }
    }
    return newFiles;
}

// ── Compile a kernel, return model + temp directory ──
static id compile_kernel(Class g_D, Class g_I, int CH, int SP, NSData **outWData,
                         NSString **outTempDir, NSDictionary *compileOpts) {
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
    *outTempDir = td;
    [[NSFileManager defaultManager] createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
        withIntermediateDirectories:YES attributes:nil error:nil];
    [md writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    [wdata writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

    NSError *e = nil;
    NSDictionary *opts = compileOpts ?: @{};
    BOOL compOk = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl,
        @selector(compileWithQoS:options:error:), 21, opts, &e);
    if (!compOk) {
        printf("    compile FAILED: %s\n", e ? [[e description] UTF8String] : "unknown");
        return nil;
    }
    e = nil;
    ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
    return mdl;
}

// ── Unload helper ──
static void unload_model(id mdl) {
    if (!mdl) return;
    NSError *ue = nil;
    ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl, @selector(unloadWithQoS:error:), 21, &ue);
}

int main() {
    @autoreleasepool {
        setbuf(stdout, NULL);

        printf("\n  ██████ ANE ANALYTICS PROFILING PROBE ██████\n\n");

        // ════════════════════════════════════════════════════════════════
        // Section A: Search for profiling-related strings in ANE framework
        // ════════════════════════════════════════════════════════════════
        printf("  ══ A: Framework string search ══\n\n");

        void *ane_handle = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
        printf("  ANE framework: %s\n", ane_handle ? "loaded" : "FAILED");

        void *esp_handle = dlopen("/System/Library/PrivateFrameworks/Espresso.framework/Espresso", RTLD_NOW);
        printf("  Espresso framework: %s\n\n", esp_handle ? "loaded" : "FAILED");

        // Search for string constants in loaded images
        printf("  ── Scanning loaded images for profiling/analytics strings ──\n");
        {
            const char *search_terms[] = {
                "PROFILING", "ANALYTICS", "ANE_DEBUG", "ANED_DEBUG",
                "COREML_PROF", "ANE_PROF", "ESPRESSO_PROF",
                "ANE_ANALYTICS", "ANE_COMPILER", "ANED_LOG",
                "ANE_LOG", "ANE_TRACE", "performance_stats",
                "compiler_analytics", "kANEF", NULL
            };

            uint32_t imageCount = _dyld_image_count();
            for (uint32_t img = 0; img < imageCount; img++) {
                const char *imageName = _dyld_get_image_name(img);
                if (!imageName) continue;
                // Only scan ANE and Espresso framework images
                if (!strstr(imageName, "AppleNeuralEngine") &&
                    !strstr(imageName, "Espresso") &&
                    !strstr(imageName, "ANE") &&
                    !strstr(imageName, "CoreML"))
                    continue;

                const struct mach_header_64 *header =
                    (const struct mach_header_64 *)_dyld_get_image_header(img);
                if (!header) continue;

                unsigned long cstring_size = 0;
                const char *cstring_sect = getsectiondata(header, "__TEXT", "__cstring",
                                                          &cstring_size);
                if (!cstring_sect || cstring_size == 0) continue;

                // Scan __cstring section for matching strings
                const char *ptr = cstring_sect;
                const char *end = cstring_sect + cstring_size;
                while (ptr < end) {
                    size_t len = strnlen(ptr, end - ptr);
                    if (len > 4 && len < 256) {
                        for (int s = 0; search_terms[s]; s++) {
                            // Case-insensitive search
                            if (strcasestr(ptr, search_terms[s])) {
                                // Get short image name
                                const char *shortName = strrchr(imageName, '/');
                                shortName = shortName ? shortName + 1 : imageName;
                                printf("    [%s] \"%s\"\n", shortName, ptr);
                                break;  // Don't print same string for multiple matches
                            }
                        }
                    }
                    ptr += len + 1;
                }
            }
        }
        printf("\n");

        // Check NSUserDefaults for ANE-related keys
        printf("  ── NSUserDefaults ANE-related keys ──\n");
        {
            NSDictionary *defaults = [[NSUserDefaults standardUserDefaults] dictionaryRepresentation];
            for (NSString *key in [defaults allKeys]) {
                if ([key rangeOfString:@"ANE" options:NSCaseInsensitiveSearch].location != NSNotFound ||
                    [key rangeOfString:@"neural" options:NSCaseInsensitiveSearch].location != NSNotFound ||
                    [key rangeOfString:@"espresso" options:NSCaseInsensitiveSearch].location != NSNotFound ||
                    [key rangeOfString:@"coreml" options:NSCaseInsensitiveSearch].location != NSNotFound) {
                    printf("    %s = %s\n", [key UTF8String], [[defaults[key] description] UTF8String]);
                }
            }
        }
        printf("\n");

        // Check for ANE-related plist files
        printf("  ── ANE-related plist files ──\n");
        {
            NSArray *searchDirs = @[
                @"/Library/Preferences",
                [NSHomeDirectory() stringByAppendingPathComponent:@"Library/Preferences"],
                @"/System/Library/LaunchDaemons",
            ];
            NSArray *searchTerms = @[@"ANE", @"ane", @"neural", @"Neural", @"espresso", @"Espresso"];
            for (NSString *dir in searchDirs) {
                NSArray *files = list_files(dir, nil);
                for (NSString *f in files) {
                    for (NSString *term in searchTerms) {
                        if ([f rangeOfString:term].location != NSNotFound) {
                            printf("    %s\n", [f UTF8String]);
                            break;
                        }
                    }
                }
            }
        }
        printf("\n");

        // Check for ANE environment variables already set
        printf("  ── Current ANE-related env vars ──\n");
        {
            extern char **environ;
            for (char **env = environ; *env; env++) {
                if (strcasestr(*env, "ANE") || strcasestr(*env, "COREML") ||
                    strcasestr(*env, "ESPRESSO") || strcasestr(*env, "NEURAL") ||
                    strcasestr(*env, "PROFIL")) {
                    printf("    %s\n", *env);
                }
            }
        }
        printf("\n");

        // ════════════════════════════════════════════════════════════════
        // Section B: Try compilation with various environment variables
        // ════════════════════════════════════════════════════════════════
        printf("  ══ B: Environment variable compilation tests ══\n\n");

        Class g_D = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class g_I = NSClassFromString(@"_ANEInMemoryModel");

        // Define env vars to test
        const char *env_vars[] = {
            "COREML_PROFILING", "1",
            "ANE_PROFILING", "1",
            "ANE_DEBUG", "1",
            "ANED_DEBUG", "1",
            "ANE_ANALYTICS", "1",
            "ESPRESSO_PROFILING", "1",
            "ANE_COMPILER_ANALYTICS", "1",
            "ANE_LOG_LEVEL", "5",
            "ANED_LOG_LEVEL", "5",
            "COREML_LOG_LEVEL", "5",
            "ANE_ENABLE_ANALYTICS", "1",
            "ANE_PERFORMANCE_STATS", "1",
            "ANE_TRACE_ENABLED", "1",
            "ANED_TRACE_ENABLED", "1",
            "COREML_ENABLE_PROFILING", "1",
            "MPS_LOG_LEVEL", "5",
            "METAL_DEBUG_ERROR_MODE", "1",
            "ANE_COMPILER_LOG", "1",
            "ESPRESSO_ANE_PROFILING", "1",
            "com.apple.ane.profiling", "1",
            NULL, NULL
        };

        // Set ALL env vars before any compilation
        printf("  Setting environment variables:\n");
        for (int i = 0; env_vars[i]; i += 2) {
            setenv(env_vars[i], env_vars[i+1], 1);
            printf("    %s=%s\n", env_vars[i], env_vars[i+1]);
        }
        printf("\n");

        // Snapshot directories before compile
        NSDictionary *snap_tmp = snapshot_dir(NSTemporaryDirectory());
        NSDictionary *snap_caches = snapshot_dir(
            [NSHomeDirectory() stringByAppendingPathComponent:@"Library/Caches"]);
        NSDictionary *snap_log = snapshot_dir(@"/tmp");

        // Compile a model
        printf("  ── Compiling with all env vars set ──\n");
        NSData *wdata = nil;
        NSString *tempDir = nil;
        id mdl = compile_kernel(g_D, g_I, 512, 64, &wdata, &tempDir, @{});
        if (mdl) {
            printf("    compile+load OK\n");
            printf("    tempDir: %s\n", [tempDir UTF8String]);

            // Check tempDir for analytics files
            printf("\n  ── Files in model temp dir ──\n");
            NSArray *tempFiles = list_files(tempDir, nil);
            for (NSString *f in tempFiles) printf("    %s\n", [f UTF8String]);
            NSArray *weightFiles = list_files([tempDir stringByAppendingPathComponent:@"weights"], nil);
            for (NSString *f in weightFiles) printf("    %s\n", [f UTF8String]);

            // Check for new files in /tmp
            printf("\n  ── New files in /tmp ──\n");
            NSDictionary *snap_tmp_after = snapshot_dir(NSTemporaryDirectory());
            NSArray *newTmp = diff_snapshots(snap_tmp, snap_tmp_after);
            for (NSString *f in newTmp) printf("    %s\n", [f UTF8String]);
            if ([newTmp count] == 0) printf("    (none)\n");

            NSDictionary *snap_log_after = snapshot_dir(@"/tmp");
            NSArray *newLog = diff_snapshots(snap_log, snap_log_after);
            for (NSString *f in newLog) printf("    %s\n", [f UTF8String]);

            // Check caches
            printf("\n  ── New files in ~/Library/Caches ──\n");
            NSDictionary *snap_caches_after = snapshot_dir(
                [NSHomeDirectory() stringByAppendingPathComponent:@"Library/Caches"]);
            NSArray *newCaches = diff_snapshots(snap_caches, snap_caches_after);
            for (NSString *f in newCaches) printf("    %s\n", [f UTF8String]);
            if ([newCaches count] == 0) printf("    (none)\n");

            // Check for ANE-related files in common locations
            printf("\n  ── ANE files in various locations ──\n");
            NSArray *checkDirs = @[
                @"/tmp",
                @"/var/tmp",
                [NSHomeDirectory() stringByAppendingPathComponent:@"Library/Caches/com.apple.aned"],
                [NSHomeDirectory() stringByAppendingPathComponent:@"Library/Caches/com.apple.ANE"],
                [NSHomeDirectory() stringByAppendingPathComponent:@"Library/Caches/com.apple.neuralengine"],
                @"/private/var/folders",
            ];
            for (NSString *dir in checkDirs) {
                NSArray *files = list_files(dir, @"ane");
                for (NSString *f in files) printf("    %s\n", [f UTF8String]);
                files = list_files(dir, @"ANE");
                for (NSString *f in files) printf("    %s\n", [f UTF8String]);
                files = list_files(dir, @"analytics");
                for (NSString *f in files) printf("    %s\n", [f UTF8String]);
                files = list_files(dir, @"profil");
                for (NSString *f in files) printf("    %s\n", [f UTF8String]);
            }

            // Check for plist/bin files created in temp
            printf("\n  ── Plist/bin files search in temp dirs ──\n");
            {
                NSArray *extensions = @[@"plist", @"bin", @"analytics", @"json", @"log"];
                NSFileManager *fm = [NSFileManager defaultManager];
                NSArray *tempContents = [fm contentsOfDirectoryAtPath:NSTemporaryDirectory() error:nil];
                for (NSString *item in tempContents) {
                    for (NSString *ext in extensions) {
                        if ([item hasSuffix:ext]) {
                            printf("    %s/%s\n", [NSTemporaryDirectory() UTF8String], [item UTF8String]);
                        }
                    }
                }
            }

            // Probe model for analytics data
            printf("\n  ── Model analytics probe ──\n");
            {
                Ivar modelIvar = class_getInstanceVariable([mdl class], "_model");
                id aneModel = modelIvar ? object_getIvar(mdl, modelIvar) : nil;
                if (aneModel) {
                    NSDictionary *attrs = ((id(*)(id,SEL))objc_msgSend)(aneModel, @selector(modelAttributes));
                    if (attrs) {
                        // Look for any analytics-related keys
                        for (NSString *key in [attrs allKeys]) {
                            if ([key rangeOfString:@"nalytics" options:NSCaseInsensitiveSearch].location != NSNotFound ||
                                [key rangeOfString:@"rofil" options:NSCaseInsensitiveSearch].location != NSNotFound ||
                                [key rangeOfString:@"perf" options:NSCaseInsensitiveSearch].location != NSNotFound ||
                                [key rangeOfString:@"stat" options:NSCaseInsensitiveSearch].location != NSNotFound) {
                                printf("    attrs[%s] = %s\n", [key UTF8String], [[attrs[key] description] UTF8String]);
                            }
                        }
                        // Also dump all top-level keys
                        printf("    All attribute keys:\n");
                        for (NSString *key in [[attrs allKeys] sortedArrayUsingSelector:@selector(compare:)]) {
                            printf("      %s (%s)\n", [key UTF8String],
                                [NSStringFromClass([attrs[key] class]) UTF8String]);
                        }
                    }
                }
            }

            unload_model(mdl);
        }
        printf("\n");

        // ════════════════════════════════════════════════════════════════
        // Section C: Probe EspressoProfilingANEcompilerAnalytics
        // ════════════════════════════════════════════════════════════════
        printf("  ══ C: EspressoProfilingANEcompilerAnalytics deep probe ══\n\n");

        // First dump all Espresso profiling classes
        printf("  ── All Espresso*Profiling* and *ANE* classes ──\n");
        {
            unsigned int cc = 0;
            Class *classes = objc_copyClassList(&cc);
            NSMutableArray *names = [NSMutableArray array];
            for (unsigned int i = 0; i < cc; i++) {
                const char *cn = class_getName(classes[i]);
                if (strstr(cn, "EspressoProfiling") || (strstr(cn, "Espresso") && strstr(cn, "ANE")))
                    [names addObject:[NSString stringWithUTF8String:cn]];
            }
            free(classes);
            [names sortUsingSelector:@selector(compare:)];
            for (NSString *n in names) printf("    %s\n", [n UTF8String]);
            printf("\n");

            // Dump each one
            for (NSString *n in names) dump_class([n UTF8String]);
        }

        // Deep probe EspressoProfilingANEcompilerAnalytics
        {
            Class cls = NSClassFromString(@"EspressoProfilingANEcompilerAnalytics");
            if (cls) {
                printf("  ── EspressoProfilingANEcompilerAnalytics instance probe ──\n");

                // Try alloc/init
                @try {
                    id inst = [[cls alloc] init];
                    printf("    alloc/init: %s\n", inst ? [[inst description] UTF8String] : "nil");

                    if (inst) {
                        // Try compiler_analytics_file_names
                        @try {
                            id fileNames = ((id(*)(id,SEL))objc_msgSend)(inst,
                                @selector(compiler_analytics_file_names));
                            printf("    compiler_analytics_file_names: %s\n",
                                fileNames ? [[fileNames description] UTF8String] : "nil");
                            if (fileNames && [fileNames isKindOfClass:[NSArray class]]) {
                                for (id fn in (NSArray*)fileNames) {
                                    printf("      file: %s\n", [[fn description] UTF8String]);
                                }
                            }
                        } @catch (NSException *ex) {
                            printf("    compiler_analytics_file_names: EXC %s\n", [[ex reason] UTF8String]);
                        }

                        // Try all getter methods
                        unsigned int mc = 0;
                        Method *meths = class_copyMethodList(cls, &mc);
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

                        // Try setting file names then reading
                        @try {
                            NSArray *testPaths = @[@"/tmp/ane_analytics_test.plist"];
                            ((void(*)(id,SEL,id))objc_msgSend)(inst,
                                @selector(setCompiler_analytics_file_names:), testPaths);
                            id check = ((id(*)(id,SEL))objc_msgSend)(inst,
                                @selector(compiler_analytics_file_names));
                            printf("    After set file_names: %s\n",
                                check ? [[check description] UTF8String] : "nil");
                        } @catch (NSException *ex) {
                            printf("    set file_names: EXC %s\n", [[ex reason] UTF8String]);
                        }

                        // Check if there's a class method to get shared/default instance
                        Class meta = object_getClass((id)cls);
                        unsigned int cmc = 0;
                        Method *cmeths = class_copyMethodList(meta, &cmc);
                        for (unsigned int m = 0; m < cmc; m++) {
                            SEL sel = method_getName(cmeths[m]);
                            const char *sn = sel_getName(sel);
                            printf("    class method: +%s\n", sn);
                        }
                        free(cmeths);
                    }
                } @catch (NSException *ex) {
                    printf("    EXC: %s\n", [[ex reason] UTF8String]);
                }
            } else {
                printf("  EspressoProfilingANEcompilerAnalytics: NOT FOUND\n");
            }
        }
        printf("\n");

        // Probe EspressoProfilingNetworkANEInfo too
        {
            Class cls = NSClassFromString(@"EspressoProfilingNetworkANEInfo");
            if (cls) {
                printf("  ── EspressoProfilingNetworkANEInfo instance probe ──\n");
                @try {
                    id inst = [[cls alloc] init];
                    printf("    alloc/init: %s\n", inst ? [[inst description] UTF8String] : "nil");
                    if (inst) {
                        unsigned int mc = 0;
                        Method *meths = class_copyMethodList(cls, &mc);
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
                    printf("    EXC: %s\n", [[ex reason] UTF8String]);
                }
            }
        }
        printf("\n");

        // ════════════════════════════════════════════════════════════════
        // Section D: Compile with various option dictionaries
        // ════════════════════════════════════════════════════════════════
        printf("  ══ D: Compile option dict tests ══\n\n");

        struct {
            const char *label;
            NSDictionary *opts;
        } option_tests[] = {
            { "kANEFPerformanceStatsMask=0xFFFFFFFF",
              @{@"kANEFPerformanceStatsMask": @(0xFFFFFFFF)} },
            { "ANECompilerAnalytics=YES",
              @{@"ANECompilerAnalytics": @YES} },
            { "kANEFEnableAnalytics=YES",
              @{@"kANEFEnableAnalytics": @YES} },
            { "EnableAnalytics=YES",
              @{@"EnableAnalytics": @YES} },
            { "AnalyticsEnabled=YES",
              @{@"AnalyticsEnabled": @YES} },
            { "profiling=YES",
              @{@"profiling": @YES} },
            { "EnableProfiling=YES",
              @{@"EnableProfiling": @YES} },
            { "kANEFCompilerAnalytics=YES",
              @{@"kANEFCompilerAnalytics": @YES} },
            { "kANEFDebugMask=0xFFFFFFFF",
              @{@"kANEFDebugMask": @(0xFFFFFFFF)} },
            { "kANEFPerformanceStatsMask=0xFF + Analytics=YES",
              @{@"kANEFPerformanceStatsMask": @(0xFF), @"ANECompilerAnalytics": @YES,
                @"EnableAnalytics": @YES} },
            { "compilerOptions analytics",
              @{@"compilerOptions": @{@"analytics": @YES, @"profiling": @YES}} },
            { "kANEFPerformanceStatsMask=1",
              @{@"kANEFPerformanceStatsMask": @(1)} },
            { "kANEFPerformanceStatsMask=2",
              @{@"kANEFPerformanceStatsMask": @(2)} },
            { "kANEFPerformanceStatsMask=4",
              @{@"kANEFPerformanceStatsMask": @(4)} },
            { "kANEFPerformanceStatsMask=8",
              @{@"kANEFPerformanceStatsMask": @(8)} },
        };
        int numTests = sizeof(option_tests) / sizeof(option_tests[0]);

        for (int t = 0; t < numTests; t++) {
            printf("  ── Test: %s ──\n", option_tests[t].label);

            NSDictionary *snap_before = snapshot_dir(NSTemporaryDirectory());

            NSData *wdata2 = nil;
            NSString *tempDir2 = nil;
            @try {
                id mdl2 = compile_kernel(g_D, g_I, 256, 32, &wdata2, &tempDir2, option_tests[t].opts);
                if (mdl2) {
                    printf("    compile+load OK\n");

                    // Check for new files
                    NSDictionary *snap_after = snapshot_dir(NSTemporaryDirectory());
                    NSArray *newFiles = diff_snapshots(snap_before, snap_after);
                    if ([newFiles count] > 0) {
                        printf("    NEW FILES in tmp:\n");
                        for (NSString *f in newFiles) printf("      %s\n", [f UTF8String]);
                    }

                    // Check model temp dir for analytics
                    if (tempDir2) {
                        NSArray *modelFiles = list_files(tempDir2, nil);
                        for (NSString *f in modelFiles) {
                            NSString *fname = [f lastPathComponent];
                            if (![fname isEqualToString:@"model.mil"] &&
                                ![fname isEqualToString:@"weights"]) {
                                printf("    model dir: %s\n", [f UTF8String]);
                            }
                        }
                    }

                    // Probe model attributes for any analytics
                    Ivar modelIvar = class_getInstanceVariable([mdl2 class], "_model");
                    id aneModel = modelIvar ? object_getIvar(mdl2, modelIvar) : nil;
                    if (aneModel) {
                        NSDictionary *attrs = ((id(*)(id,SEL))objc_msgSend)(aneModel, @selector(modelAttributes));
                        if (attrs) {
                            // Check for analytics/perf keys in nested structures
                            for (NSString *key in attrs) {
                                id val = attrs[key];
                                if ([key rangeOfString:@"nalytics" options:NSCaseInsensitiveSearch].location != NSNotFound ||
                                    [key rangeOfString:@"rofil" options:NSCaseInsensitiveSearch].location != NSNotFound ||
                                    [key rangeOfString:@"perf" options:NSCaseInsensitiveSearch].location != NSNotFound ||
                                    [key rangeOfString:@"stat" options:NSCaseInsensitiveSearch].location != NSNotFound ||
                                    [key rangeOfString:@"debug" options:NSCaseInsensitiveSearch].location != NSNotFound) {
                                    printf("    attr[%s] = %s\n", [key UTF8String],
                                        [[val description] UTF8String]);
                                }
                            }
                        }

                        // Try to access compilerAnalytics directly from the model
                        @try {
                            if ([aneModel respondsToSelector:@selector(compilerAnalytics)]) {
                                id ca = ((id(*)(id,SEL))objc_msgSend)(aneModel, @selector(compilerAnalytics));
                                printf("    compilerAnalytics: %s\n",
                                    ca ? [[ca description] UTF8String] : "nil");
                            }
                        } @catch (NSException *ex) {}

                        // Try analyticsData
                        @try {
                            if ([aneModel respondsToSelector:@selector(analyticsData)]) {
                                id ad = ((id(*)(id,SEL))objc_msgSend)(aneModel, @selector(analyticsData));
                                printf("    analyticsData: %s\n",
                                    ad ? [[ad description] UTF8String] : "nil");
                            }
                        } @catch (NSException *ex) {}

                        // Try performanceStats
                        @try {
                            if ([aneModel respondsToSelector:@selector(performanceStats)]) {
                                id ps = ((id(*)(id,SEL))objc_msgSend)(aneModel, @selector(performanceStats));
                                printf("    performanceStats: %s\n",
                                    ps ? [[ps description] UTF8String] : "nil");
                            }
                        } @catch (NSException *ex) {}
                    }

                    unload_model(mdl2);
                } else {
                    printf("    compile FAILED\n");
                }
            } @catch (NSException *ex) {
                printf("    EXCEPTION: %s\n", [[ex reason] UTF8String]);
            }
            printf("\n");
        }

        // ════════════════════════════════════════════════════════════════
        // Section E: Probe _ANEClient for analytics-related methods
        // ════════════════════════════════════════════════════════════════
        printf("  ══ E: _ANEClient / _ANEModel analytics methods ══\n\n");

        {
            // Dump _ANEClient methods related to analytics/profiling
            const char *probe_classes[] = {
                "_ANEClient", "_ANEModel", "_ANEDeviceInfo",
                "_ANEInMemoryModel", "_ANEInMemoryModelDescriptor",
                NULL
            };
            for (int i = 0; probe_classes[i]; i++) {
                Class cls = NSClassFromString([NSString stringWithUTF8String:probe_classes[i]]);
                if (!cls) continue;
                unsigned int mc = 0;
                Method *meths = class_copyMethodList(cls, &mc);
                BOOL found = NO;
                for (unsigned int m = 0; m < mc; m++) {
                    SEL sel = method_getName(meths[m]);
                    const char *sn = sel_getName(sel);
                    if (strcasestr(sn, "analytics") || strcasestr(sn, "profil") ||
                        strcasestr(sn, "perf") || strcasestr(sn, "stat") ||
                        strcasestr(sn, "debug") || strcasestr(sn, "log") ||
                        strcasestr(sn, "trace")) {
                        if (!found) {
                            printf("  [%s] analytics-related methods:\n", probe_classes[i]);
                            found = YES;
                        }
                        printf("    - %s  (%s)\n", sn, method_getTypeEncoding(meths[m]) ?: "?");
                    }
                }
                free(meths);
                // Also check class methods
                Class meta = object_getClass((id)cls);
                meths = class_copyMethodList(meta, &mc);
                for (unsigned int m = 0; m < mc; m++) {
                    SEL sel = method_getName(meths[m]);
                    const char *sn = sel_getName(sel);
                    if (strcasestr(sn, "analytics") || strcasestr(sn, "profil") ||
                        strcasestr(sn, "perf") || strcasestr(sn, "stat") ||
                        strcasestr(sn, "debug") || strcasestr(sn, "log") ||
                        strcasestr(sn, "trace")) {
                        if (!found) {
                            printf("  [%s] analytics-related methods:\n", probe_classes[i]);
                            found = YES;
                        }
                        printf("    + %s  (%s)\n", sn, method_getTypeEncoding(meths[m]) ?: "?");
                    }
                }
                free(meths);
                if (found) printf("\n");
            }
        }

        // ════════════════════════════════════════════════════════════════
        // Section F: Check os_log/os_signpost for ANE analytics categories
        // ════════════════════════════════════════════════════════════════
        printf("  ══ F: Scan for os_log categories in ANE framework ══\n\n");
        {
            uint32_t imageCount = _dyld_image_count();
            for (uint32_t img = 0; img < imageCount; img++) {
                const char *imageName = _dyld_get_image_name(img);
                if (!imageName) continue;
                if (!strstr(imageName, "AppleNeuralEngine")) continue;

                const struct mach_header_64 *header =
                    (const struct mach_header_64 *)_dyld_get_image_header(img);
                if (!header) continue;

                // Scan __objc_methnames for analytics strings
                unsigned long size = 0;
                const char *data = getsectiondata(header, "__TEXT", "__objc_methnames", &size);
                if (data && size > 0) {
                    printf("  __objc_methnames analytics hits:\n");
                    const char *ptr = data;
                    const char *end = data + size;
                    while (ptr < end) {
                        size_t len = strnlen(ptr, end - ptr);
                        if (len > 4 && (strcasestr(ptr, "analytics") || strcasestr(ptr, "profil"))) {
                            printf("    %s\n", ptr);
                        }
                        ptr += len + 1;
                    }
                }

                // Also scan __objc_classname
                data = getsectiondata(header, "__TEXT", "__objc_classname", &size);
                if (data && size > 0) {
                    printf("  __objc_classname analytics hits:\n");
                    const char *ptr = data;
                    const char *end = data + size;
                    while (ptr < end) {
                        size_t len = strnlen(ptr, end - ptr);
                        if (len > 4 && (strcasestr(ptr, "analytics") || strcasestr(ptr, "profil") ||
                            strcasestr(ptr, "debug") || strcasestr(ptr, "perf"))) {
                            printf("    %s\n", ptr);
                        }
                        ptr += len + 1;
                    }
                }
            }
        }
        printf("\n");

        // ════════════════════════════════════════════════════════════════
        // Section G: Check if CFPreferences/launchd plist can enable tracing
        // ════════════════════════════════════════════════════════════════
        printf("  ══ G: CFPreferences / Daemon config check ══\n\n");
        {
            // Check if aned has a launchd plist
            NSArray *launchdPaths = @[
                @"/System/Library/LaunchDaemons/com.apple.aned.plist",
                @"/Library/LaunchDaemons/com.apple.aned.plist",
                @"/System/Library/LaunchDaemons/com.apple.ANECompilerService.plist",
            ];
            for (NSString *path in launchdPaths) {
                BOOL exists = [[NSFileManager defaultManager] fileExistsAtPath:path];
                printf("  %s: %s\n", [path UTF8String], exists ? "EXISTS" : "not found");
                if (exists) {
                    NSDictionary *plist = [NSDictionary dictionaryWithContentsOfFile:path];
                    if (plist) {
                        NSData *json = [NSJSONSerialization dataWithJSONObject:plist
                            options:NSJSONWritingPrettyPrinted|NSJSONWritingSortedKeys error:nil];
                        if (json) {
                            NSString *str = [[NSString alloc] initWithData:json encoding:NSUTF8StringEncoding];
                            printf("    %s\n", [str UTF8String]);
                        }
                    }
                }
            }
        }
        printf("\n");

        // ════════════════════════════════════════════════════════════════
        // Section H: Try direct symbol lookup for analytics functions
        // ════════════════════════════════════════════════════════════════
        printf("  ══ H: Direct symbol lookup ══\n\n");
        {
            const char *symbols[] = {
                "_ANECompilerAnalyticsCreate",
                "_ANECompilerAnalyticsGetBuffer",
                "_ANEEnableAnalytics",
                "_ANESetProfilingEnabled",
                "_ANEGetProfilingEnabled",
                "_ane_compiler_analytics_create",
                "_ane_compiler_analytics_get",
                "_ane_enable_profiling",
                "_ane_set_debug_level",
                "ANECompilerAnalyticsCreate",
                "ANEEnableAnalytics",
                "ANESetDebugLevel",
                NULL
            };
            for (int i = 0; symbols[i]; i++) {
                void *sym = dlsym(RTLD_DEFAULT, symbols[i]);
                printf("  %s: %s\n", symbols[i], sym ? "FOUND" : "not found");
            }
        }
        printf("\n");

        // ════════════════════════════════════════════════════════════════
        // Section I: Final comprehensive Espresso network probe
        // ════════════════════════════════════════════════════════════════
        printf("  ══ I: Espresso network-level profiling probe ══\n\n");
        {
            // Look for Espresso__net classes related to profiling
            unsigned int cc = 0;
            Class *classes = objc_copyClassList(&cc);
            printf("  Profiling-related classes (all frameworks):\n");
            for (unsigned int i = 0; i < cc; i++) {
                const char *cn = class_getName(classes[i]);
                if (strcasestr(cn, "profiling") || strcasestr(cn, "Profiling")) {
                    printf("    %s\n", cn);
                }
            }
            free(classes);
        }
        printf("\n");

        // Cleanup env vars (optional, for clean state)
        for (int i = 0; env_vars[i]; i += 2) {
            unsetenv(env_vars[i]);
        }

        printf("\n  ██████ ANALYTICS PROFILING PROBE COMPLETE ██████\n\n");
    }
    return 0;
}
