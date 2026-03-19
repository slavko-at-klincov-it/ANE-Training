// test_analytics_plist.m — Test if compiler_analytics_on.plist triggers analytics output
// Places the plist in the model temp directory before compilation,
// then checks /tmp/ for analytics output files.
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>

static IOSurfaceRef make_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0});
}

// Snapshot /tmp/ files matching a pattern
static NSArray *tmp_files_matching(NSString *pattern) {
    NSFileManager *fm = [NSFileManager defaultManager];
    NSArray *all = [fm contentsOfDirectoryAtPath:@"/tmp" error:nil];
    NSMutableArray *matches = [NSMutableArray array];
    for (NSString *f in all) {
        if ([f rangeOfString:pattern].location != NSNotFound) {
            [matches addObject:f];
        }
    }
    return matches;
}

static id compile_kernel_with_plist(int CH, int SP, NSString *plistName, NSDictionary *plistContent) {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);

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
    NSFileManager *fm = [NSFileManager defaultManager];
    [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
        withIntermediateDirectories:YES attributes:nil error:nil];
    [md writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    [wdata writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

    // *** Place the analytics plist BEFORE compilation ***
    if (plistName && plistContent) {
        NSString *plistPath = [td stringByAppendingPathComponent:plistName];
        [plistContent writeToFile:plistPath atomically:YES];
        printf("    Placed: %s\n", [plistPath UTF8String]);

        // Also try in parent of td
        NSString *parentPlist = [[td stringByDeletingLastPathComponent] stringByAppendingPathComponent:plistName];
        [plistContent writeToFile:parentPlist atomically:YES];
        printf("    Placed: %s\n", [parentPlist UTF8String]);
    }

    // Also try placing as compiler_options.plist and net_options.plist
    if (plistContent) {
        for (NSString *altName in @[@"compiler_options.plist", @"net_options.plist"]) {
            NSString *altPath = [td stringByAppendingPathComponent:altName];
            [plistContent writeToFile:altPath atomically:YES];
            printf("    Placed: %s\n", [altPath UTF8String]);
        }
    }

    printf("    Model dir: %s\n", [td UTF8String]);
    printf("    Compiling %dx%d sp%d ...\n", CH, CH, SP);

    NSError *e = nil;
    BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
        mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e);
    printf("    Compile: %s\n", ok ? "OK" : [[e description] UTF8String]);

    if (ok) {
        // List ALL files in model dir after compilation
        printf("    Files after compile:\n");
        NSDirectoryEnumerator *en = [fm enumeratorAtPath:td];
        NSString *file;
        while ((file = [en nextObject])) {
            NSDictionary *attrs = [fm attributesOfItemAtPath:[td stringByAppendingPathComponent:file] error:nil];
            printf("      %s (%llu bytes)\n", [file UTF8String], [attrs fileSize]);
        }
    }

    return ok ? mdl : nil;
}

int main() {
    @autoreleasepool {
        setbuf(stdout, NULL);
        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);

        printf("\n  ██████ ANALYTICS PLIST TRIGGER TEST ██████\n\n");

        // Snapshot /tmp before
        NSArray *before = tmp_files_matching(@"compiler_analytics");
        printf("  /tmp/ analytics files before: %lu\n", (unsigned long)before.count);
        for (NSString *f in before) printf("    %s\n", [f UTF8String]);

        // ════════════════════════════════════════
        // Test 1: compiler_analytics_on.plist (empty)
        // ════════════════════════════════════════
        printf("\n  ── Test 1: compiler_analytics_on.plist (empty dict) ──\n");
        compile_kernel_with_plist(256, 64, @"compiler_analytics_on.plist", @{});

        NSArray *after1 = tmp_files_matching(@"compiler_analytics");
        printf("    /tmp/ analytics files after: %lu\n", (unsigned long)after1.count);
        for (NSString *f in after1) {
            if (![before containsObject:f]) printf("    NEW: %s\n", [f UTF8String]);
        }

        // ════════════════════════════════════════
        // Test 2: compiler_analytics_on.plist with enable flags
        // ════════════════════════════════════════
        printf("\n  ── Test 2: compiler_analytics_on.plist (with flags) ──\n");
        compile_kernel_with_plist(512, 64, @"compiler_analytics_on.plist", @{
            @"GenerateStaticPerfAnalytics": @YES,
            @"GenerateAnalyticsBuffer": @YES,
            @"generate-static-perf-analytics": @YES,
            @"generate-analytics-buffer": @YES,
        });

        NSArray *after2 = tmp_files_matching(@"compiler_analytics");
        printf("    /tmp/ analytics files after: %lu\n", (unsigned long)after2.count);
        for (NSString *f in after2) {
            if (![before containsObject:f]) printf("    NEW: %s\n", [f UTF8String]);
        }

        // ════════════════════════════════════════
        // Test 3: compiler_analytics_on.plist with output path
        // ════════════════════════════════════════
        printf("\n  ── Test 3: compiler_analytics_on.plist (with output path) ──\n");
        compile_kernel_with_plist(256, 32, @"compiler_analytics_on.plist", @{
            @"GenerateStaticPerfAnalytics": @YES,
            @"GenerateAnalyticsBuffer": @YES,
            @"AnalyticsOutputPath": @"/tmp/ane_analytics_test",
            @"compiler_analytics_out": @"/tmp/ane_analytics_test",
        });

        NSArray *after3 = tmp_files_matching(@"analytics");
        printf("    /tmp/ analytics files after: %lu\n", (unsigned long)after3.count);
        for (NSString *f in after3) {
            if (![before containsObject:f]) printf("    NEW: %s\n", [f UTF8String]);
        }

        // ════════════════════════════════════════
        // Test 4: Try compile options dict with analytics flags
        // ════════════════════════════════════════
        printf("\n  ── Test 4: Compile with analytics options dict ──\n");
        {
            Class g_D  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
            Class g_I  = NSClassFromString(@"_ANEInMemoryModel");
            int CH = 128, SP = 32;
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

            // Try optionsPlist with analytics flags
            NSDictionary *optsPlist = @{
                @"GenerateStaticPerfAnalytics": @YES,
                @"GenerateAnalyticsBuffer": @YES,
                @"generate-static-perf-analytics": @YES,
                @"generate-analytics-buffer": @YES,
            };
            NSData *plistData = [NSPropertyListSerialization dataWithPropertyList:optsPlist
                format:NSPropertyListXMLFormat_v1_0 options:0 error:nil];

            id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D, @selector(modelWithMILText:weights:optionsPlist:),
                md, @{@"@model_path/weights/weight.bin": @{@"offset":@0, @"data":wdata}}, plistData);
            id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(g_I, @selector(inMemoryModelWithDescriptor:), desc);
            id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
            NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
            NSFileManager *fm = [NSFileManager defaultManager];
            [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
                withIntermediateDirectories:YES attributes:nil error:nil];
            [md writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
            [wdata writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

            // Also place the plist files
            [@{} writeToFile:[td stringByAppendingPathComponent:@"compiler_analytics_on.plist"] atomically:YES];
            [optsPlist writeToFile:[td stringByAppendingPathComponent:@"compiler_options.plist"] atomically:YES];
            [optsPlist writeToFile:[td stringByAppendingPathComponent:@"net_options.plist"] atomically:YES];

            printf("    Model dir: %s\n", [td UTF8String]);

            // Compile with analytics options
            NSError *e = nil;
            NSDictionary *compileOpts = @{
                @"GenerateStaticPerfAnalytics": @YES,
                @"GenerateAnalyticsBuffer": @YES,
                @"kANEFPerformanceStatsMask": @(0xFFFFFFFF),
                @"kANEFCompilerOptionsFilenameKey": @"compiler_options.plist",
            };
            BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(compileWithQoS:options:error:), 21, compileOpts, &e);
            printf("    Compile: %s\n", ok ? "OK" : [[e description] UTF8String]);

            if (ok) {
                printf("    Files after compile:\n");
                NSDirectoryEnumerator *en = [fm enumeratorAtPath:td];
                NSString *file;
                while ((file = [en nextObject])) {
                    NSDictionary *attrs = [fm attributesOfItemAtPath:[td stringByAppendingPathComponent:file] error:nil];
                    printf("      %s (%llu bytes)\n", [file UTF8String], [attrs fileSize]);
                }
            }

            NSArray *after4 = tmp_files_matching(@"analytics");
            printf("    /tmp/ analytics files: %lu\n", (unsigned long)after4.count);
            for (NSString *f in after4) {
                if (![before containsObject:f]) printf("    NEW: %s\n", [f UTF8String]);
            }
        }

        // ════════════════════════════════════════
        // Test 5: Check broader /tmp/ for ANY new files
        // ════════════════════════════════════════
        printf("\n  ── Test 5: Broad /tmp/ scan for any compiler output ──\n");
        NSArray *allTmp = tmp_files_matching(@"compiler");
        printf("    /tmp/ files with 'compiler': %lu\n", (unsigned long)allTmp.count);
        for (NSString *f in allTmp) printf("    %s\n", [f UTF8String]);

        allTmp = tmp_files_matching(@"ane");
        printf("    /tmp/ files with 'ane': %lu\n", (unsigned long)allTmp.count);
        for (NSString *f in allTmp) printf("    %s\n", [f UTF8String]);

        allTmp = tmp_files_matching(@"analytics");
        printf("    /tmp/ files with 'analytics': %lu\n", (unsigned long)allTmp.count);
        for (NSString *f in allTmp) printf("    %s\n", [f UTF8String]);

        printf("\n  Done. Compiles used: probably 4\n\n");
    }
    return 0;
}
