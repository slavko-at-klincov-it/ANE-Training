// test_ffn_bwd_debug.m — Debug why gen_ffn_bwd() produces all-zero output
// Tests:
//   1. FFN forward kernel (gen_ffn_fwd_taps) — known to work, used as baseline
//   2. FFN backward kernel (gen_ffn_bwd) at full size (DIM=768, HIDDEN=2048, SEQ=256)
//   3. FFN backward kernel at small size (DIM=64, HIDDEN=128, SEQ=16) to isolate SRAM issues
//   4. Minimal identity-like test: single conv with known weights to verify backward path
//
// For each test: create random weights + inputs, compile, eval, check output norm.

#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>
#include <math.h>
#include <arm_neon.h>

// ---- Inline infrastructure (not using stories_config.h to allow custom dims) ----

static Class g_D, g_I, g_AR, g_AIO;
static int g_compile_count = 0;

static void ane_init(void) {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    g_D  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_I  = NSClassFromString(@"_ANEInMemoryModel");
    g_AR = NSClassFromString(@"_ANERequest");
    g_AIO= NSClassFromString(@"_ANEIOSurfaceObject");
}

static IOSurfaceRef make_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0});
}

typedef struct { void *model; IOSurfaceRef ioIn, ioOut; void *request; void *tmpDir; } Kern;

static Kern *compile_kern_mil_w(NSString *mil, NSDictionary *weights, int ic_bytes, int oc_bytes) {
    @autoreleasepool {
    NSData *md = [mil dataUsingEncoding:NSUTF8StringEncoding];
    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D, @selector(modelWithMILText:weights:optionsPlist:), md, weights, nil);
    if (!desc) { printf("  [compile] desc=NULL\n"); return NULL; }
    id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(g_I, @selector(inMemoryModelWithDescriptor:), desc);
    id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    [[NSFileManager defaultManager] createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"] withIntermediateDirectories:YES attributes:nil error:nil];
    [md writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    for (NSString *path in weights) {
        NSString *rel = [path stringByReplacingOccurrencesOfString:@"@model_path/" withString:@""];
        [weights[path][@"data"] writeToFile:[td stringByAppendingPathComponent:rel] atomically:YES];
    }
    NSError *e = nil;
    if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
        printf("  [compile] FAIL: %s\n", e ? [[e description] UTF8String] : "no error"); return NULL;
    }
    if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e)) {
        printf("  [compile] load FAIL\n"); return NULL;
    }
    g_compile_count++;
    Kern *k = (Kern*)calloc(1, sizeof(Kern));
    k->model = (void*)CFBridgingRetain(mdl);
    k->ioIn = make_surface(ic_bytes);
    k->ioOut = make_surface(oc_bytes);
    id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), k->ioIn);
    id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), k->ioOut);
    k->request = (void*)CFBridgingRetain(((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        @[wI], @[@0], @[wO], @[@0], nil, nil, @0));
    k->tmpDir = (void*)CFBridgingRetain(td);
    return k;
    }
}

static void free_kern(Kern *k) {
    if (!k) return;
    id mdl = (__bridge id)k->model; NSError *e = nil;
    ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl, @selector(unloadWithQoS:error:), 21, &e);
    CFRelease(k->ioIn); CFRelease(k->ioOut);
    [[NSFileManager defaultManager] removeItemAtPath:(__bridge id)k->tmpDir error:nil];
    CFRelease(k->model); CFRelease(k->request); CFRelease(k->tmpDir);
    free(k);
}

static void ane_eval(Kern *k) {
    id mdl = (__bridge id)k->model; id req = (__bridge id)k->request; NSError *e = nil;
    BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(mdl, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
    if (!ok || e) printf("  ANE EVAL ERROR: ok=%d err=%s\n", ok, e ? [[e description] UTF8String] : "nil");
}

// ---- fp16 <-> fp32 ----
static void cvt_f32_f16(_Float16 *dst, const float *src, int n) {
    for (int i = 0; i < n; i++) dst[i] = (_Float16)src[i];
}
static void cvt_f16_f32(float *dst, const _Float16 *src, int n) {
    for (int i = 0; i < n; i++) dst[i] = (float)src[i];
}

// ---- IO helpers ----
static void io_write_fp16(IOSurfaceRef s, const float *data, int channels, int sp) {
    IOSurfaceLock(s, 0, NULL);
    cvt_f32_f16((_Float16*)IOSurfaceGetBaseAddress(s), data, channels * sp);
    IOSurfaceUnlock(s, 0, NULL);
}
static void io_write_fp16_at(IOSurfaceRef s, int ch_off, const float *data, int channels, int sp) {
    IOSurfaceLock(s, 0, NULL);
    cvt_f32_f16((_Float16*)IOSurfaceGetBaseAddress(s) + ch_off * sp, data, channels * sp);
    IOSurfaceUnlock(s, 0, NULL);
}
static void io_read_fp16(IOSurfaceRef s, float *data, int ch_off, int channels, int sp) {
    IOSurfaceLock(s, kIOSurfaceLockReadOnly, NULL);
    cvt_f16_f32(data, (_Float16*)IOSurfaceGetBaseAddress(s) + ch_off * sp, channels * sp);
    IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, NULL);
}

// ---- Blob builders ----
static NSData *build_blob(const float *w, int rows, int cols) {
    int ws = rows*cols*2, tot = 128+ws;
    uint8_t *b = (uint8_t*)calloc(tot, 1);
    b[0]=1; b[4]=2; b[64]=0xEF; b[65]=0xBE; b[66]=0xAD; b[67]=0xDE; b[68]=1;
    *(uint32_t*)(b+72)=ws; *(uint32_t*)(b+80)=128;
    _Float16 *fp16 = (_Float16*)(b+128);
    for (int i = 0; i < rows*cols; i++) fp16[i] = (_Float16)w[i];
    return [NSData dataWithBytesNoCopy:b length:tot freeWhenDone:YES];
}
static NSData *build_blob_t(const float *w, int rows, int cols) {
    int ws = cols*rows*2, tot = 128+ws;
    uint8_t *b = (uint8_t*)calloc(tot, 1);
    b[0]=1; b[4]=2; b[64]=0xEF; b[65]=0xBE; b[66]=0xAD; b[67]=0xDE; b[68]=1;
    *(uint32_t*)(b+72)=ws; *(uint32_t*)(b+80)=128;
    _Float16 *fp16 = (_Float16*)(b+128);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            fp16[j*rows+i] = (_Float16)w[i*cols+j];
    return [NSData dataWithBytesNoCopy:b length:tot freeWhenDone:YES];
}

// ---- MIL generators (parameterized by dim/hidden/seq) ----

#define MIL_HDR \
    @"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, " \
    "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, " \
    "{\"coremltools-version\", \"9.0\"}})]\n{\n"
#define CONV_CONST \
    "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n" \
    "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n" \
    "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n" \
    "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n" \
    "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"

// FFN backward: parameterized by D, H, S
static NSString *gen_ffn_bwd_p(int D, int H, int S) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", D+2*H, S];
    [m appendString:@CONV_CONST];
    [m appendString:@"        tensor<int32, [4]> bd = const()[name=string(\"bd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendFormat:@"        tensor<int32, [4]> sd = const()[name=string(\"sd\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", D, S];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dffn = slice_by_size(x=x,begin=bd,size=sd)[name=string(\"s0\")];\n", D, S];
    [m appendFormat:@"        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", D];
    [m appendFormat:@"        tensor<int32, [4]> s1 = const()[name=string(\"s1\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", H, S];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> h1 = slice_by_size(x=x,begin=b1,size=s1)[name=string(\"s1x\")];\n", H, S];
    [m appendFormat:@"        tensor<int32, [4]> b3 = const()[name=string(\"b3\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", D+H];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> h3 = slice_by_size(x=x,begin=b3,size=s1)[name=string(\"s3x\")];\n", H, S];
    // W2^T: [H, D, 1, 1] conv weight
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> W2t = const()[name=string(\"W2t\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w2t.bin\"), offset=uint64(64)))];\n", H, D, H, D];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dsilu = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W2t,x=dffn)[name=string(\"cw2\")];\n", H, S];
    // SiLU derivative chain
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> sig = sigmoid(x=h1)[name=string(\"sg\")];\n", H, S];
    [m appendString:@"        fp16 one = const()[name=string(\"one\"), val=fp16(1.0)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> oms = sub(x=one,y=sig)[name=string(\"oms\")];\n", H, S];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> homs = mul(x=h1,y=oms)[name=string(\"homs\")];\n", H, S];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> brk = add(x=one,y=homs)[name=string(\"brk\")];\n", H, S];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dsd = mul(x=sig,y=brk)[name=string(\"dsd\")];\n", H, S];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> t1 = mul(x=dsilu,y=h3)[name=string(\"t1\")];\n", H, S];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dh1 = mul(x=t1,y=dsd)[name=string(\"dh1\")];\n", H, S];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> slh = mul(x=h1,y=sig)[name=string(\"slh\")];\n", H, S];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dh3 = mul(x=dsilu,y=slh)[name=string(\"dh3\")];\n", H, S];
    // W1^T, W3^T conv
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> W1t = const()[name=string(\"W1t\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w1t.bin\"), offset=uint64(64)))];\n", D, H, D, H];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> W3t = const()[name=string(\"W3t\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w3t.bin\"), offset=uint64(64)))];\n", D, H, D, H];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dx1 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W1t,x=dh1)[name=string(\"cw1\")];\n", D, S];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dx3 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W3t,x=dh3)[name=string(\"cw3\")];\n", D, S];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dx = add(x=dx1,y=dx3)[name=string(\"adx\")];\n", D, S];
    [m appendString:@"        int32 cax = const()[name=string(\"cax\"), val=int32(1)];\n"];
    [m appendString:@"        bool cid = const()[name=string(\"cid\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = concat(axis=cax,interleave=cid,values=(dx,dh1,dh3))[name=string(\"cat\")];\n", D+2*H, S];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// FFN forward + taps: parameterized
static NSString *gen_ffn_fwd_taps_p(int D, int H, int S) {
    float invd = 1.0f/(float)D;
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", D, S];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> sq = mul(x=x,y=x)[name=string(\"sq\")];\n", D, S];
    [m appendFormat:@"        tensor<int32, [1]> rax = const()[name=string(\"rax\"), val=tensor<int32, [1]>([1])];\n"];
    [m appendFormat:@"        bool kd = const()[name=string(\"kd\"), val=bool(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss = reduce_sum(x=sq,axes=rax,keep_dims=kd)[name=string(\"ss\")];\n", S];
    [m appendFormat:@"        fp16 invd = const()[name=string(\"invd\"), val=fp16(%f)];\n", invd];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss2 = mul(x=ss,y=invd)[name=string(\"ss2\")];\n", S];
    [m appendFormat:@"        fp16 eps = const()[name=string(\"eps\"), val=fp16(0.00001)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss3 = add(x=ss2,y=eps)[name=string(\"ss3\")];\n", S];
    [m appendFormat:@"        fp16 nhalf = const()[name=string(\"nhalf\"), val=fp16(-0.5)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> rrms = pow(x=ss3,y=nhalf)[name=string(\"rrms\")];\n", S];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xr = mul(x=x,y=rrms)[name=string(\"xr\")];\n", D, S];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,1]> rw = const()[name=string(\"rw\"), val=tensor<fp16, [1,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/rms2.bin\"), offset=uint64(64)))];\n", D, D];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xn = mul(x=xr,y=rw)[name=string(\"xn\")];\n", D, S];
    [m appendString:@CONV_CONST];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> W1 = const()[name=string(\"W1\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w1.bin\"), offset=uint64(64)))];\n", H,D,H,D];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> W3 = const()[name=string(\"W3\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w3.bin\"), offset=uint64(64)))];\n", H,D,H,D];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> W2 = const()[name=string(\"W2\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w2.bin\"), offset=uint64(64)))];\n", D,H,D,H];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> h1 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W1,x=xn)[name=string(\"c1\")];\n", H,S];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> h3 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W3,x=xn)[name=string(\"c3\")];\n", H,S];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> sig = sigmoid(x=h1)[name=string(\"sg\")];\n", H,S];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> silu = mul(x=h1,y=sig)[name=string(\"si\")];\n", H,S];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> gate = mul(x=silu,y=h3)[name=string(\"gt\")];\n", H,S];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> y = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W2,x=gate)[name=string(\"c2\")];\n", D,S];
    [m appendString:@"        int32 cax = const()[name=string(\"cax\"), val=int32(1)];\n"];
    [m appendString:@"        bool cid = const()[name=string(\"cid\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = concat(axis=cax,interleave=cid,values=(y,h1,h3,gate,xn))[name=string(\"cat\")];\n", 2*D+3*H,S];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// Simple conv-only kernel: x -> conv(W, x) for a single matmul test
static NSString *gen_simple_conv(int IC, int OC, int S) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", IC, S];
    [m appendString:@CONV_CONST];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> W = const()[name=string(\"W\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w.bin\"), offset=uint64(64)))];\n", OC, IC, OC, IC];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W,x=x)[name=string(\"c\")];\n", OC, S];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// ---- Helpers ----
static void fill_random(float *buf, int n, float scale) {
    for (int i = 0; i < n; i++) buf[i] = scale * (2.0f * drand48() - 1.0f);
}

static double l2_norm(const float *buf, int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) sum += (double)buf[i] * buf[i];
    return sqrt(sum);
}

static int count_nonzero(const float *buf, int n) {
    int cnt = 0;
    for (int i = 0; i < n; i++) if (buf[i] != 0.0f) cnt++;
    return cnt;
}

static double max_abs(const float *buf, int n) {
    double mx = 0;
    for (int i = 0; i < n; i++) { double a = fabs(buf[i]); if (a > mx) mx = a; }
    return mx;
}

static void print_first_n(const char *label, const float *buf, int n, int show) {
    printf("  %s first %d: [", label, show);
    for (int i = 0; i < show && i < n; i++) printf("%.4f%s", buf[i], i<show-1?", ":"");
    printf("]\n");
}

static void dump_raw_fp16(IOSurfaceRef s, int total_elems, const char *label) {
    IOSurfaceLock(s, kIOSurfaceLockReadOnly, NULL);
    _Float16 *raw = (_Float16*)IOSurfaceGetBaseAddress(s);
    double sum = 0;
    int nz = 0;
    for (int i = 0; i < total_elems; i++) {
        float v = (float)raw[i];
        sum += (double)v * v;
        if (v != 0.0f) nz++;
    }
    printf("  %s raw fp16: L2=%.6f, nonzero=%d/%d\n", label, sqrt(sum), nz, total_elems);
    // show first 8
    printf("  %s first 8 raw: [", label);
    for (int i = 0; i < 8 && i < total_elems; i++) printf("%.4f%s", (float)raw[i], i<7?", ":"");
    printf("]\n");
    IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, NULL);
}

// ===================================================================
int main() {
    @autoreleasepool {
        setbuf(stdout, NULL);
        srand48(42);
        ane_init();

        printf("===============================================\n");
        printf("FFN Backward Debug Test\n");
        printf("===============================================\n\n");

        // ===== TEST 1: Simple conv at full size (DIM->HIDDEN) =====
        {
            int D = 768, H = 2048, S = 256;
            printf("--- TEST 1: Simple conv W2^T (DIM=%d -> HIDDEN=%d, SEQ=%d) ---\n", D, H, S);

            float *W2 = (float*)malloc(D * H * sizeof(float));
            float *input = (float*)malloc(D * S * sizeof(float));
            float *output = (float*)malloc(H * S * sizeof(float));

            // Initialize: small random weights, ones input
            float sc = 1.0f / sqrtf(D);
            fill_random(W2, D * H, sc);
            for (int i = 0; i < D * S; i++) input[i] = 0.01f;

            NSString *mil = gen_simple_conv(D, H, S);
            Kern *k = compile_kern_mil_w(mil, (@{
                @"@model_path/weights/w.bin": @{@"offset":@0, @"data":build_blob_t(W2, D, H)},
            }), D*S*2, H*S*2);

            if (!k) { printf("  COMPILE FAILED\n\n"); }
            else {
                io_write_fp16(k->ioIn, input, D, S);
                dump_raw_fp16(k->ioIn, D*S, "input");
                ane_eval(k);
                dump_raw_fp16(k->ioOut, H*S, "output");
                io_read_fp16(k->ioOut, output, 0, H, S);
                printf("  Output L2=%.6f, nonzero=%d/%d, max=%.6f\n",
                    l2_norm(output, H*S), count_nonzero(output, H*S), H*S, max_abs(output, H*S));
                print_first_n("output", output, H*S, 8);
                free_kern(k);
            }
            free(W2); free(input); free(output);
            printf("\n");
        }

        // ===== TEST 2: FFN forward at full size =====
        {
            int D = 768, H = 2048, S = 256;
            printf("--- TEST 2: FFN forward (DIM=%d, HIDDEN=%d, SEQ=%d) ---\n", D, H, S);

            float *W1 = (float*)malloc(H*D*sizeof(float));
            float *W2 = (float*)malloc(D*H*sizeof(float));
            float *W3 = (float*)malloc(H*D*sizeof(float));
            float *rms = (float*)malloc(D*sizeof(float));
            float *input = (float*)malloc(D*S*sizeof(float));
            int out_ch = 2*D + 3*H;
            float *output = (float*)malloc(out_ch*S*sizeof(float));

            float sc = 0.02f;
            fill_random(W1, H*D, sc);
            fill_random(W2, D*H, sc);
            fill_random(W3, H*D, sc);
            for (int i = 0; i < D; i++) rms[i] = 1.0f;
            fill_random(input, D*S, 0.1f);

            NSString *mil = gen_ffn_fwd_taps_p(D, H, S);
            Kern *k = compile_kern_mil_w(mil, (@{
                @"@model_path/weights/rms2.bin": @{@"offset":@0, @"data":build_blob(rms, 1, D)},
                @"@model_path/weights/w1.bin": @{@"offset":@0, @"data":build_blob(W1, H, D)},
                @"@model_path/weights/w3.bin": @{@"offset":@0, @"data":build_blob(W3, H, D)},
                @"@model_path/weights/w2.bin": @{@"offset":@0, @"data":build_blob(W2, D, H)},
            }), D*S*2, out_ch*S*2);

            if (!k) { printf("  COMPILE FAILED\n\n"); }
            else {
                io_write_fp16(k->ioIn, input, D, S);
                ane_eval(k);
                // Read FFN output (first D channels)
                io_read_fp16(k->ioOut, output, 0, D, S);
                printf("  ffn_out (ch 0..%d): L2=%.6f, nonzero=%d/%d, max=%.6f\n",
                    D-1, l2_norm(output, D*S), count_nonzero(output, D*S), D*S, max_abs(output, D*S));
                print_first_n("ffn_out", output, D*S, 8);
                // Read h1 (channels D..D+H-1)
                float *h1 = (float*)malloc(H*S*sizeof(float));
                io_read_fp16(k->ioOut, h1, D, H, S);
                printf("  h1 (ch %d..%d): L2=%.6f, nonzero=%d/%d\n",
                    D, D+H-1, l2_norm(h1, H*S), count_nonzero(h1, H*S), H*S);
                free(h1);
                free_kern(k);
            }
            free(W1); free(W2); free(W3); free(rms); free(input); free(output);
            printf("\n");
        }

        // ===== TEST 3: FFN backward at SMALL size =====
        {
            int D = 64, H = 128, S = 16;
            printf("--- TEST 3: FFN backward SMALL (DIM=%d, HIDDEN=%d, SEQ=%d) ---\n", D, H, S);

            float *W1 = (float*)malloc(H*D*sizeof(float));
            float *W2 = (float*)malloc(D*H*sizeof(float));
            float *W3 = (float*)malloc(H*D*sizeof(float));
            float sc = 0.1f;
            fill_random(W1, H*D, sc);
            fill_random(W2, D*H, sc);
            fill_random(W3, H*D, sc);

            // Input: [1, D+2H, 1, S]
            //   ch 0..D-1     = dffn (random)
            //   ch D..D+H-1   = h1 (random, must be non-zero for SiLU derivative)
            //   ch D+H..D+2H-1 = h3 (random)
            int in_ch = D + 2*H;
            float *dffn = (float*)malloc(D*S*sizeof(float));
            float *h1   = (float*)malloc(H*S*sizeof(float));
            float *h3   = (float*)malloc(H*S*sizeof(float));
            fill_random(dffn, D*S, 0.1f);
            fill_random(h1, H*S, 1.0f);  // larger values so sigmoid != 0.5
            fill_random(h3, H*S, 1.0f);

            int out_ch = D + 2*H;
            float *output = (float*)malloc(out_ch*S*sizeof(float));

            NSString *mil = gen_ffn_bwd_p(D, H, S);

            // Print MIL for inspection
            printf("  MIL program length: %lu bytes\n", (unsigned long)[mil lengthOfBytesUsingEncoding:NSUTF8StringEncoding]);

            Kern *k = compile_kern_mil_w(mil, (@{
                @"@model_path/weights/w2t.bin": @{@"offset":@0, @"data":build_blob_t(W2, D, H)},
                @"@model_path/weights/w1t.bin": @{@"offset":@0, @"data":build_blob_t(W1, H, D)},
                @"@model_path/weights/w3t.bin": @{@"offset":@0, @"data":build_blob_t(W3, H, D)},
            }), in_ch*S*2, out_ch*S*2);

            if (!k) { printf("  COMPILE FAILED\n\n"); }
            else {
                // Write input parts
                io_write_fp16_at(k->ioIn, 0, dffn, D, S);
                io_write_fp16_at(k->ioIn, D, h1, H, S);
                io_write_fp16_at(k->ioIn, D+H, h3, H, S);

                dump_raw_fp16(k->ioIn, in_ch*S, "input");
                ane_eval(k);
                dump_raw_fp16(k->ioOut, out_ch*S, "output");

                // Read output parts
                float *dx  = (float*)malloc(D*S*sizeof(float));
                float *dh1 = (float*)malloc(H*S*sizeof(float));
                float *dh3 = (float*)malloc(H*S*sizeof(float));
                io_read_fp16(k->ioOut, dx,  0,   D, S);
                io_read_fp16(k->ioOut, dh1, D,   H, S);
                io_read_fp16(k->ioOut, dh3, D+H, H, S);

                printf("  dx  (ch 0..%d):    L2=%.6f, nonzero=%d/%d, max=%.6f\n",
                    D-1, l2_norm(dx, D*S), count_nonzero(dx, D*S), D*S, max_abs(dx, D*S));
                printf("  dh1 (ch %d..%d): L2=%.6f, nonzero=%d/%d, max=%.6f\n",
                    D, D+H-1, l2_norm(dh1, H*S), count_nonzero(dh1, H*S), H*S, max_abs(dh1, H*S));
                printf("  dh3 (ch %d..%d): L2=%.6f, nonzero=%d/%d, max=%.6f\n",
                    D+H, D+2*H-1, l2_norm(dh3, H*S), count_nonzero(dh3, H*S), H*S, max_abs(dh3, H*S));
                print_first_n("dx", dx, D*S, 8);
                print_first_n("dh1", dh1, H*S, 8);
                print_first_n("dh3", dh3, H*S, 8);

                // CPU reference for dx = W1^T @ dh1 + W3^T @ dh3
                // But first we need the intermediate dh1, dh3 from the ANE computation
                // Skip CPU verification for now — focus on whether output is zero

                free(dx); free(dh1); free(dh3);
                free_kern(k);
            }
            free(W1); free(W2); free(W3);
            free(dffn); free(h1); free(h3); free(output);
            printf("\n");
        }

        // ===== TEST 4: FFN backward at FULL size =====
        {
            int D = 768, H = 2048, S = 256;
            printf("--- TEST 4: FFN backward FULL (DIM=%d, HIDDEN=%d, SEQ=%d) ---\n", D, H, S);

            float *W1 = (float*)malloc(H*D*sizeof(float));
            float *W2 = (float*)malloc(D*H*sizeof(float));
            float *W3 = (float*)malloc(H*D*sizeof(float));
            float sc = 0.02f;
            fill_random(W1, H*D, sc);
            fill_random(W2, D*H, sc);
            fill_random(W3, H*D, sc);

            int in_ch = D + 2*H;
            float *dffn = (float*)malloc(D*S*sizeof(float));
            float *h1   = (float*)malloc(H*S*sizeof(float));
            float *h3   = (float*)malloc(H*S*sizeof(float));
            fill_random(dffn, D*S, 0.1f);
            fill_random(h1, H*S, 1.0f);
            fill_random(h3, H*S, 1.0f);

            int out_ch = D + 2*H;
            float *output = (float*)malloc(out_ch*S*sizeof(float));

            NSString *mil = gen_ffn_bwd_p(D, H, S);

            Kern *k = compile_kern_mil_w(mil, (@{
                @"@model_path/weights/w2t.bin": @{@"offset":@0, @"data":build_blob_t(W2, D, H)},
                @"@model_path/weights/w1t.bin": @{@"offset":@0, @"data":build_blob_t(W1, H, D)},
                @"@model_path/weights/w3t.bin": @{@"offset":@0, @"data":build_blob_t(W3, H, D)},
            }), in_ch*S*2, out_ch*S*2);

            if (!k) { printf("  COMPILE FAILED\n\n"); }
            else {
                io_write_fp16_at(k->ioIn, 0, dffn, D, S);
                io_write_fp16_at(k->ioIn, D, h1, H, S);
                io_write_fp16_at(k->ioIn, D+H, h3, H, S);

                dump_raw_fp16(k->ioIn, in_ch*S, "input");
                ane_eval(k);
                dump_raw_fp16(k->ioOut, out_ch*S, "output");

                float *dx  = (float*)malloc(D*S*sizeof(float));
                float *dh1 = (float*)malloc(H*S*sizeof(float));
                float *dh3 = (float*)malloc(H*S*sizeof(float));
                io_read_fp16(k->ioOut, dx,  0,   D, S);
                io_read_fp16(k->ioOut, dh1, D,   H, S);
                io_read_fp16(k->ioOut, dh3, D+H, H, S);

                printf("  dx  (ch 0..%d):      L2=%.6f, nonzero=%d/%d, max=%.6f\n",
                    D-1, l2_norm(dx, D*S), count_nonzero(dx, D*S), D*S, max_abs(dx, D*S));
                printf("  dh1 (ch %d..%d):  L2=%.6f, nonzero=%d/%d, max=%.6f\n",
                    D, D+H-1, l2_norm(dh1, H*S), count_nonzero(dh1, H*S), H*S, max_abs(dh1, H*S));
                printf("  dh3 (ch %d..%d): L2=%.6f, nonzero=%d/%d, max=%.6f\n",
                    D+H, D+2*H-1, l2_norm(dh3, H*S), count_nonzero(dh3, H*S), H*S, max_abs(dh3, H*S));
                print_first_n("dx", dx, D*S, 8);
                print_first_n("dh1", dh1, H*S, 8);
                print_first_n("dh3", dh3, H*S, 8);

                free(dx); free(dh1); free(dh3);
                free_kern(k);
            }
            free(W1); free(W2); free(W3);
            free(dffn); free(h1); free(h3); free(output);
            printf("\n");
        }

        // ===== TEST 5: FFN backward — medium sizes to find the threshold =====
        {
            int test_dims[][3] = {
                {128, 256, 32},
                {256, 512, 64},
                {384, 1024, 128},
                {512, 1024, 256},
                {768, 2048, 64},   // full D/H, small SEQ
                {768, 2048, 128},  // full D/H, medium SEQ
            };
            int ntests = sizeof(test_dims)/sizeof(test_dims[0]);

            printf("--- TEST 5: FFN backward sweep across sizes ---\n");
            for (int t = 0; t < ntests; t++) {
                int D = test_dims[t][0], H = test_dims[t][1], S = test_dims[t][2];
                int in_ch = D + 2*H;
                int out_ch = D + 2*H;

                // Weight memory: W2t [H*D] + W1t [D*H] + W3t [D*H] = H*D + 2*D*H = 3*D*H
                double weight_mb = 3.0*D*H*2.0/1024/1024;
                double io_mb = (double)(in_ch + out_ch) * S * 2.0 / 1024 / 1024;
                printf("  D=%d H=%d S=%d  weights=%.1fMB  IO=%.1fMB  total=%.1fMB ... ",
                    D, H, S, weight_mb, io_mb, weight_mb + io_mb);

                float *W1 = (float*)malloc(H*D*sizeof(float));
                float *W2 = (float*)malloc(D*H*sizeof(float));
                float *W3 = (float*)malloc(H*D*sizeof(float));
                fill_random(W1, H*D, 0.05f);
                fill_random(W2, D*H, 0.05f);
                fill_random(W3, H*D, 0.05f);

                float *dffn = (float*)malloc(D*S*sizeof(float));
                float *h1   = (float*)malloc(H*S*sizeof(float));
                float *h3   = (float*)malloc(H*S*sizeof(float));
                fill_random(dffn, D*S, 0.1f);
                fill_random(h1, H*S, 1.0f);
                fill_random(h3, H*S, 1.0f);

                NSString *mil = gen_ffn_bwd_p(D, H, S);
                Kern *k = compile_kern_mil_w(mil, (@{
                    @"@model_path/weights/w2t.bin": @{@"offset":@0, @"data":build_blob_t(W2, D, H)},
                    @"@model_path/weights/w1t.bin": @{@"offset":@0, @"data":build_blob_t(W1, H, D)},
                    @"@model_path/weights/w3t.bin": @{@"offset":@0, @"data":build_blob_t(W3, H, D)},
                }), in_ch*S*2, out_ch*S*2);

                if (!k) { printf("COMPILE FAILED\n"); }
                else {
                    io_write_fp16_at(k->ioIn, 0, dffn, D, S);
                    io_write_fp16_at(k->ioIn, D, h1, H, S);
                    io_write_fp16_at(k->ioIn, D+H, h3, H, S);
                    ane_eval(k);

                    float *dx = (float*)malloc(D*S*sizeof(float));
                    io_read_fp16(k->ioOut, dx, 0, D, S);
                    int nz = count_nonzero(dx, D*S);
                    printf("dx L2=%.6f nonzero=%d/%d %s\n",
                        l2_norm(dx, D*S), nz, D*S,
                        nz == 0 ? "*** ALL ZERO ***" : "OK");
                    free(dx);
                    free_kern(k);
                }

                free(W1); free(W2); free(W3);
                free(dffn); free(h1); free(h3);
            }
            printf("\n");
        }

        // ===== TEST 6: Verify output surface isn't stale — zero before eval =====
        {
            int D = 64, H = 128, S = 16;
            printf("--- TEST 6: Output surface pre-check (D=%d, H=%d, S=%d) ---\n", D, H, S);

            float *W1 = (float*)malloc(H*D*sizeof(float));
            float *W2 = (float*)malloc(D*H*sizeof(float));
            float *W3 = (float*)malloc(H*D*sizeof(float));
            fill_random(W1, H*D, 0.1f);
            fill_random(W2, D*H, 0.1f);
            fill_random(W3, H*D, 0.1f);

            int in_ch = D + 2*H;
            int out_ch = D + 2*H;
            float *dffn = (float*)malloc(D*S*sizeof(float));
            float *h1   = (float*)malloc(H*S*sizeof(float));
            float *h3   = (float*)malloc(H*S*sizeof(float));
            fill_random(dffn, D*S, 0.1f);
            fill_random(h1, H*S, 1.0f);
            fill_random(h3, H*S, 1.0f);

            NSString *mil = gen_ffn_bwd_p(D, H, S);
            Kern *k = compile_kern_mil_w(mil, (@{
                @"@model_path/weights/w2t.bin": @{@"offset":@0, @"data":build_blob_t(W2, D, H)},
                @"@model_path/weights/w1t.bin": @{@"offset":@0, @"data":build_blob_t(W1, H, D)},
                @"@model_path/weights/w3t.bin": @{@"offset":@0, @"data":build_blob_t(W3, H, D)},
            }), in_ch*S*2, out_ch*S*2);

            if (!k) { printf("  COMPILE FAILED\n\n"); }
            else {
                // Write known pattern to output surface BEFORE eval
                IOSurfaceLock(k->ioOut, 0, NULL);
                _Float16 *raw_out = (_Float16*)IOSurfaceGetBaseAddress(k->ioOut);
                for (int i = 0; i < out_ch*S; i++) raw_out[i] = (_Float16)42.0f;
                IOSurfaceUnlock(k->ioOut, 0, NULL);
                printf("  Pre-filled output with 42.0\n");

                io_write_fp16_at(k->ioIn, 0, dffn, D, S);
                io_write_fp16_at(k->ioIn, D, h1, H, S);
                io_write_fp16_at(k->ioIn, D+H, h3, H, S);

                ane_eval(k);

                // Check: did ANE write to the output surface?
                IOSurfaceLock(k->ioOut, kIOSurfaceLockReadOnly, NULL);
                raw_out = (_Float16*)IOSurfaceGetBaseAddress(k->ioOut);
                int still_42 = 0, is_zero = 0, other = 0;
                for (int i = 0; i < out_ch*S; i++) {
                    float v = (float)raw_out[i];
                    if (v == 42.0f) still_42++;
                    else if (v == 0.0f) is_zero++;
                    else other++;
                }
                IOSurfaceUnlock(k->ioOut, kIOSurfaceLockReadOnly, NULL);
                printf("  After eval: still_42=%d, zero=%d, other=%d (of %d)\n",
                    still_42, is_zero, other, out_ch*S);
                if (still_42 == out_ch*S) printf("  *** ANE DID NOT WRITE TO OUTPUT SURFACE ***\n");
                else if (is_zero == out_ch*S) printf("  *** ANE WROTE ALL ZEROS ***\n");
                else printf("  ANE produced actual output values\n");

                dump_raw_fp16(k->ioOut, out_ch*S, "output_after_eval");
                free_kern(k);
            }
            free(W1); free(W2); free(W3);
            free(dffn); free(h1); free(h3);
            printf("\n");
        }

        printf("===============================================\n");
        printf("Total ANE compiles used: %d\n", g_compile_count);
        printf("===============================================\n");
    }
    return 0;
}
