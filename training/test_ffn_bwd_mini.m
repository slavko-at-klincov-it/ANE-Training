// test_ffn_bwd_mini.m — Minimal test of ffnBwd kernel to isolate zero-output bug
// Tests: 1) single conv W2t at full size, 2) silu derivative chain, 3) full ffnBwd
#import <Foundation/Foundation.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#include <math.h>
// From stories_config.h — just the constants we need
#define DIM 768
#define HIDDEN 2048
#define SEQ 256
#define HD 64
#define HEADS 12
#define NLAYERS 12
#define VOCAB 32000

static Class g_D, g_I, g_AR, g_AIO;
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
static NSData *build_blob_t(const float *w, int rows, int cols) {
    int ws=cols*rows*2, tot=128+ws;
    uint8_t *b=(uint8_t*)calloc(tot,1);
    b[0]=1;b[4]=2;b[64]=0xEF;b[65]=0xBE;b[66]=0xAD;b[67]=0xDE;b[68]=1;
    *(uint32_t*)(b+72)=ws;*(uint32_t*)(b+80)=128;
    _Float16 *fp16=(_Float16*)(b+128);
    for(int i=0;i<rows;i++) for(int j=0;j<cols;j++) fp16[j*rows+i]=(_Float16)w[i*cols+j];
    return [NSData dataWithBytesNoCopy:b length:tot freeWhenDone:YES];
}
static NSData *build_blob(const float *w, int rows, int cols) {
    int ws=rows*cols*2, tot=128+ws;
    uint8_t *b=(uint8_t*)calloc(tot,1);
    b[0]=1;b[4]=2;b[64]=0xEF;b[65]=0xBE;b[66]=0xAD;b[67]=0xDE;b[68]=1;
    *(uint32_t*)(b+72)=ws;*(uint32_t*)(b+80)=128;
    _Float16 *fp16=(_Float16*)(b+128);
    for(int i=0;i<rows*cols;i++) fp16[i]=(_Float16)w[i];
    return [NSData dataWithBytesNoCopy:b length:tot freeWhenDone:YES];
}

typedef struct { id model; IOSurfaceRef ioIn, ioOut; NSString *td; } TestKern;

static TestKern compile_test(NSString *mil, NSDictionary *weights, int ic_bytes, int oc_bytes) {
    TestKern k = {nil, NULL, NULL, nil};
    NSData *md = [mil dataUsingEncoding:NSUTF8StringEncoding];
    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D, @selector(modelWithMILText:weights:optionsPlist:), md, weights, nil);
    if (!desc) { printf("  desc=NULL\n"); return k; }
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
        printf("  compile FAIL: %s\n", e?[[e description] UTF8String]:""); return k; }
    if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e)) {
        printf("  load FAIL\n"); return k; }
    k.model = mdl; k.ioIn = make_surface(ic_bytes); k.ioOut = make_surface(oc_bytes); k.td = td;
    return k;
}

static BOOL eval_test(TestKern *k) {
    id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), k->ioIn);
    id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), k->ioOut);
    id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        @[wI], @[@0], @[wO], @[@0], nil, nil, @0);
    NSError *e = nil;
    return ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
        k->model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
}

static double io_norm(_Float16 *buf, int count) {
    double n = 0;
    for(int i=0; i<count; i++) n += (float)buf[i] * (float)buf[i];
    return sqrt(n);
}

int main() {
    @autoreleasepool {
        setbuf(stdout, NULL);
        ane_init();
        srand48(42);

        printf("=== FFN Backward Kernel Debug ===\n");
        printf("DIM=%d HIDDEN=%d SEQ=%d\n\n", DIM, HIDDEN, SEQ);

        float *W2 = (float*)malloc(DIM*HIDDEN*sizeof(float));
        float *W1 = (float*)malloc(HIDDEN*DIM*sizeof(float));
        float *W3 = (float*)malloc(HIDDEN*DIM*sizeof(float));
        float sc = 1.0f/sqrtf(DIM);
        for(int i=0;i<DIM*HIDDEN;i++) W2[i]=sc*(2*drand48()-1);
        for(int i=0;i<HIDDEN*DIM;i++){W1[i]=sc*(2*drand48()-1);W3[i]=sc*(2*drand48()-1);}

        // Prepare input data
        float *dffn_f32 = (float*)malloc(SEQ*DIM*sizeof(float));
        float *h1_f32 = (float*)malloc(SEQ*HIDDEN*sizeof(float));
        float *h3_f32 = (float*)malloc(SEQ*HIDDEN*sizeof(float));
        for(int i=0;i<SEQ*DIM;i++) dffn_f32[i] = 0.1f * sinf(i*0.01f);
        for(int i=0;i<SEQ*HIDDEN;i++){h1_f32[i]=0.5f*sinf(i*0.007f);h3_f32[i]=0.5f*cosf(i*0.011f);}

        // ============================================================
        // TEST 1: Single conv W2t @ dffn (minimal)
        // ============================================================
        printf("TEST 1: Single conv W2t @ dffn\n");
        {
            NSString *mil = [NSString stringWithFormat:
                @"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
                "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
                "{\"coremltools-version\", \"9.0\"}})]\n{\n"
                "    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
                "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
                "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
                "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
                "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
                "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
                "        tensor<fp16, [%d,%d,1,1]> W2t = const()[name=string(\"W2t\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w2t.bin\"), offset=uint64(64)))];\n"
                "        tensor<fp16, [1,%d,1,%d]> out = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W2t,x=x)[name=string(\"out\")];\n"
                "    } -> (out);\n}\n",
                DIM, SEQ, HIDDEN, DIM, HIDDEN, DIM, HIDDEN, SEQ];
            NSDictionary *wd = @{@"@model_path/weights/w2t.bin": @{@"offset":@0, @"data":build_blob_t(W2,DIM,HIDDEN)}};
            TestKern k = compile_test(mil, wd, DIM*SEQ*2, HIDDEN*SEQ*2);
            if (!k.model) { printf("  SKIP\n"); }
            else {
                IOSurfaceLock(k.ioIn, 0, NULL);
                _Float16 *inp = (_Float16*)IOSurfaceGetBaseAddress(k.ioIn);
                for(int i=0;i<DIM*SEQ;i++) inp[i] = (_Float16)dffn_f32[i];
                IOSurfaceUnlock(k.ioIn, 0, NULL);
                BOOL ok = eval_test(&k);
                IOSurfaceLock(k.ioOut, kIOSurfaceLockReadOnly, NULL);
                _Float16 *out = (_Float16*)IOSurfaceGetBaseAddress(k.ioOut);
                double n = io_norm(out, HIDDEN*SEQ);
                printf("  eval=%s |output|=%.6f (expect >0)\n", ok?"OK":"FAIL", n);
                IOSurfaceUnlock(k.ioOut, kIOSurfaceLockReadOnly, NULL);
            }
        }

        // ============================================================
        // TEST 2: Full gen_ffn_bwd() kernel
        // ============================================================
        printf("\nTEST 2: Full gen_ffn_bwd kernel\n");
        {
            NSMutableString *m = [NSMutableString string];
            [m appendString:@"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
                "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
                "{\"coremltools-version\", \"9.0\"}})]\n{\n"];
            [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", DIM+2*HIDDEN, SEQ];
            [m appendString:@"        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
                "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
                "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
                "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
                "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"];
            [m appendFormat:@"        tensor<int32, [4]> bd = const()[name=string(\"bd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
            [m appendFormat:@"        tensor<int32, [4]> sd = const()[name=string(\"sd\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, SEQ];
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dffn = slice_by_size(x=x,begin=bd,size=sd)[name=string(\"s0\")];\n", DIM, SEQ];
            [m appendFormat:@"        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", DIM];
            [m appendFormat:@"        tensor<int32, [4]> s1 = const()[name=string(\"s1\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", HIDDEN, SEQ];
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> h1 = slice_by_size(x=x,begin=b1,size=s1)[name=string(\"s1x\")];\n", HIDDEN, SEQ];
            [m appendFormat:@"        tensor<int32, [4]> b3 = const()[name=string(\"b3\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", DIM+HIDDEN];
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> h3 = slice_by_size(x=x,begin=b3,size=s1)[name=string(\"s3x\")];\n", HIDDEN, SEQ];
            [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> W2t = const()[name=string(\"W2t\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w2t.bin\"), offset=uint64(64)))];\n", HIDDEN, DIM, HIDDEN, DIM];
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dsilu = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W2t,x=dffn)[name=string(\"cw2\")];\n", HIDDEN, SEQ];
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> sig = sigmoid(x=h1)[name=string(\"sg\")];\n", HIDDEN, SEQ];
            [m appendString:@"        fp16 one = const()[name=string(\"one\"), val=fp16(1.0)];\n"];
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> oms = sub(x=one,y=sig)[name=string(\"oms\")];\n", HIDDEN, SEQ];
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> homs = mul(x=h1,y=oms)[name=string(\"homs\")];\n", HIDDEN, SEQ];
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> brk = add(x=one,y=homs)[name=string(\"brk\")];\n", HIDDEN, SEQ];
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dsd = mul(x=sig,y=brk)[name=string(\"dsd\")];\n", HIDDEN, SEQ];
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> t1 = mul(x=dsilu,y=h3)[name=string(\"t1\")];\n", HIDDEN, SEQ];
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dh1 = mul(x=t1,y=dsd)[name=string(\"dh1\")];\n", HIDDEN, SEQ];
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> slh = mul(x=h1,y=sig)[name=string(\"slh\")];\n", HIDDEN, SEQ];
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dh3 = mul(x=dsilu,y=slh)[name=string(\"dh3\")];\n", HIDDEN, SEQ];
            [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> W1t = const()[name=string(\"W1t\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w1t.bin\"), offset=uint64(64)))];\n", DIM, HIDDEN, DIM, HIDDEN];
            [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> W3t = const()[name=string(\"W3t\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w3t.bin\"), offset=uint64(64)))];\n", DIM, HIDDEN, DIM, HIDDEN];
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dx1 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W1t,x=dh1)[name=string(\"cw1\")];\n", DIM, SEQ];
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dx3 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W3t,x=dh3)[name=string(\"cw3\")];\n", DIM, SEQ];
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dx = add(x=dx1,y=dx3)[name=string(\"adx\")];\n", DIM, SEQ];
            [m appendString:@"        int32 cax = const()[name=string(\"cax\"), val=int32(1)];\n"];
            [m appendString:@"        bool cid = const()[name=string(\"cid\"), val=bool(false)];\n"];
            [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = concat(axis=cax,interleave=cid,values=(dx,dh1,dh3))[name=string(\"cat\")];\n", DIM+2*HIDDEN, SEQ];
            [m appendString:@"    } -> (out);\n}\n"];

            NSDictionary *wd = @{
                @"@model_path/weights/w2t.bin": @{@"offset":@0, @"data":build_blob_t(W2,DIM,HIDDEN)},
                @"@model_path/weights/w1t.bin": @{@"offset":@0, @"data":build_blob_t(W1,HIDDEN,DIM)},
                @"@model_path/weights/w3t.bin": @{@"offset":@0, @"data":build_blob_t(W3,HIDDEN,DIM)},
            };
            TestKern k = compile_test(m, wd, (DIM+2*HIDDEN)*SEQ*2, (DIM+2*HIDDEN)*SEQ*2);
            if (!k.model) { printf("  SKIP\n"); }
            else {
                // Write fp16 input: concat(dffn, h1, h3)
                IOSurfaceLock(k.ioIn, 0, NULL);
                _Float16 *inp = (_Float16*)IOSurfaceGetBaseAddress(k.ioIn);
                // dffn at channels 0..DIM-1
                for(int c=0;c<DIM;c++) for(int t=0;t<SEQ;t++) inp[c*SEQ+t] = (_Float16)dffn_f32[t*DIM+c];
                // h1 at channels DIM..DIM+HIDDEN-1
                for(int c=0;c<HIDDEN;c++) for(int t=0;t<SEQ;t++) inp[(DIM+c)*SEQ+t] = (_Float16)h1_f32[t*HIDDEN+c];
                // h3 at channels DIM+HIDDEN..DIM+2*HIDDEN-1
                for(int c=0;c<HIDDEN;c++) for(int t=0;t<SEQ;t++) inp[(DIM+HIDDEN+c)*SEQ+t] = (_Float16)h3_f32[t*HIDDEN+c];
                IOSurfaceUnlock(k.ioIn, 0, NULL);

                double inp_norm = io_norm((_Float16*)IOSurfaceGetBaseAddress(k.ioIn), (DIM+2*HIDDEN)*SEQ);
                printf("  |input|=%.6f\n", inp_norm);

                BOOL ok = eval_test(&k);
                IOSurfaceLock(k.ioOut, kIOSurfaceLockReadOnly, NULL);
                _Float16 *out = (_Float16*)IOSurfaceGetBaseAddress(k.ioOut);
                double n_dx = io_norm(out, DIM*SEQ);
                double n_dh1 = io_norm(out + DIM*SEQ, HIDDEN*SEQ);
                double n_dh3 = io_norm(out + (DIM+HIDDEN)*SEQ, HIDDEN*SEQ);
                double n_total = io_norm(out, (DIM+2*HIDDEN)*SEQ);
                printf("  eval=%s |dx|=%.6f |dh1|=%.6f |dh3|=%.6f |total|=%.6f\n",
                    ok?"OK":"FAIL", n_dx, n_dh1, n_dh3, n_total);
                // Print first 8 values
                printf("  out[0..7]=");
                for(int i=0;i<8;i++) printf("%.4f ", (float)out[i]);
                printf("\n");
                IOSurfaceUnlock(k.ioOut, kIOSurfaceLockReadOnly, NULL);
            }
        }

        printf("\nDONE\n");
        free(W2); free(W1); free(W3);
        free(dffn_f32); free(h1_f32); free(h3_f32);
    }
    return 0;
}
