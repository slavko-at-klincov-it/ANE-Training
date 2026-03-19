// ane_gpu.m — Metal GPU acceleration for ANE Training
// Loads Metal.framework at runtime via dlopen, accesses all Metal objects
// via objc_msgSend (same pattern as ane.m). No compile-time Metal linkage.

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <objc/message.h>
#include <dlfcn.h>
#include "ane_gpu.h"

// ===== Metal object references (resolved at runtime) =====
static id    g_device                  = nil;   // id<MTLDevice>
static id    g_queue                   = nil;   // id<MTLCommandQueue>
static id    g_pso_matmul              = nil;   // id<MTLComputePipelineState>
static id    g_pso_rmsnorm             = nil;
static id    g_pso_softmax             = nil;
static bool  g_gpu_init                = false;

// dlopen handle
static void *g_metal_handle = NULL;

// ===== Metal C function pointer: MTLCreateSystemDefaultDevice =====
typedef id (*MTLCreateSystemDefaultDeviceFn)(void);
static MTLCreateSystemDefaultDeviceFn g_createDevice = NULL;

// ===== Device info cache =====
static char g_device_name[128] = {0};
static uint64_t g_max_buffer_length = 0;
static bool g_supports_shared_events = false;

// ===== Metal Shading Language (MSL) inline shaders =====

static const char *MSL_MATMUL =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "\n"
    "struct MatmulParams {\n"
    "    uint M;\n"
    "    uint K;\n"
    "    uint N;\n"
    "};\n"
    "\n"
    "kernel void matmul(\n"
    "    device const float *A [[buffer(0)]],\n"
    "    device const float *B [[buffer(1)]],\n"
    "    device float *C       [[buffer(2)]],\n"
    "    constant MatmulParams &params [[buffer(3)]],\n"
    "    uint2 gid [[thread_position_in_grid]])\n"
    "{\n"
    "    uint row = gid.y;\n"
    "    uint col = gid.x;\n"
    "    if (row >= params.M || col >= params.N) return;\n"
    "\n"
    "    float sum = 0.0f;\n"
    "    for (uint i = 0; i < params.K; i++) {\n"
    "        sum += A[row * params.K + i] * B[i * params.N + col];\n"
    "    }\n"
    "    C[row * params.N + col] = sum;\n"
    "}\n";

static const char *MSL_RMSNORM =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "\n"
    "struct RMSNormParams {\n"
    "    uint dim;\n"
    "    uint seq;\n"
    "};\n"
    "\n"
    "kernel void rmsnorm(\n"
    "    device const float *x      [[buffer(0)]],\n"
    "    device const float *weight  [[buffer(1)]],\n"
    "    device float *out           [[buffer(2)]],\n"
    "    constant RMSNormParams &params [[buffer(3)]],\n"
    "    uint gid [[thread_position_in_grid]])\n"
    "{\n"
    "    if (gid >= params.seq) return;\n"
    "\n"
    "    uint offset = gid * params.dim;\n"
    "    float ss = 0.0f;\n"
    "    for (uint i = 0; i < params.dim; i++) {\n"
    "        float v = x[offset + i];\n"
    "        ss += v * v;\n"
    "    }\n"
    "    float rms = rsqrt(ss / float(params.dim) + 1e-6f);\n"
    "    for (uint i = 0; i < params.dim; i++) {\n"
    "        out[offset + i] = x[offset + i] * weight[i] * rms;\n"
    "    }\n"
    "}\n";

static const char *MSL_SOFTMAX =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "\n"
    "struct SoftmaxParams {\n"
    "    uint vocab;\n"
    "    uint seq;\n"
    "};\n"
    "\n"
    "kernel void softmax_kernel(\n"
    "    device const float *x [[buffer(0)]],\n"
    "    device float *out     [[buffer(1)]],\n"
    "    constant SoftmaxParams &params [[buffer(2)]],\n"
    "    uint gid [[thread_position_in_grid]])\n"
    "{\n"
    "    if (gid >= params.seq) return;\n"
    "\n"
    "    uint offset = gid * params.vocab;\n"
    "\n"
    "    // Find max for numerical stability\n"
    "    float max_val = x[offset];\n"
    "    for (uint i = 1; i < params.vocab; i++) {\n"
    "        max_val = max(max_val, x[offset + i]);\n"
    "    }\n"
    "\n"
    "    // Compute exp and sum\n"
    "    float sum = 0.0f;\n"
    "    for (uint i = 0; i < params.vocab; i++) {\n"
    "        float e = exp(x[offset + i] - max_val);\n"
    "        out[offset + i] = e;\n"
    "        sum += e;\n"
    "    }\n"
    "\n"
    "    // Normalize\n"
    "    float inv_sum = 1.0f / sum;\n"
    "    for (uint i = 0; i < params.vocab; i++) {\n"
    "        out[offset + i] *= inv_sum;\n"
    "    }\n"
    "}\n";

// ===== Helper: compile MSL source into a pipeline state =====
// Returns a retained PSO. The library is only needed during compilation
// and is released when this function returns (PSO retains what it needs).
static id compile_shader(const char *source, const char *func_name) {
    @autoreleasepool {
        NSString *src = [NSString stringWithUTF8String:source];
        NSError *error = nil;

        // id<MTLLibrary> lib = [device newLibraryWithSource:src options:nil error:&error]
        id lib = ((id(*)(id, SEL, id, id, NSError**))objc_msgSend)(
            g_device, sel_registerName("newLibraryWithSource:options:error:"),
            src, nil, &error);
        if (!lib) {
            fprintf(stderr, "ane_gpu: shader compile failed (%s): %s\n",
                    func_name, error ? [[error description] UTF8String] : "unknown");
            return nil;
        }

        // id<MTLFunction> func = [lib newFunctionWithName:@"..."]
        NSString *fname = [NSString stringWithUTF8String:func_name];
        id func = ((id(*)(id, SEL, id))objc_msgSend)(
            lib, sel_registerName("newFunctionWithName:"), fname);
        if (!func) {
            fprintf(stderr, "ane_gpu: function '%s' not found in compiled library\n", func_name);
            return nil;
        }

        // id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:func error:&error]
        error = nil;
        id pso = ((id(*)(id, SEL, id, NSError**))objc_msgSend)(
            g_device, sel_registerName("newComputePipelineStateWithFunction:error:"),
            func, &error);
        if (!pso) {
            fprintf(stderr, "ane_gpu: pipeline creation failed (%s): %s\n",
                    func_name, error ? [[error description] UTF8String] : "unknown");
            return nil;
        }

        return pso;
    }
}

// ===== Public API =====

int ane_gpu_init(void) {
    if (g_gpu_init) return 0;

    @autoreleasepool {
        // --- Step 1: Load Metal.framework via dlopen ---
        const char *metal_paths[] = {
            "/System/Library/Frameworks/Metal.framework/Metal",
            "/System/Library/PrivateFrameworks/Metal.framework/Metal",
            NULL
        };

        for (int i = 0; metal_paths[i]; i++) {
            g_metal_handle = dlopen(metal_paths[i], RTLD_NOW);
            if (g_metal_handle) break;
        }
        if (!g_metal_handle) {
            fprintf(stderr, "ane_gpu: Metal.framework not available\n");
            return -1;
        }

        // --- Step 2: Resolve MTLCreateSystemDefaultDevice ---
        g_createDevice = (MTLCreateSystemDefaultDeviceFn)dlsym(g_metal_handle, "MTLCreateSystemDefaultDevice");
        if (!g_createDevice) {
            fprintf(stderr, "ane_gpu: MTLCreateSystemDefaultDevice not found\n");
            dlclose(g_metal_handle);
            g_metal_handle = NULL;
            return -1;
        }

        // --- Step 3: Create device ---
        g_device = g_createDevice();
        if (!g_device) {
            fprintf(stderr, "ane_gpu: no Metal device available\n");
            dlclose(g_metal_handle);
            g_metal_handle = NULL;
            return -1;
        }

        // --- Step 4: Cache device info ---
        NSString *name = ((NSString*(*)(id, SEL))objc_msgSend)(
            g_device, sel_registerName("name"));
        if (name && [name isKindOfClass:[NSString class]]) {
            strncpy(g_device_name, [name UTF8String], sizeof(g_device_name) - 1);
        }

        g_max_buffer_length = ((uint64_t(*)(id, SEL))objc_msgSend)(
            g_device, sel_registerName("maxBufferLength"));

        g_supports_shared_events = [g_device respondsToSelector:sel_registerName("newSharedEvent")];

        // --- Step 5: Create command queue ---
        g_queue = ((id(*)(id, SEL))objc_msgSend)(
            g_device, sel_registerName("newCommandQueue"));
        if (!g_queue) {
            fprintf(stderr, "ane_gpu: failed to create command queue\n");
            g_device = nil;
            dlclose(g_metal_handle);
            g_metal_handle = NULL;
            return -1;
        }

        // --- Step 6: Compile shaders ---
        g_pso_matmul = compile_shader(MSL_MATMUL, "matmul");
        if (!g_pso_matmul) {
            fprintf(stderr, "ane_gpu: matmul shader compilation failed\n");
            g_queue = nil; g_device = nil;
            dlclose(g_metal_handle); g_metal_handle = NULL;
            return -1;
        }

        g_pso_rmsnorm = compile_shader(MSL_RMSNORM, "rmsnorm");
        if (!g_pso_rmsnorm) {
            fprintf(stderr, "ane_gpu: rmsnorm shader compilation failed\n");
            g_pso_matmul = nil; g_queue = nil; g_device = nil;
            dlclose(g_metal_handle); g_metal_handle = NULL;
            return -1;
        }

        g_pso_softmax = compile_shader(MSL_SOFTMAX, "softmax_kernel");
        if (!g_pso_softmax) {
            fprintf(stderr, "ane_gpu: softmax shader compilation failed\n");
            g_pso_matmul = nil; g_pso_rmsnorm = nil; g_queue = nil; g_device = nil;
            dlclose(g_metal_handle); g_metal_handle = NULL;
            return -1;
        }

        g_gpu_init = true;
        return 0;
    }
}

bool ane_has_gpu(void) {
    return g_gpu_init;
}

ANEGPUInfo ane_gpu_info(void) {
    ANEGPUInfo info = {0};
    if (!g_gpu_init) return info;
    info.name = g_device_name;
    info.max_buffer_length = g_max_buffer_length;
    info.supports_shared_events = g_supports_shared_events;
    return info;
}

// ===== Sync API =====

struct ANEGPUSync {
    id shared_event;    // id<MTLSharedEvent>
    uint64_t counter;   // monotonic signal counter
};

ANEGPUSync *ane_gpu_sync_create(void) {
    if (!g_gpu_init || !g_supports_shared_events) return NULL;

    @autoreleasepool {
        id event = ((id(*)(id, SEL))objc_msgSend)(
            g_device, sel_registerName("newSharedEvent"));
        if (!event) return NULL;

        ANEGPUSync *sync = (ANEGPUSync *)calloc(1, sizeof(ANEGPUSync));
        sync->shared_event = event;
        sync->counter = 0;
        return sync;
    }
}

void ane_gpu_sync_free(ANEGPUSync *sync) {
    if (!sync) return;
    sync->shared_event = nil;
    free(sync);
}

void ane_gpu_sync_signal(ANEGPUSync *sync, uint64_t value) {
    if (!sync || !sync->shared_event) return;

    @autoreleasepool {
        id cmdBuf = ((id(*)(id, SEL))objc_msgSend)(
            g_queue, sel_registerName("commandBuffer"));
        if (!cmdBuf) return;

        ((void(*)(id, SEL, id, uint64_t))objc_msgSend)(
            cmdBuf, sel_registerName("encodeSignalEvent:value:"),
            sync->shared_event, value);

        ((void(*)(id, SEL))objc_msgSend)(cmdBuf, sel_registerName("commit"));
    }
}

void ane_gpu_sync_wait(ANEGPUSync *sync, uint64_t value) {
    if (!sync || !sync->shared_event) return;

    @autoreleasepool {
        id cmdBuf = ((id(*)(id, SEL))objc_msgSend)(
            g_queue, sel_registerName("commandBuffer"));
        if (!cmdBuf) return;

        ((void(*)(id, SEL, id, uint64_t))objc_msgSend)(
            cmdBuf, sel_registerName("encodeWaitForEvent:value:"),
            sync->shared_event, value);

        ((void(*)(id, SEL))objc_msgSend)(cmdBuf, sel_registerName("commit"));
        ((void(*)(id, SEL))objc_msgSend)(cmdBuf, sel_registerName("waitUntilCompleted"));
    }
}

// ===== Helper: run a compute shader synchronously =====
// Uses __unsafe_unretained for the buffer array to avoid ARC writeback issues
// with C arrays of id. The buffers are already retained by callers' locals.
static void gpu_dispatch(id pso, __unsafe_unretained id *buffers, int n_buffers,
                         const void *params, size_t params_size,
                         int params_index,
                         uint32_t grid_x, uint32_t grid_y, uint32_t grid_z) {
    @autoreleasepool {
        // Create command buffer
        id cmdBuf = ((id(*)(id, SEL))objc_msgSend)(
            g_queue, sel_registerName("commandBuffer"));
        if (!cmdBuf) return;

        // Create compute encoder
        id encoder = ((id(*)(id, SEL))objc_msgSend)(
            cmdBuf, sel_registerName("computeCommandEncoder"));
        if (!encoder) return;

        // Set pipeline state
        ((void(*)(id, SEL, id))objc_msgSend)(
            encoder, sel_registerName("setComputePipelineState:"), pso);

        // Set buffers
        for (int i = 0; i < n_buffers; i++) {
            ((void(*)(id, SEL, id, uint64_t, uint64_t))objc_msgSend)(
                encoder, sel_registerName("setBuffer:offset:atIndex:"),
                buffers[i], (uint64_t)0, (uint64_t)i);
        }

        // Set params as bytes
        if (params && params_size > 0) {
            ((void(*)(id, SEL, const void*, uint64_t, uint64_t))objc_msgSend)(
                encoder, sel_registerName("setBytes:length:atIndex:"),
                params, (uint64_t)params_size, (uint64_t)params_index);
        }

        // Get max threads per threadgroup
        uint64_t max_tg = ((uint64_t(*)(id, SEL))objc_msgSend)(
            pso, sel_registerName("maxTotalThreadsPerThreadgroup"));
        if (max_tg == 0) max_tg = 256;

        // Threadgroup sizing
        uint32_t tg_x, tg_y, tg_z;
        tg_z = 1;
        if (grid_y > 1) {
            // 2D dispatch (matmul)
            tg_x = 16;
            tg_y = 16;
            if ((uint64_t)tg_x * tg_y > max_tg) {
                tg_x = 8;
                tg_y = 8;
            }
        } else {
            // 1D dispatch (rmsnorm, softmax)
            tg_x = (uint32_t)(max_tg > 256 ? 256 : max_tg);
            tg_y = 1;
        }

        // MTLSize is 3 x NSUInteger (uint64_t on 64-bit)
        typedef struct { uint64_t w, h, d; } MTLSize;
        MTLSize gridSize = { grid_x, grid_y, grid_z };
        MTLSize tgSize   = { tg_x, tg_y, tg_z };

        // dispatchThreads:threadsPerThreadgroup:
        ((void(*)(id, SEL, MTLSize, MTLSize))objc_msgSend)(
            encoder, sel_registerName("dispatchThreads:threadsPerThreadgroup:"),
            gridSize, tgSize);

        // End encoding, commit, wait
        ((void(*)(id, SEL))objc_msgSend)(encoder, sel_registerName("endEncoding"));
        ((void(*)(id, SEL))objc_msgSend)(cmdBuf, sel_registerName("commit"));
        ((void(*)(id, SEL))objc_msgSend)(cmdBuf, sel_registerName("waitUntilCompleted"));
    }
}

// ===== Helper: create MTLBuffer from host data =====
static id make_buffer(const void *data, size_t bytes) {
    // MTLResourceStorageModeShared = 0
    return ((id(*)(id, SEL, const void*, uint64_t, uint64_t))objc_msgSend)(
        g_device, sel_registerName("newBufferWithBytes:length:options:"),
        data, (uint64_t)bytes, (uint64_t)0);
}

// ===== Helper: create empty MTLBuffer =====
static id make_buffer_empty(size_t bytes) {
    return ((id(*)(id, SEL, uint64_t, uint64_t))objc_msgSend)(
        g_device, sel_registerName("newBufferWithLength:options:"),
        (uint64_t)bytes, (uint64_t)0);
}

// ===== Helper: read MTLBuffer contents to host =====
static void read_buffer(id buffer, void *dst, size_t bytes) {
    void *ptr = ((void*(*)(id, SEL))objc_msgSend)(
        buffer, sel_registerName("contents"));
    if (ptr) memcpy(dst, ptr, bytes);
}

// ===== GPU Compute Operations =====

void ane_gpu_matmul(const float *A, const float *B, float *C,
                    int M, int K, int N) {
    if (!g_gpu_init || !g_pso_matmul) return;

    @autoreleasepool {
        id bufA = make_buffer(A, (size_t)M * K * sizeof(float));
        id bufB = make_buffer(B, (size_t)K * N * sizeof(float));
        id bufC = make_buffer_empty((size_t)M * N * sizeof(float));
        if (!bufA || !bufB || !bufC) return;

        struct { uint32_t M, K, N; } params = {
            (uint32_t)M, (uint32_t)K, (uint32_t)N
        };

        __unsafe_unretained id buffers[] = { bufA, bufB, bufC };
        gpu_dispatch(g_pso_matmul, buffers, 3,
                     &params, sizeof(params), 3,
                     (uint32_t)N, (uint32_t)M, 1);

        read_buffer(bufC, C, (size_t)M * N * sizeof(float));
    }
}

void ane_gpu_rmsnorm(const float *x, const float *weight, float *out,
                     int dim, int seq) {
    if (!g_gpu_init || !g_pso_rmsnorm) return;

    @autoreleasepool {
        id bufX   = make_buffer(x, (size_t)seq * dim * sizeof(float));
        id bufW   = make_buffer(weight, (size_t)dim * sizeof(float));
        id bufOut = make_buffer_empty((size_t)seq * dim * sizeof(float));
        if (!bufX || !bufW || !bufOut) return;

        struct { uint32_t dim, seq; } params = {
            (uint32_t)dim, (uint32_t)seq
        };

        __unsafe_unretained id buffers[] = { bufX, bufW, bufOut };
        gpu_dispatch(g_pso_rmsnorm, buffers, 3,
                     &params, sizeof(params), 3,
                     (uint32_t)seq, 1, 1);

        read_buffer(bufOut, out, (size_t)seq * dim * sizeof(float));
    }
}

void ane_gpu_softmax(const float *x, float *out, int vocab, int seq) {
    if (!g_gpu_init || !g_pso_softmax) return;

    @autoreleasepool {
        id bufX   = make_buffer(x, (size_t)seq * vocab * sizeof(float));
        id bufOut = make_buffer_empty((size_t)seq * vocab * sizeof(float));
        if (!bufX || !bufOut) return;

        struct { uint32_t vocab, seq; } params = {
            (uint32_t)vocab, (uint32_t)seq
        };

        __unsafe_unretained id buffers[] = { bufX, bufOut };
        gpu_dispatch(g_pso_softmax, buffers, 2,
                     &params, sizeof(params), 2,
                     (uint32_t)seq, 1, 1);

        read_buffer(bufOut, out, (size_t)seq * vocab * sizeof(float));
    }
}

// ===== Cleanup =====

void ane_gpu_shutdown(void) {
    if (!g_gpu_init) return;

    @autoreleasepool {
        g_pso_matmul  = nil;
        g_pso_rmsnorm = nil;
        g_pso_softmax = nil;
        g_queue       = nil;
        g_device      = nil;
    }

    // Don't dlclose Metal — ObjC runtime references persist
    g_gpu_init = false;
}
