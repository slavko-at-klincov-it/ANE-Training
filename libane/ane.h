// ane.h — Clean C API for Apple Neural Engine (M-series chips)
// Built on reverse-engineered private APIs from AppleNeuralEngine.framework
// Tested on M3 Pro (h15g), macOS 26.3.1
//
// Usage:
//   ane_init();
//   ANEDeviceInfo info = ane_device_info();
//   ANEKernel *k = ane_compile_mil(mil_text, weights, n_weights, n_in, in_sizes, n_out, out_sizes);
//   ane_write(k, 0, data, bytes);
//   ane_eval(k, ANE_QOS_BACKGROUND);
//   ane_read(k, 0, data, bytes);
//   ane_free(k);

#ifndef ANE_H
#define ANE_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ===== Device Info =====

typedef struct {
    bool has_ane;
    int num_cores;
    int num_units;
    const char *arch;           // e.g. "h15g" (M3 Pro), "h16g" (M4)
    const char *sub_type;       // e.g. "h15"
    const char *variant;        // e.g. "g"
    int board_type;
    const char *product;        // "macOS" or "iOS"
    const char *build;          // e.g. "25D2128"
    bool is_virtual;
} ANEDeviceInfo;

// ===== QoS Levels =====

typedef enum {
    ANE_QOS_BACKGROUND       = 9,   // Fastest on M3 Pro! Best for training.
    ANE_QOS_UTILITY          = 17,
    ANE_QOS_DEFAULT          = 21,  // What most code uses
    ANE_QOS_USER_INITIATED   = 25,
    ANE_QOS_USER_INTERACTIVE = 33,
    ANE_QOS_REALTIME         = 0,   // Special mode
} ANEQoS;

// ===== Data Types =====

typedef enum {
    ANE_DTYPE_FP16 = 0,
    ANE_DTYPE_FP32 = 1,
    ANE_DTYPE_INT8 = 2,
} ANEDtype;

// ===== Weight Blob =====

typedef struct {
    uint8_t *data;
    size_t len;
    const char *name;   // e.g. "@model_path/weights/wq.bin"
} ANEWeight;

// ===== Kernel Handle =====

typedef struct ANEKernel ANEKernel;

// ===== API Version Detection =====

// Describes which private API surface was discovered at runtime.
// If Apple changes class names or selectors in a future macOS,
// libane will try known alternatives and report what it found.
typedef struct {
    int api_version;            // 1 = current (macOS 15-26), 0 = unknown
    const char *macos_build;    // e.g. "25D2128"
    const char *framework_path; // path that was loaded
    bool has_descriptor;        // _ANEInMemoryModelDescriptor found
    bool has_model;             // _ANEInMemoryModel found
    bool has_request;           // _ANERequest found
    bool has_iosurface_obj;     // _ANEIOSurfaceObject found
    bool has_client;            // _ANEClient found
    bool has_device_info;       // _ANEDeviceInfo found
    bool has_qos_mapper;        // _ANEQoSMapper found
    bool has_chaining;          // _ANEChainingRequest found
    bool has_perf_stats;        // _ANEPerformanceStats found
    bool has_buffer;            // _ANEBuffer found
    int  classes_found;         // total ANE classes discovered
    const char *descriptor_class;   // actual class name used (for debugging)
    const char *model_class;        // actual class name used
} ANEAPIInfo;

// ===== Initialization =====

// Load AppleNeuralEngine.framework and resolve private classes.
// Automatically detects API version and tries known class name alternatives.
// Returns 0 on success, -1 if framework not found, -2 if critical classes missing.
int ane_init(void);

// Query which API surface was detected. Call after ane_init().
ANEAPIInfo ane_api_info(void);

// Print a diagnostic report of API compatibility to stderr.
void ane_print_diagnostics(void);

// Query device hardware info. Call after ane_init().
ANEDeviceInfo ane_device_info(void);

// ===== Compilation =====

// Compile a MIL program into an ANE kernel.
// mil: UTF-8 MIL text
// weights: array of weight blobs (can be NULL if n_weights==0)
// n_weights: number of weight blobs
// n_inputs: number of input tensors
// input_sizes: byte size of each input IOSurface
// n_outputs: number of output tensors
// output_sizes: byte size of each output IOSurface
// qos: compilation quality of service
ANEKernel *ane_compile(const char *mil, size_t mil_len,
                       const ANEWeight *weights, int n_weights,
                       int n_inputs, const size_t *input_sizes,
                       int n_outputs, const size_t *output_sizes,
                       ANEQoS qos);

// ===== Evaluation =====

// Run the kernel on ANE. Returns true on success.
bool ane_eval(ANEKernel *k, ANEQoS qos);

// ===== Tensor I/O =====

// Write data to input tensor idx.
void ane_write(ANEKernel *k, int idx, const void *data, size_t bytes);

// Read data from output tensor idx.
void ane_read(ANEKernel *k, int idx, void *data, size_t bytes);

// Get raw IOSurface pointer for zero-copy access.
// Lock before use, unlock after.
void *ane_input_ptr(ANEKernel *k, int idx);
void *ane_output_ptr(ANEKernel *k, int idx);
void ane_lock_input(ANEKernel *k, int idx);
void ane_unlock_input(ANEKernel *k, int idx);
void ane_lock_output(ANEKernel *k, int idx);
void ane_unlock_output(ANEKernel *k, int idx);

// ===== Weight Blob Builders =====

// Build FP16 weight blob from float32 source. Caller frees .data.
ANEWeight ane_weight_fp16(const char *name, const float *src, int rows, int cols);

// Build transposed FP16 weight blob. Caller frees .data.
ANEWeight ane_weight_fp16_transposed(const char *name, const float *src, int rows, int cols);

// Build INT8 quantized weight blob. Returns scale via out_scale. Caller frees .data.
ANEWeight ane_weight_int8(const char *name, const float *src, int rows, int cols, float *out_scale);

// Free weight blob data.
void ane_weight_free(ANEWeight *w);

// ===== MIL Generation Helpers =====

// Generate MIL for a linear layer (1x1 conv, weights baked at compile time).
// Input: [1, in_ch, 1, seq] fp32 → Output: [1, out_ch, 1, seq] fp32
char *ane_mil_linear(int in_ch, int out_ch, int seq, const char *weight_name);

// Generate MIL for a dynamic linear layer (weights packed in input, NO recompilation needed).
// Input: [1, in_ch + in_ch*out_ch, 1, seq] fp32 → Output: [1, out_ch, 1, seq] fp32
// Pack activations in channels [0..in_ch), weights in channels [in_ch..in_ch+in_ch*out_ch).
// Weight layout: W[i][j] at channel (in_ch + i*in_ch + j), spatial position 0.
// Compile once, then update weights via IOSurface writes — zero recompilation.
char *ane_mil_linear_dynamic(int in_ch, int out_ch, int seq);

// Write a weight matrix into the input IOSurface for dynamic linear layers.
// Packs W[out_ch][in_ch] into the correct channel/spatial layout.
void ane_write_dynamic_weights(ANEKernel *k, int idx, const float *W,
                                int in_ch, int out_ch, int seq);

// Generate MIL for N stacked 1x1 convolutions (for benchmarks). Caller frees.
// Input/Output: [1, ch, 1, sp] fp32
char *ane_mil_stacked_conv(int ch, int sp, int depth, const char *weight_name);

// Build multi-chunk weight blob with random FP16 data for stacked conv. Caller frees .data.
ANEWeight ane_weight_stacked(const char *name, int ch, int depth);

// Generate MIL header (program declaration + buildInfo).
char *ane_mil_header(void);

// ===== Lifecycle =====

// Compile budget: ANE silently fails after ~119 compilations per process.
// Use ane_compile_count() to monitor, restart process before hitting limit.
#define ANE_COMPILE_BUDGET 119
#define ANE_COMPILE_SAFE_LIMIT 110

// Get current compile count (for exec() restart budgeting).
int ane_compile_count(void);

// EXPERIMENTAL: Reload weights via disk patching (delta compilation).
// In practice, ANE bakes weights into HWX at compile time and does not
// re-read them on reload. Use ane_mil_linear_dynamic() instead for training.
// Returns true if reload succeeded, false otherwise.
bool ane_reload_weights(ANEKernel *k, const ANEWeight *weights, int n_weights, ANEQoS qos);

// Free a compiled kernel and all resources.
void ane_free(ANEKernel *k);

#ifdef __cplusplus
}
#endif

#endif // ANE_H
