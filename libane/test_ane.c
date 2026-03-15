// test_ane.c — Test libane API
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ane.h"

int main(void) {
    printf("=== libane Test Suite ===\n\n");

    // Init
    int rc = ane_init();
    if (rc != 0) { printf("FAIL: ane_init (rc=%d)\n", rc); return 1; }
    printf("ane_init: OK\n");

    // API diagnostics
    ane_print_diagnostics();

    // API info
    ANEAPIInfo api = ane_api_info();
    printf("\nAPI version: %d, classes: %d, build: %s\n",
        api.api_version, api.classes_found, api.macos_build ? api.macos_build : "?");

    // Device info
    ANEDeviceInfo info = ane_device_info();
    printf("\nDevice Info:\n");
    printf("  has_ane:    %s\n", info.has_ane ? "yes" : "no");
    printf("  cores:      %d\n", info.num_cores);
    printf("  arch:       %s\n", info.arch ? info.arch : "?");
    printf("  sub_type:   %s\n", info.sub_type ? info.sub_type : "?");
    printf("  variant:    %s\n", info.variant ? info.variant : "?");
    printf("  board:      %d\n", info.board_type);
    printf("  product:    %s\n", info.product ? info.product : "?");
    printf("  build:      %s\n", info.build ? info.build : "?");
    printf("  virtual:    %s\n", info.is_virtual ? "yes" : "no");

    // Test 1: Identity linear (256→256, seq=64)
    printf("\n--- Test 1: Identity Linear ---\n");
    {
        int in_ch = 256, out_ch = 256, seq = 64;

        // Create identity weight matrix
        float *W = (float *)calloc(out_ch * in_ch, sizeof(float));
        for (int i = 0; i < in_ch; i++) W[i * in_ch + i] = 1.0f;

        ANEWeight w = ane_weight_fp16("@model_path/weights/weight.bin", W, out_ch, in_ch);
        char *mil = ane_mil_linear(in_ch, out_ch, seq, "@model_path/weights/weight.bin");

        size_t in_bytes = in_ch * seq * 4;  // fp32
        size_t out_bytes = out_ch * seq * 4;

        ANEKernel *k = ane_compile(mil, strlen(mil), &w, 1,
                                   1, &in_bytes, 1, &out_bytes,
                                   ANE_QOS_BACKGROUND);
        if (!k) { printf("  FAIL: compile\n"); free(W); free(mil); ane_weight_free(&w); return 1; }
        printf("  compiled OK (compile_count=%d)\n", ane_compile_count());

        // Write input: all 1.0
        float *input = (float *)malloc(in_bytes);
        for (int i = 0; i < in_ch * seq; i++) input[i] = 1.0f;
        ane_write(k, 0, input, in_bytes);

        // Eval
        if (!ane_eval(k, ANE_QOS_BACKGROUND)) { printf("  FAIL: eval\n"); return 1; }

        // Read output
        float *output = (float *)malloc(out_bytes);
        ane_read(k, 0, output, out_bytes);

        printf("  output[0..3]: %.2f %.2f %.2f %.2f (expect 1.0)\n",
            output[0], output[1], output[2], output[3]);

        // Verify
        int ok = 1;
        for (int i = 0; i < out_ch * seq; i++) {
            if (fabsf(output[i] - 1.0f) > 0.01f) { ok = 0; break; }
        }
        printf("  result: %s\n", ok ? "PASS" : "FAIL");

        ane_free(k);
        free(W); free(input); free(output); free(mil);
        ane_weight_free(&w);
    }

    // Test 2: Projection (256→512, seq=64)
    printf("\n--- Test 2: Projection 256→512 ---\n");
    {
        int in_ch = 256, out_ch = 512, seq = 64;

        // Random-ish weight (small values)
        float *W = (float *)malloc(out_ch * in_ch * sizeof(float));
        for (int i = 0; i < out_ch * in_ch; i++)
            W[i] = (float)(i % 7 - 3) * 0.01f;

        ANEWeight w = ane_weight_fp16("@model_path/weights/weight.bin", W, out_ch, in_ch);
        char *mil = ane_mil_linear(in_ch, out_ch, seq, "@model_path/weights/weight.bin");

        size_t in_bytes = in_ch * seq * 4;
        size_t out_bytes = out_ch * seq * 4;

        ANEKernel *k = ane_compile(mil, strlen(mil), &w, 1,
                                   1, &in_bytes, 1, &out_bytes,
                                   ANE_QOS_BACKGROUND);
        if (!k) { printf("  FAIL: compile\n"); return 1; }
        printf("  compiled OK\n");

        float *input = (float *)malloc(in_bytes);
        for (int i = 0; i < in_ch * seq; i++) input[i] = 1.0f;
        ane_write(k, 0, input, in_bytes);

        if (!ane_eval(k, ANE_QOS_BACKGROUND)) { printf("  FAIL: eval\n"); return 1; }

        float *output = (float *)malloc(out_bytes);
        ane_read(k, 0, output, out_bytes);

        // For input all 1.0, output[i] = sum of row i of W
        float expected0 = 0;
        for (int j = 0; j < in_ch; j++) expected0 += W[0 * in_ch + j];
        printf("  output[0]: %.4f (expected ~%.4f)\n", output[0], expected0);
        printf("  match: %s\n", fabsf(output[0] - expected0) < 0.1f ? "PASS" : "FAIL");

        ane_free(k);
        free(W); free(input); free(output); free(mil);
        ane_weight_free(&w);
    }

    // Test 3: Zero-copy access
    printf("\n--- Test 3: Zero-Copy Access ---\n");
    {
        int ch = 256, seq = 64;
        float *W = (float *)calloc(ch * ch, sizeof(float));
        for (int i = 0; i < ch; i++) W[i * ch + i] = 2.0f; // scale by 2

        ANEWeight w = ane_weight_fp16("@model_path/weights/weight.bin", W, ch, ch);
        char *mil = ane_mil_linear(ch, ch, seq, "@model_path/weights/weight.bin");
        size_t bytes = ch * seq * 4;

        ANEKernel *k = ane_compile(mil, strlen(mil), &w, 1,
                                   1, &bytes, 1, &bytes,
                                   ANE_QOS_BACKGROUND);
        if (!k) { printf("  FAIL\n"); return 1; }

        // Zero-copy write
        ane_lock_input(k, 0);
        float *ptr = (float *)ane_input_ptr(k, 0);
        for (int i = 0; i < ch * seq; i++) ptr[i] = 3.0f;
        ane_unlock_input(k, 0);

        ane_eval(k, ANE_QOS_BACKGROUND);

        // Zero-copy read
        ane_lock_output(k, 0);
        float *out = (float *)ane_output_ptr(k, 0);
        printf("  output[0..3]: %.1f %.1f %.1f %.1f (expect 6.0)\n",
            out[0], out[1], out[2], out[3]);
        printf("  result: %s\n", fabsf(out[0] - 6.0f) < 0.1f ? "PASS" : "FAIL");
        ane_unlock_output(k, 0);

        ane_free(k);
        free(W); free(mil);
        ane_weight_free(&w);
    }

    printf("\n=== All Tests Complete (compiles: %d) ===\n", ane_compile_count());
    return 0;
}
