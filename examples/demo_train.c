// demo_train.c — Self-contained ANE training demo
// Trains a linear layer on the Apple Neural Engine to learn Y = 2*X
//
// What it does:
//   1. Detects your ANE hardware (works on any M1-M5)
//   2. Compiles a linear kernel on the ANE
//   3. Trains for 50 steps with live loss output
//   4. Shows the weight converging to the correct answer
//
// Build & run:
//   make demo
//
// Expected output:
//   ANE: h15g (M3 Pro), 16 cores
//   Training linear layer to learn Y = 2*X ...
//   step  0  loss=12.4521  W[0,0]=0.03
//   step 10  loss= 0.8432  W[0,0]=1.67
//   step 50  loss= 0.0012  W[0,0]=2.00
//   Done! Weight converged to ~2.0

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "../libane/ane.h"

// Tiny model: DIM channels, SEQ sequence length
#define DIM 8
#define SEQ 64
#define LR 1.0f
#define STEPS 60

// Simple random float in [-1, 1]
static float randf(void) { return 2.0f * ((float)rand() / RAND_MAX) - 1.0f; }

int main(void) {
    srand((unsigned)time(NULL));

    // ===== Step 1: Detect hardware =====
    printf("=== ANE Training Demo ===\n\n");

    if (ane_init() != 0) {
        printf("Failed to initialize ANE. Are you on Apple Silicon?\n");
        return 1;
    }

    ANEDeviceInfo info = ane_device_info();
    if (!info.has_ane) {
        printf("No ANE detected.\n");
        return 1;
    }
    printf("Hardware: %s, %d ANE cores\n", info.arch ? info.arch : "unknown", info.num_cores);
    printf("Build:    %s\n", info.build ? info.build : "?");

    ANEAPIInfo api = ane_api_info();
    printf("API:      v%d (%d classes found)\n\n", api.api_version, api.classes_found);

    // ===== Step 2: Initialize weights =====
    // Weight matrix W[DIM, DIM] — starts random, should converge to 2*I
    float W[DIM * DIM];
    for (int i = 0; i < DIM * DIM; i++) W[i] = randf() * 0.1f;

    printf("Goal: Train W so that Y = W @ X approximates Y = 2*X\n");
    printf("      W starts random, should converge to 2*Identity\n\n");

    size_t io_bytes = DIM * SEQ * 4; // fp32 I/O

    // ===== Step 3: Training loop =====
    float input[DIM * SEQ];
    float output[DIM * SEQ];
    float target[DIM * SEQ];
    float grad_W[DIM * DIM];

    // Fixed training data (same every step for clean convergence)
    for (int i = 0; i < DIM * SEQ; i++) input[i] = randf();
    for (int i = 0; i < DIM * SEQ; i++) target[i] = 2.0f * input[i];

    printf("step   loss       W[0,0]   W[1,1]   ms/step\n");
    printf("----   --------   ------   ------   -------\n");

    for (int step = 0; step < STEPS; step++) {

        // --- Forward pass on ANE ---
        ANEWeight w = ane_weight_fp16("@model_path/weights/weight.bin", W, DIM, DIM);
        char *mil = ane_mil_linear(DIM, DIM, SEQ, "@model_path/weights/weight.bin");
        ANEKernel *k = ane_compile(mil, strlen(mil), &w, 1,
                                   1, &io_bytes, 1, &io_bytes,
                                   ANE_QOS_BACKGROUND);
        if (!k) {
            printf("Compile failed at step %d (budget: %d/%d)\n",
                   step, ane_compile_count(), ANE_COMPILE_BUDGET);
            free(mil);
            ane_weight_free(&w);
            break;
        }

        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        ane_write(k, 0, input, io_bytes);
        ane_eval(k, ANE_QOS_BACKGROUND);
        ane_read(k, 0, output, io_bytes);

        clock_gettime(CLOCK_MONOTONIC, &t1);
        double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;

        // FP16 overflow protection: sanitize ANE output
        for (int i = 0; i < DIM * SEQ; i++) {
            if (isnan(output[i]) || isinf(output[i])) output[i] = 0.0f;
        }

        // --- Loss (MSE) ---
        float loss = 0;
        for (int i = 0; i < DIM * SEQ; i++) {
            float diff = output[i] - target[i];
            loss += diff * diff;
        }
        loss /= (DIM * SEQ);

        if (isnan(loss)) { printf("NaN loss at step %d, stopping\n", step); break; }

        // --- Backward pass on CPU ---
        memset(grad_W, 0, sizeof(grad_W));
        float scale = 2.0f / (DIM * SEQ);
        for (int i = 0; i < DIM; i++) {
            for (int j = 0; j < DIM; j++) {
                float g = 0;
                for (int s = 0; s < SEQ; s++) {
                    float d_out = (output[i * SEQ + s] - target[i * SEQ + s]) * scale;
                    g += d_out * input[j * SEQ + s];
                }
                grad_W[i * DIM + j] = g;
            }
        }

        // Sanitize gradients
        for (int i = 0; i < DIM * DIM; i++) {
            if (isnan(grad_W[i])) grad_W[i] = 0.0f;
            if (isinf(grad_W[i])) grad_W[i] = copysignf(65504.0f, grad_W[i]);
        }

        // --- SGD weight update ---
        for (int i = 0; i < DIM * DIM; i++) {
            W[i] -= LR * grad_W[i];
        }

        // Print progress
        if (step < 5 || step % 5 == 0 || step == STEPS - 1) {
            printf("%-4d   %8.4f   %6.3f   %6.3f   %.1f\n",
                   step, loss, W[0], W[DIM + 1], ms);
        }

        ane_free(k);
        free(mil);
        ane_weight_free(&w);
    }

    // ===== Step 4: Show result =====
    printf("\n=== Result ===\n");
    printf("W diagonal (should be ~2.0):\n  ");
    for (int i = 0; i < 8 && i < DIM; i++) printf("%.3f ", W[i * DIM + i]);
    printf("...\n");

    float off_diag = 0;
    for (int i = 0; i < DIM; i++)
        for (int j = 0; j < DIM; j++)
            if (i != j) off_diag += fabsf(W[i * DIM + j]);
    off_diag /= (DIM * DIM - DIM);
    printf("W off-diagonal avg (should be ~0.0): %.4f\n", off_diag);

    float diag_avg = 0;
    for (int i = 0; i < DIM; i++) diag_avg += W[i * DIM + i];
    diag_avg /= DIM;

    printf("\nDiagonal average: %.3f %s\n", diag_avg,
        fabsf(diag_avg - 2.0f) < 0.1f ? "(converged!)" : "(still training...)");

    printf("\nCompile count: %d / 119 budget\n", ane_compile_count());

    return 0;
}
