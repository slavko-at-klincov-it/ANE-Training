// sweep_stacked.c — Exhaustive ANE stacked-conv TFLOPS sweep
// Finds peak configuration across channel/spatial/depth combos.
//
// Build: xcrun clang -O2 -Wall -fobjc-arc -I../../libane -o sweep_stacked sweep_stacked.c ../../libane/ane.m -framework Foundation -framework IOSurface -ldl

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../libane/ane.h"

#define MAX_RESULTS 256

typedef struct {
    int ch;
    int sp;
    int depth;
    double gflop;
    double ms;
    double tflops;
    int spill;
    int ok;
} Result;

static int cmp_tflops_desc(const void *a, const void *b) {
    double ta = ((const Result *)a)->tflops;
    double tb = ((const Result *)b)->tflops;
    if (tb > ta) return 1;
    if (tb < ta) return -1;
    return 0;
}

int main(void) {
    printf("\n  ====== ANE Stacked-Conv Sweep ======\n\n");

    if (ane_init() != 0) {
        printf("  ERROR: Failed to initialize ANE.\n");
        return 1;
    }

    ANEDeviceInfo info = ane_device_info();
    if (!info.has_ane) { printf("  ERROR: No ANE detected.\n"); return 1; }

    printf("  Chip:    %s, %d cores\n", info.arch ? info.arch : "?", info.num_cores);
    printf("  Thermal: %s\n", ane_thermal_state_str(ane_thermal_state()));
    printf("  Budget:  %d compiles max\n\n", ANE_COMPILE_BUDGET);

    const char *wname = "@model_path/weights/weight.bin";

    int channels[] = {256, 384, 512, 640, 768, 1024};
    int spatials[] = {32, 64, 128};
    int depths[]   = {32, 64, 128, 256};
    int nch = 6, nsp = 3, ndp = 4;
    int total = nch * nsp * ndp;
    // Note: depth=512 excluded — ANE compiler service rejects all depth>=512 MIL programs

    printf("  Sweep: %d ch x %d sp x %d depth = %d combos\n", nch, nsp, ndp, total);
    printf("  Warmup: 5 evals, Timed: 20 evals per config\n\n");

    printf("  %-6s %-4s %-5s %8s %9s %8s %5s\n",
           "Ch", "Sp", "Depth", "GFLOP", "Latency", "TFLOPS", "Spill");
    printf("  %-6s %-4s %-5s %8s %9s %8s %5s\n",
           "------", "----", "-----", "--------", "---------", "--------", "-----");

    Result results[MAX_RESULTS];
    int n_results = 0;

    for (int ic = 0; ic < nch; ic++) {
        for (int is = 0; is < nsp; is++) {
            for (int id = 0; id < ndp; id++) {
                int ch = channels[ic];
                int sp = spatials[is];
                int depth = depths[id];

                // Check compile budget before compiling
                int used = ane_compile_count();
                if (used >= ANE_COMPILE_BUDGET - 1) {
                    printf("\n  STOPPING: compile budget exhausted (%d/%d)\n", used, ANE_COMPILE_BUDGET);
                    goto done;
                }

                double gflop = 2.0 * ch * ch * sp * depth / 1e9;

                ANEWeight w = ane_weight_stacked(wname, ch, depth);
                char *mil = ane_mil_stacked_conv(ch, sp, depth, wname);
                size_t io_bytes = (size_t)ch * sp * 4;

                ANEKernel *k = ane_compile(mil, strlen(mil), &w, 1,
                                            1, &io_bytes, 1, &io_bytes,
                                            ANE_QOS_BACKGROUND);

                Result r = {ch, sp, depth, gflop, -1, 0, 0, 0};

                if (!k) {
                    printf("  %-6d %-4d %-5d %8.2f  FAILED\n", ch, sp, depth, gflop);
                    free(mil);
                    ane_weight_free(&w);
                    results[n_results++] = r;
                    continue;
                }

                r.spill = ane_sram_spill(k) ? 1 : 0;

                // Write input
                float *inp = (float *)calloc(ch * sp, sizeof(float));
                for (int i = 0; i < ch * sp; i++) inp[i] = 0.5f;
                ane_write(k, 0, inp, io_bytes);
                free(inp);

                // Warmup
                for (int i = 0; i < 5; i++) ane_eval(k, ANE_QOS_BACKGROUND);

                // Timed
                int iters = 20;
                struct timespec t0, t1;
                clock_gettime(CLOCK_MONOTONIC, &t0);
                for (int i = 0; i < iters; i++) ane_eval(k, ANE_QOS_BACKGROUND);
                clock_gettime(CLOCK_MONOTONIC, &t1);

                double ms = ((t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6) / iters;
                double tflops = gflop / ms;  // GFLOP / ms = TFLOPS

                r.ms = ms;
                r.tflops = tflops;
                r.ok = 1;

                printf("  %-6d %-4d %-5d %8.2f %7.3f ms %7.2f  %s\n",
                       ch, sp, depth, gflop, ms, tflops,
                       r.spill ? "YES" : "no");
                fflush(stdout);

                ane_free(k);
                free(mil);
                ane_weight_free(&w);

                results[n_results++] = r;
            }
        }
    }

done:
    // Sort by TFLOPS descending
    qsort(results, n_results, sizeof(Result), cmp_tflops_desc);

    printf("\n  ====== TOP 10 CONFIGURATIONS ======\n\n");
    printf("  %-4s %-6s %-4s %-5s %8s %9s %8s %5s\n",
           "Rank", "Ch", "Sp", "Depth", "GFLOP", "Latency", "TFLOPS", "Spill");
    printf("  %-4s %-6s %-4s %-5s %8s %9s %8s %5s\n",
           "----", "------", "----", "-----", "--------", "---------", "--------", "-----");

    int shown = 0;
    for (int i = 0; i < n_results && shown < 10; i++) {
        if (!results[i].ok) continue;
        shown++;
        printf("  #%-3d %-6d %-4d %-5d %8.2f %7.3f ms %7.2f  %s\n",
               shown, results[i].ch, results[i].sp, results[i].depth,
               results[i].gflop, results[i].ms, results[i].tflops,
               results[i].spill ? "YES" : "no");
    }

    // Print absolute peak
    if (n_results > 0 && results[0].ok) {
        printf("\n  *** PEAK: %.2f TFLOPS @ %dch sp%d x%d depth ***\n",
               results[0].tflops, results[0].ch, results[0].sp, results[0].depth);
    }

    printf("\n  Compiles used: %d / %d\n", ane_compile_count(), ANE_COMPILE_BUDGET);
    printf("  Thermal: %s\n\n", ane_thermal_state_str(ane_thermal_state()));

    return 0;
}
