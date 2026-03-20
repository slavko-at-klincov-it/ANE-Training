// sweep_training.c — Focused ANE kernel shape sweep for Stories-110M training
// Tests shapes matching actual transformer layer dimensions:
//   dim=768, hidden=2048, plus neighboring configs
//
// Goal: Find optimal single-kernel throughput for training workload shapes
//       and map SRAM spillover boundaries (~32MB)
//
// Build: xcrun clang -O2 -Wall -Wno-deprecated-declarations -fobjc-arc \
//        -o sweep_training sweep_training.c ../libane/ane.m \
//        -framework Foundation -framework IOSurface -ldl -lm

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "../libane/ane.h"

#define MAX_RESULTS   256
#define WARMUP_ITERS  10
#define TIMED_ITERS   100
#define COMPILE_STOP  108  // Leave room for safety

typedef struct {
    int ch_in;
    int ch_out;
    int sp;
    double tflops;
    double latency_ms;
    double gflop;
    double wt_mb;
    double act_mb;  // activation memory (input + output)
    int spill;
    int failed;
    const char *label;
} Result;

static int cmp_tflops_desc(const void *a, const void *b) {
    double da = ((const Result *)a)->tflops;
    double db = ((const Result *)b)->tflops;
    if (db > da) return 1;
    if (db < da) return -1;
    return 0;
}

// Test configuration
typedef struct {
    int ch_in;
    int ch_out;
    int sp;
    const char *label;
} Config;

int main(void) {
    printf("\n");
    printf("  ====== ANE TRAINING SHAPE SWEEP (Stories-110M Focus) ======\n");
    printf("\n");

    if (ane_init() != 0) { printf("  ERROR: ane_init failed\n"); return 1; }
    ANEDeviceInfo info = ane_device_info();
    if (!info.has_ane) { printf("  ERROR: No ANE\n"); return 1; }

    printf("  Chip:    %s, %d cores\n", info.arch ? info.arch : "?", info.num_cores);
    printf("  Thermal: %s\n", ane_thermal_state_str(ane_thermal_state()));
    printf("  Budget:  %d compiles (stopping at %d)\n", ANE_COMPILE_BUDGET, COMPILE_STOP);
    printf("  Iters:   %d warmup + %d timed\n", WARMUP_ITERS, TIMED_ITERS);
    printf("\n");

    // ===== Phase 1: Exact Stories-110M shapes =====
    // The transformer uses: Q/K/V projections (768->768), FFN up (768->2048), FFN down (2048->768)
    // With various sequence lengths (spatial dim)

    Config configs[] = {
        // -- Stories-110M exact shapes --
        // Q/K/V projections: 768 -> 768
        {768,  768,  1,    "QKV sp1"},
        {768,  768,  64,   "QKV sp64"},
        {768,  768,  128,  "QKV sp128"},
        {768,  768,  256,  "QKV sp256"},
        {768,  768,  512,  "QKV sp512"},
        {768,  768,  1024, "QKV sp1024"},
        {768,  768,  2048, "QKV sp2048"},

        // FFN up: 768 -> 2048
        {768,  2048, 1,    "FFN-up sp1"},
        {768,  2048, 64,   "FFN-up sp64"},
        {768,  2048, 128,  "FFN-up sp128"},
        {768,  2048, 256,  "FFN-up sp256"},
        {768,  2048, 512,  "FFN-up sp512"},
        {768,  2048, 1024, "FFN-up sp1024"},
        {768,  2048, 2048, "FFN-up sp2048"},

        // FFN down: 2048 -> 768
        {2048, 768,  1,    "FFN-dn sp1"},
        {2048, 768,  64,   "FFN-dn sp64"},
        {2048, 768,  128,  "FFN-dn sp128"},
        {2048, 768,  256,  "FFN-dn sp256"},
        {2048, 768,  512,  "FFN-dn sp512"},
        {2048, 768,  1024, "FFN-dn sp1024"},
        {2048, 768,  2048, "FFN-dn sp2048"},

        // -- Neighboring channel sizes (training-relevant) --
        // Smaller model dims
        {512,  512,  128,  "512x512 sp128"},
        {512,  512,  256,  "512x512 sp256"},
        {512,  512,  512,  "512x512 sp512"},
        {512,  1024, 128,  "512x1024 sp128"},
        {512,  1024, 256,  "512x1024 sp256"},
        {1024, 512,  128,  "1024x512 sp128"},
        {1024, 512,  256,  "1024x512 sp256"},

        // Larger model dims
        {1024, 1024, 128,  "1024x1024 sp128"},
        {1024, 1024, 256,  "1024x1024 sp256"},
        {1024, 1024, 512,  "1024x1024 sp512"},
        {1024, 2048, 128,  "1024x2048 sp128"},
        {1024, 2048, 256,  "1024x2048 sp256"},
        {2048, 1024, 128,  "2048x1024 sp128"},
        {2048, 1024, 256,  "2048x1024 sp256"},

        // Large spatial to test SRAM boundaries
        {768,  768,  4096, "QKV sp4096"},
        {768,  2048, 4096, "FFN-up sp4096"},
        {2048, 768,  4096, "FFN-dn sp4096"},

        // Power-of-2 channel combos for comparison
        {64,   64,   256,  "64x64 sp256"},
        {128,  128,  256,  "128x128 sp256"},
        {256,  256,  256,  "256x256 sp256"},
        {256,  512,  256,  "256x512 sp256"},
        {512,  256,  256,  "512x256 sp256"},

        // Very large channels (SRAM spill test)
        {2048, 2048, 64,   "2048x2048 sp64"},
        {2048, 2048, 128,  "2048x2048 sp128"},
        {2048, 2048, 256,  "2048x2048 sp256"},
        {2048, 2048, 512,  "2048x2048 sp512"},
        {4096, 4096, 64,   "4096x4096 sp64"},

        // Stories-110M output projection: 768 -> vocab_size (32000)
        // Too large for single kernel, but let's test smaller vocab-like
        {768,  4096, 64,   "768x4096 sp64"},
        {768,  4096, 128,  "768x4096 sp128"},
        {768,  4096, 256,  "768x4096 sp256"},

        // Asymmetric shapes testing dispatch efficiency
        {256,  2048, 256,  "256x2048 sp256"},
        {2048, 256,  256,  "2048x256 sp256"},
        {384,  2048, 128,  "384x2048 sp128"},
        {640,  2048, 128,  "640x2048 sp128"},

        // High spatial, small channels (attention-like)
        {64,   64,   1024, "64x64 sp1024"},
        {64,   64,   2048, "64x64 sp2048"},
        {64,   64,   4096, "64x64 sp4096"},
        {128,  128,  1024, "128x128 sp1024"},
        {128,  128,  2048, "128x128 sp2048"},

        // Very high spatial with model dim
        {512,  512,  1024, "512x512 sp1024"},
        {512,  512,  2048, "512x512 sp2048"},
        {512,  512,  4096, "512x512 sp4096"},
        {256,  256,  1024, "256x256 sp1024"},
        {256,  256,  2048, "256x256 sp2048"},
        {256,  256,  4096, "256x256 sp4096"},
    };

    int n_configs = sizeof(configs) / sizeof(configs[0]);

    // Filter by size constraints
    int valid = 0;
    for (int i = 0; i < n_configs; i++) {
        size_t wt = (size_t)configs[i].ch_in * configs[i].ch_out * 2;
        size_t io_in = (size_t)configs[i].ch_in * configs[i].sp * 4;
        size_t io_out = (size_t)configs[i].ch_out * configs[i].sp * 4;
        if (wt <= 64*1024*1024 && io_in <= 64*1024*1024 && io_out <= 64*1024*1024)
            valid++;
    }

    printf("  Configs: %d total, %d valid (within IOSurface limits)\n", n_configs, valid);
    printf("\n");

    const char *wname = "@model_path/weights/weight.bin";
    Result results[MAX_RESULTS];
    int n_results = 0;

    printf("  %-20s %-6s %-6s %-6s %7s %7s %7s %9s %7s %5s\n",
           "Label", "ch_in", "ch_out", "sp", "Wt(MB)", "Act(MB)", "GFLOP", "Latency", "TFLOPS", "Spill");
    printf("  %-20s %-6s %-6s %-6s %7s %7s %7s %9s %7s %5s\n",
           "--------------------", "------", "------", "------", "-------", "-------", "-------",
           "---------", "-------", "-----");

    for (int i = 0; i < n_configs; i++) {
        int ci = configs[i].ch_in;
        int co = configs[i].ch_out;
        int sp = configs[i].sp;
        const char *label = configs[i].label;

        size_t wt_bytes = (size_t)ci * co * 2;
        size_t in_bytes = (size_t)ci * sp * 4;
        size_t out_bytes = (size_t)co * sp * 4;

        if (wt_bytes > 64*1024*1024 || in_bytes > 64*1024*1024 || out_bytes > 64*1024*1024) {
            printf("  %-20s %-6d %-6d %-6d  SKIP (too large)\n", label, ci, co, sp);
            continue;
        }

        if (ane_compile_count() >= COMPILE_STOP) {
            printf("\n  STOPPING: compile budget (%d/%d)\n", ane_compile_count(), ANE_COMPILE_BUDGET);
            break;
        }

        double gflop = 2.0 * ci * co * sp / 1e9;
        double wt_mb = (double)wt_bytes / (1024.0 * 1024.0);
        double act_mb = (double)(in_bytes + out_bytes) / (1024.0 * 1024.0);

        float *dummy = (float *)calloc((size_t)ci * co, sizeof(float));
        if (!dummy) { printf("  %-20s  OOM\n", label); continue; }
        for (size_t j = 0; j < (size_t)ci * co; j++) dummy[j] = 0.01f;

        ANEWeight w = ane_weight_fp16(wname, dummy, co, ci);
        char *mil = ane_mil_linear(ci, co, sp, wname);

        ANEKernel *k = ane_compile(mil, strlen(mil), &w, 1,
                                    1, &in_bytes, 1, &out_bytes,
                                    ANE_QOS_BACKGROUND);
        free(dummy);

        Result r = {0};
        r.ch_in = ci;
        r.ch_out = co;
        r.sp = sp;
        r.gflop = gflop;
        r.wt_mb = wt_mb;
        r.act_mb = act_mb;
        r.label = label;

        if (!k) {
            r.failed = 1;
            printf("  %-20s %-6d %-6d %-6d %6.1f %6.1f %6.2f  FAILED\n",
                   label, ci, co, sp, wt_mb, act_mb, gflop);
            fflush(stdout);
            free(mil);
            ane_weight_free(&w);
            if (n_results < MAX_RESULTS) results[n_results++] = r;
            continue;
        }

        r.spill = ane_sram_spill(k) ? 1 : 0;

        float *inp = (float *)calloc((size_t)ci * sp, sizeof(float));
        if (inp) {
            for (size_t j = 0; j < (size_t)ci * sp; j++) inp[j] = 0.5f;
            ane_write(k, 0, inp, in_bytes);
            free(inp);
        }

        for (int t = 0; t < WARMUP_ITERS; t++)
            ane_eval(k, ANE_QOS_BACKGROUND);

        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (int t = 0; t < TIMED_ITERS; t++)
            ane_eval(k, ANE_QOS_BACKGROUND);
        clock_gettime(CLOCK_MONOTONIC, &t1);

        double ms = ((t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6) / TIMED_ITERS;
        r.latency_ms = ms;
        r.tflops = (ms > 0) ? gflop / ms : 0;

        printf("  %-20s %-6d %-6d %-6d %6.1f %6.1f %6.2f %7.3f ms %6.2f %s\n",
               label, ci, co, sp, wt_mb, act_mb, gflop, ms, r.tflops,
               r.spill ? " SPILL" : "");
        fflush(stdout);

        ane_free(k);
        free(mil);
        ane_weight_free(&w);
        if (n_results < MAX_RESULTS) results[n_results++] = r;
    }

    printf("\n  Compiles used: %d / %d\n", ane_compile_count(), ANE_COMPILE_BUDGET);
    printf("  Thermal: %s\n", ane_thermal_state_str(ane_thermal_state()));

    // Sort by TFLOPS
    qsort(results, n_results, sizeof(Result), cmp_tflops_desc);

    // ===== TOP 15 =====
    printf("\n  ===== TOP 15 CONFIGURATIONS =====\n\n");
    printf("  %-4s %-20s %-6s %-6s %-6s %7s %7s %9s %7s %5s\n",
           "Rank", "Label", "ch_in", "ch_out", "sp", "Act(MB)", "GFLOP", "Latency", "TFLOPS", "Spill");
    printf("  %-4s %-20s %-6s %-6s %-6s %7s %7s %9s %7s %5s\n",
           "----", "--------------------", "------", "------", "------", "-------", "-------",
           "---------", "-------", "-----");

    int shown = 0;
    for (int i = 0; i < n_results && shown < 15; i++) {
        if (results[i].failed) continue;
        shown++;
        printf("  #%-3d %-20s %-6d %-6d %-6d %6.1f %6.2f %7.3f ms %6.2f %s\n",
               shown, results[i].label, results[i].ch_in, results[i].ch_out, results[i].sp,
               results[i].act_mb, results[i].gflop, results[i].latency_ms, results[i].tflops,
               results[i].spill ? " SPILL" : "");
    }

    // ===== Stories-110M specific analysis =====
    printf("\n  ===== STORIES-110M LAYER ANALYSIS =====\n\n");
    printf("  Layer type vs spatial dimension:\n\n");
    printf("  %-12s", "Spatial:");
    int test_sps[] = {1, 64, 128, 256, 512, 1024, 2048, 4096};
    for (int s = 0; s < 8; s++) printf(" %6d", test_sps[s]);
    printf("\n");

    // QKV row
    printf("  %-12s", "QKV 768x768");
    for (int s = 0; s < 8; s++) {
        int found = 0;
        for (int i = 0; i < n_results; i++) {
            if (results[i].ch_in == 768 && results[i].ch_out == 768 &&
                results[i].sp == test_sps[s] && !results[i].failed) {
                printf(" %5.2fT", results[i].tflops);
                found = 1;
                break;
            }
        }
        if (!found) printf("     - ");
    }
    printf("\n");

    // FFN up row
    printf("  %-12s", "FFN 768x2048");
    for (int s = 0; s < 8; s++) {
        int found = 0;
        for (int i = 0; i < n_results; i++) {
            if (results[i].ch_in == 768 && results[i].ch_out == 2048 &&
                results[i].sp == test_sps[s] && !results[i].failed) {
                printf(" %5.2fT", results[i].tflops);
                found = 1;
                break;
            }
        }
        if (!found) printf("     - ");
    }
    printf("\n");

    // FFN down row
    printf("  %-12s", "FFN 2048x768");
    for (int s = 0; s < 8; s++) {
        int found = 0;
        for (int i = 0; i < n_results; i++) {
            if (results[i].ch_in == 2048 && results[i].ch_out == 768 &&
                results[i].sp == test_sps[s] && !results[i].failed) {
                printf(" %5.2fT", results[i].tflops);
                found = 1;
                break;
            }
        }
        if (!found) printf("     - ");
    }
    printf("\n");

    // ===== SRAM analysis =====
    printf("\n  ===== SRAM SPILL ANALYSIS =====\n\n");
    int spill_count = 0, ok_count = 0;
    double spill_threshold_act = 1e18;
    for (int i = 0; i < n_results; i++) {
        if (results[i].failed) continue;
        if (results[i].spill) {
            spill_count++;
            if (results[i].act_mb < spill_threshold_act)
                spill_threshold_act = results[i].act_mb;
        } else {
            ok_count++;
        }
    }
    printf("  Spilled: %d / %d configs\n", spill_count, spill_count + ok_count);
    if (spill_count > 0)
        printf("  Lowest spilling activation size: %.1f MB\n", spill_threshold_act);
    else
        printf("  No SRAM spills detected in any configuration!\n");

    // Print all configs that spilled
    if (spill_count > 0) {
        printf("\n  Configs that spilled:\n");
        for (int i = 0; i < n_results; i++) {
            if (!results[i].failed && results[i].spill) {
                printf("    %s  (wt=%.1fMB, act=%.1fMB, %.2f TFLOPS)\n",
                       results[i].label, results[i].wt_mb, results[i].act_mb,
                       results[i].tflops);
            }
        }
    }

    // ===== Efficiency analysis =====
    printf("\n  ===== THROUGHPUT vs COMPUTE SIZE =====\n\n");
    printf("  %-20s %8s %8s %8s\n", "Label", "GFLOP", "TFLOPS", "Eff(%)");
    printf("  %-20s %8s %8s %8s\n", "--------------------", "--------", "--------", "--------");
    double peak = (n_results > 0 && !results[0].failed) ? results[0].tflops : 1.0;
    for (int i = 0; i < n_results; i++) {
        if (results[i].failed) continue;
        double eff = results[i].tflops / peak * 100.0;
        printf("  %-20s %8.3f %7.2f  %6.1f%%\n",
               results[i].label, results[i].gflop, results[i].tflops, eff);
    }

    printf("\n  ===== SUMMARY =====\n");
    if (n_results > 0 && !results[0].failed) {
        printf("  Peak single-kernel: %.2f TFLOPS (%s)\n", results[0].tflops, results[0].label);
    }
    // Find best for each Stories-110M layer type
    for (int i = 0; i < n_results; i++) {
        if (results[i].failed) continue;
        if (results[i].ch_in == 768 && results[i].ch_out == 768) {
            printf("  Best QKV (768x768): %.2f TFLOPS @ sp%d (%.3f ms)\n",
                   results[i].tflops, results[i].sp, results[i].latency_ms);
            break;
        }
    }
    for (int i = 0; i < n_results; i++) {
        if (results[i].failed) continue;
        if (results[i].ch_in == 768 && results[i].ch_out == 2048) {
            printf("  Best FFN-up (768x2048): %.2f TFLOPS @ sp%d (%.3f ms)\n",
                   results[i].tflops, results[i].sp, results[i].latency_ms);
            break;
        }
    }
    for (int i = 0; i < n_results; i++) {
        if (results[i].failed) continue;
        if (results[i].ch_in == 2048 && results[i].ch_out == 768) {
            printf("  Best FFN-down (2048x768): %.2f TFLOPS @ sp%d (%.3f ms)\n",
                   results[i].tflops, results[i].sp, results[i].latency_ms);
            break;
        }
    }
    printf("\n");

    return 0;
}
