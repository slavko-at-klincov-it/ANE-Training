// sweep_single.c — Exhaustive single-kernel TFLOPS sweep on ANE
// Finds the maximum-throughput kernel shape by testing all valid
// (ch_in, ch_out, spatial) combinations within ANE size constraints.
//
// Discovered peak: 11.64 TFLOPS sustained @ 512x4096 sp4096 on M3 Pro (h15g)
// Previous best:   4.73 TFLOPS @ 768x2048 sp256 — 2.5x improvement
//
// Key findings from exhaustive sweep (5 phases, ~250 configurations):
//   - Larger spatial (2048-4096) dramatically improves throughput by
//     amortizing per-eval dispatch overhead across more FLOPs
//   - ch_in=512 with large ch_out (2048-4096) is the sweet spot
//   - SRAM spill warnings appear but don't cause actual spills
//   - 0 of all tested configs actually spilled SRAM
//   - Burst (20 iters) vs sustained (200 iters) can differ by ~20%
//
// Build: make sweep_single
//   or:  xcrun clang -O2 -Wall -Wno-deprecated-declarations -fobjc-arc \
//        -I../../libane -o sweep_single sweep_single.c ../../libane/ane.m \
//        -framework Foundation -framework IOSurface -ldl -lm
//
// Run: ./sweep_single

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "../../libane/ane.h"

// ===== Sweep grid =====
static const int CH_IN[]  = {256, 384, 448, 512, 640, 768, 1024, 1536, 2048, 3072, 4096};
#define N_CI 11
static const int CH_OUT[] = {256, 512, 768, 1024, 1536, 2048, 3072, 3584, 4096};
#define N_CO 9
static const int SP[]     = {64, 128, 256, 512, 1024, 2048, 4096};
#define N_SP 7

#define MAX_WEIGHT_BYTES  (64 * 1024 * 1024)
#define MAX_IO_BYTES      (64 * 1024 * 1024)
#define MAX_RESULTS       1024
#define WARMUP_ITERS      10
#define TIMED_ITERS       200
#define COMPILE_STOP      110

typedef struct {
    int ch_in;
    int ch_out;
    int sp;
    double tflops;
    double latency_ms;
    double gflop;
    double wt_mb;
    int spill;
    int failed;
} Result;

static int cmp_tflops_desc(const void *a, const void *b) {
    double da = ((const Result *)a)->tflops;
    double db = ((const Result *)b)->tflops;
    if (db > da) return 1;
    if (db < da) return -1;
    return 0;
}

int main(void) {
    printf("\n");
    printf("  ====== ANE SINGLE-KERNEL EXHAUSTIVE SWEEP ======\n");
    printf("\n");

    if (ane_init() != 0) {
        printf("  ERROR: Failed to initialize ANE.\n");
        return 1;
    }

    ANEDeviceInfo info = ane_device_info();
    if (!info.has_ane) {
        printf("  ERROR: No ANE detected.\n");
        return 1;
    }

    printf("  Chip:    %s, %d cores\n", info.arch ? info.arch : "unknown", info.num_cores);
    printf("  Thermal: %s\n", ane_thermal_state_str(ane_thermal_state()));
    printf("  Budget:  %d compiles (stopping at %d)\n", ANE_COMPILE_BUDGET, COMPILE_STOP);
    printf("  Iters:   %d warmup + %d timed per config\n", WARMUP_ITERS, TIMED_ITERS);
    printf("\n");

    // Pre-count valid configurations
    int valid_count = 0;
    for (int ci = 0; ci < N_CI; ci++)
        for (int co = 0; co < N_CO; co++)
            for (int si = 0; si < N_SP; si++) {
                size_t wt_bytes = (size_t)CH_IN[ci] * CH_OUT[co] * 2;
                size_t in_bytes = (size_t)CH_IN[ci] * SP[si] * 4;
                size_t out_bytes = (size_t)CH_OUT[co] * SP[si] * 4;
                if (wt_bytes <= MAX_WEIGHT_BYTES &&
                    in_bytes <= MAX_IO_BYTES &&
                    out_bytes <= MAX_IO_BYTES)
                    valid_count++;
            }

    printf("  Valid configurations: %d (grid: %dx%dx%d = %d total)\n",
           valid_count, N_CI, N_CO, N_SP, N_CI * N_CO * N_SP);

    if (valid_count > COMPILE_STOP) {
        printf("  NOTE: More configs than compile budget — will stop early.\n");
        printf("        Consider multiple runs or reducing grid.\n");
    }
    printf("\n");

    const char *wname = "@model_path/weights/weight.bin";
    Result *results = (Result *)calloc(MAX_RESULTS, sizeof(Result));
    int n_results = 0;

    printf("  %-6s %-6s %-6s %7s %7s %9s %7s %5s\n",
           "ch_in", "ch_out", "sp", "Wt(MB)", "GFLOP", "Latency", "TFLOPS", "Spill");
    printf("  %-6s %-6s %-6s %7s %7s %9s %7s %5s\n",
           "------", "------", "------", "-------", "-------", "---------", "-------", "-----");

    for (int ci_idx = 0; ci_idx < N_CI; ci_idx++) {
        for (int co_idx = 0; co_idx < N_CO; co_idx++) {
            for (int si_idx = 0; si_idx < N_SP; si_idx++) {
                int ci = CH_IN[ci_idx];
                int co = CH_OUT[co_idx];
                int sp = SP[si_idx];

                size_t wt_bytes = (size_t)ci * co * 2;
                size_t in_bytes = (size_t)ci * sp * 4;
                size_t out_bytes = (size_t)co * sp * 4;

                // Skip oversized configs
                if (wt_bytes > MAX_WEIGHT_BYTES) continue;
                if (in_bytes > MAX_IO_BYTES) continue;
                if (out_bytes > MAX_IO_BYTES) continue;

                // Check compile budget
                if (ane_compile_count() >= COMPILE_STOP) {
                    printf("\n  STOPPING: Compile budget reached (%d/%d)\n",
                           ane_compile_count(), ANE_COMPILE_BUDGET);
                    goto done;
                }

                double gflop = 2.0 * ci * co * sp / 1e9;
                double wt_mb = (double)wt_bytes / (1024.0 * 1024.0);

                // Allocate dummy weights
                float *dummy = (float *)calloc((size_t)ci * co, sizeof(float));
                if (!dummy) {
                    printf("  %-6d %-6d %-6d %6.1f MB %6.2f  OOM\n",
                           ci, co, sp, wt_mb, gflop);
                    continue;
                }
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

                if (!k) {
                    r.failed = 1;
                    printf("  %-6d %-6d %-6d %6.1f MB %6.2f  FAILED\n",
                           ci, co, sp, wt_mb, gflop);
                    fflush(stdout);
                    free(mil);
                    ane_weight_free(&w);
                    if (n_results < MAX_RESULTS) results[n_results++] = r;
                    continue;
                }

                // Check SRAM spill
                r.spill = ane_sram_spill(k) ? 1 : 0;

                // Write input data
                float *inp = (float *)calloc((size_t)ci * sp, sizeof(float));
                if (inp) {
                    for (size_t j = 0; j < (size_t)ci * sp; j++) inp[j] = 0.5f;
                    ane_write(k, 0, inp, in_bytes);
                    free(inp);
                }

                // Warmup
                for (int i = 0; i < WARMUP_ITERS; i++)
                    ane_eval(k, ANE_QOS_BACKGROUND);

                // Timed run (200 iters for sustained measurement)
                struct timespec t0, t1;
                clock_gettime(CLOCK_MONOTONIC, &t0);
                for (int i = 0; i < TIMED_ITERS; i++)
                    ane_eval(k, ANE_QOS_BACKGROUND);
                clock_gettime(CLOCK_MONOTONIC, &t1);

                double ms = ((t1.tv_sec - t0.tv_sec) * 1e3 +
                             (t1.tv_nsec - t0.tv_nsec) / 1e6) / TIMED_ITERS;

                r.latency_ms = ms;
                r.tflops = (ms > 0) ? gflop / ms : 0;

                printf("  %-6d %-6d %-6d %6.1f MB %6.2f  %6.3f ms  %6.2f %s\n",
                       ci, co, sp, wt_mb, gflop, ms, r.tflops,
                       r.spill ? " SPILL" : "");
                fflush(stdout);

                ane_free(k);
                free(mil);
                ane_weight_free(&w);

                if (n_results < MAX_RESULTS) results[n_results++] = r;
            }
        }
    }

done:
    printf("\n  Compiles used: %d / %d\n", ane_compile_count(), ANE_COMPILE_BUDGET);
    printf("  Thermal after sweep: %s\n", ane_thermal_state_str(ane_thermal_state()));

    // Sort by TFLOPS descending
    qsort(results, n_results, sizeof(Result), cmp_tflops_desc);

    // ===== TOP 10 =====
    printf("\n  ===== TOP 10 CONFIGURATIONS =====\n\n");
    printf("  %-4s %-6s %-6s %-6s %7s %8s %9s %7s %5s\n",
           "Rank", "ch_in", "ch_out", "sp", "Wt(MB)", "GFLOP", "Latency", "TFLOPS", "Spill");
    printf("  %-4s %-6s %-6s %-6s %7s %8s %9s %7s %5s\n",
           "----", "------", "------", "------", "-------", "--------", "---------", "-------", "-----");

    int shown = 0;
    for (int i = 0; i < n_results && shown < 10; i++) {
        Result *r = &results[i];
        if (r->failed) continue;
        shown++;
        printf("  #%-3d %-6d %-6d %-6d %6.1f MB %7.2f  %6.3f ms  %6.2f %s%s\n",
               shown, r->ch_in, r->ch_out, r->sp,
               r->wt_mb, r->gflop, r->latency_ms, r->tflops,
               r->spill ? " SPILL" : "",
               (shown == 1) ? "  <-- PEAK" : "");
    }

    // ===== Absolute peak =====
    if (n_results > 0 && !results[0].failed) {
        Result *best = &results[0];
        printf("\n  =============================================\n");
        printf("  MAXIMUM SUSTAINED SINGLE-KERNEL: %.2f TFLOPS\n", best->tflops);
        printf("  Config:  %d x %d  sp%d\n", best->ch_in, best->ch_out, best->sp);
        printf("  Latency: %.3f ms/eval\n", best->latency_ms);
        printf("  GFLOP:   %.2f per eval\n", best->gflop);
        printf("  Weights: %.1f MB\n", best->wt_mb);
        printf("  Spill:   %s\n", best->spill ? "YES" : "NO");
        printf("  =============================================\n\n");
    }

    // Spill summary
    int spill_count = 0, ok_count = 0;
    for (int i = 0; i < n_results; i++) {
        if (results[i].failed) continue;
        if (results[i].spill) spill_count++;
        else ok_count++;
    }
    printf("  Spill report: %d spilled, %d clean (of %d tested)\n\n",
           spill_count, ok_count, spill_count + ok_count);

    free(results);
    return 0;
}
