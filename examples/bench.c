// bench.c — ANE Auto-Benchmark
// Detects chip, measures TFLOPS across configs, shows ASCII barchart
//
// Build & run: make bench

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "../libane/ane.h"

// ===== Reference data for known chips =====
// tflops: measured peak (fp16 matmul via libane bench)
// apple_tops: Apple's marketing spec (INT8 TOPS, Neural Engine page)
typedef struct { const char *arch; const char *name; double tflops; double apple_tops; } ChipRef;
static const ChipRef CHIPS[] = {
    {"h13g", "M1",      5.5,  11.0},
    {"h13p", "M1 Pro",  5.5,  11.0},
    {"h14g", "M2 Pro",  9.0,  15.8},
    {"h14p", "M2 Max",  9.2,  15.8},
    {"h15g", "M3 Pro",  9.4,  18.0},
    {"h15p", "M3 Max",  9.5,  18.0},
    {"h16g", "M4",     11.0,  38.0},
    {"h16p", "M4 Pro", 12.0,  38.0},
    {NULL, NULL, 0, 0}
};

static const char *chip_name_for(const char *arch) {
    for (int i = 0; CHIPS[i].arch; i++)
        if (arch && strcmp(arch, CHIPS[i].arch) == 0) return CHIPS[i].name;
    return NULL;
}

// ===== ASCII barchart =====
static void bar(const char *label, double val, double max_val, int width) {
    int fill = (max_val > 0) ? (int)(val / max_val * width + 0.5) : 0;
    if (fill > width) fill = width;
    if (fill < 1 && val > 0) fill = 1;
    printf("  %-22s %6.2f TFLOPS  ", label, val);
    for (int i = 0; i < fill; i++) printf("\xe2\x96\x88");
    for (int i = fill; i < width; i++) printf("\xe2\x96\x91");
    printf("\n");
}

// ===== Stacked benchmark run =====
static double bench_stacked(int ch, int sp, int depth, const char *wname) {
    ANEWeight w = ane_weight_stacked(wname, ch, depth);
    char *mil = ane_mil_stacked_conv(ch, sp, depth, wname);
    size_t io_bytes = (size_t)ch * sp * 4;

    ANEKernel *k = ane_compile(mil, strlen(mil), &w, 1,
                                1, &io_bytes, 1, &io_bytes,
                                ANE_QOS_BACKGROUND);
    if (!k) { free(mil); ane_weight_free(&w); return -1; }

    float *inp = (float *)calloc(ch * sp, sizeof(float));
    for (int i = 0; i < ch * sp; i++) inp[i] = 0.5f;
    ane_write(k, 0, inp, io_bytes);
    free(inp);

    for (int i = 0; i < 5; i++) ane_eval(k, ANE_QOS_BACKGROUND);

    int iters = 20;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < iters; i++) ane_eval(k, ANE_QOS_BACKGROUND);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double ms = ((t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6) / iters;

    ane_free(k);
    free(mil);
    ane_weight_free(&w);
    return ms;
}

// ===== QoS benchmark =====
static double bench_qos(int ch, int sp, ANEQoS qos, const char *wname) {
    float *dummy = (float *)calloc((size_t)ch * ch, sizeof(float));
    ANEWeight w = ane_weight_fp16(wname, dummy, ch, ch);
    char *mil = ane_mil_linear(ch, ch, sp, wname);
    size_t io_bytes = (size_t)ch * sp * 4;

    ANEKernel *k = ane_compile(mil, strlen(mil), &w, 1,
                                1, &io_bytes, 1, &io_bytes, qos);
    free(dummy);
    if (!k) { free(mil); ane_weight_free(&w); return -1; }

    float *inp = (float *)calloc(ch * sp, sizeof(float));
    for (int i = 0; i < ch * sp; i++) inp[i] = 0.5f;
    ane_write(k, 0, inp, io_bytes);
    free(inp);

    for (int i = 0; i < 5; i++) ane_eval(k, qos);

    int iters = 30;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < iters; i++) ane_eval(k, qos);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double ms = ((t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6) / iters;

    ane_free(k);
    free(mil);
    ane_weight_free(&w);
    return ms;
}

// ===== Quick peak measurement (~2-3s) =====
// Compiles 2 kernels, runs 200 evals each, prints one summary line.
// Output format: "TFLOPS:3.64|TOPS:18|CHIP:M3 Pro|ARCH:h15g|CORES:16|API:v1(35)"
// Designed for machine parsing by the CLI.
static int quick_peak(void) {
    if (ane_init() != 0) { printf("ERROR\n"); return 1; }
    ANEDeviceInfo info = ane_device_info();
    if (!info.has_ane) { printf("ERROR\n"); return 1; }

    const char *name = chip_name_for(info.arch);
    double apple_tops = 0;
    for (int i = 0; CHIPS[i].arch; i++)
        if (info.arch && strcmp(info.arch, CHIPS[i].arch) == 0) apple_tops = CHIPS[i].apple_tops;

    const char *wname = "@model_path/weights/weight.bin";
    double peak = 0;

    // Test 2 kernel shapes known to find the peak
    struct { int ci; int co; int sp; } tests[] = {
        {768, 2048, 256},
        {2048, 2048, 128},
    };
    for (int t = 0; t < 2; t++) {
        int ci = tests[t].ci, co = tests[t].co, sp = tests[t].sp;
        double gflop = 2.0 * ci * co * sp / 1e9;
        float *dummy = (float *)calloc((size_t)ci * co, sizeof(float));
        for (int j = 0; j < ci * co; j++) dummy[j] = 0.01f;
        ANEWeight w = ane_weight_fp16(wname, dummy, co, ci);
        char *mil = ane_mil_linear(ci, co, sp, wname);
        size_t in_bytes = (size_t)ci * sp * 4;
        size_t out_bytes = (size_t)co * sp * 4;
        ANEKernel *k = ane_compile(mil, strlen(mil), &w, 1,
                                    1, &in_bytes, 1, &out_bytes,
                                    ANE_QOS_BACKGROUND);
        free(dummy);
        if (k) {
            float *inp = (float *)calloc(ci * sp, sizeof(float));
            for (int j = 0; j < ci * sp; j++) inp[j] = 0.5f;
            ane_write(k, 0, inp, in_bytes);
            free(inp);
            for (int j = 0; j < 5; j++) ane_eval(k, ANE_QOS_BACKGROUND);
            struct timespec t0, t1;
            clock_gettime(CLOCK_MONOTONIC, &t0);
            for (int j = 0; j < 200; j++) ane_eval(k, ANE_QOS_BACKGROUND);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            double ms = ((t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6) / 200;
            double tflops = gflop / ms;
            if (tflops > peak) peak = tflops;
            ane_free(k);
        }
        free(mil); ane_weight_free(&w);
    }

    printf("TFLOPS:%.2f|TOPS:%.0f|CHIP:%s|ARCH:%s|CORES:%d|API:v%d(%d)\n",
        peak, apple_tops,
        name ? name : "unknown",
        info.arch ? info.arch : "?",
        info.num_cores,
        ane_api_info().api_version, ane_api_info().classes_found);
    return 0;
}

int main(int argc, char *argv[]) {
    // --quick: fast peak measurement for CLI integration
    if (argc > 1 && strcmp(argv[1], "--quick") == 0)
        return quick_peak();

    // --save-profile=PATH: write profile after bench completes
    const char *profile_path = NULL;
    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "--save-profile=", 15) == 0)
            profile_path = argv[i] + 15;
    }

    printf("\n");
    printf("  \xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88 ANE BENCHMARK \xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\xe2\x96\x88\n");
    printf("\n");

    if (ane_init() != 0) {
        printf("  ERROR: Failed to initialize ANE.\n");
        return 1;
    }

    ANEDeviceInfo info = ane_device_info();
    if (!info.has_ane) { printf("  ERROR: No ANE detected.\n"); return 1; }

    const char *name = chip_name_for(info.arch);
    printf("  Chip:   %s%s%s%s, %d cores\n",
        info.arch ? info.arch : "unknown",
        name ? " (" : "", name ? name : "",
        name ? ")" : "", info.num_cores);
    printf("  Build:  %s\n", info.build ? info.build : "?");
    printf("  API:    v%d (%d classes)\n", ane_api_info().api_version, ane_api_info().classes_found);

    // Apple marketing spec for this chip
    double apple_tops = 0;
    for (int i = 0; CHIPS[i].arch; i++)
        if (info.arch && strcmp(info.arch, CHIPS[i].arch) == 0) apple_tops = CHIPS[i].apple_tops;
    if (apple_tops > 0)
        printf("  Apple:  %.0f TOPS (marketing, INT8)\n", apple_tops);

    const char *wname = "@model_path/weights/weight.bin";

    // ===== Phase 1: Single-Conv Sweep =====
    printf("\n  ---- Single Conv Sweep ----\n\n");
    printf("  %-22s %7s %7s %9s %7s\n", "Config", "Weights", "GFLOP", "Latency", "TFLOPS");
    printf("  %-22s %7s %7s %9s %7s\n", "----------------------", "-------", "-------", "---------", "-------");

    struct { int ch_in; int ch_out; int sp; } singles[] = {
        // Square, small spatial (original)
        { 256,  256,  64},
        { 512,  512,  64},
        {1024, 1024,  64},
        {2048, 2048,  64},
        // Larger spatial (where peak lives)
        { 768,  768, 256},
        {2048,  768, 256},
        { 768, 2048, 256},
        {1024, 1024, 256},
        {2048, 2048, 128},
    };
    int n_singles = sizeof(singles) / sizeof(singles[0]);
    double peak_single = 0;
    double results_single[9];
    int best_single_idx = 0;

    for (int i = 0; i < n_singles; i++) {
        int ci = singles[i].ch_in, co = singles[i].ch_out, sp = singles[i].sp;
        double gflop = 2.0 * ci * co * sp / 1e9;
        double wt_mb = (double)ci * co * 2 / (1024 * 1024);

        // Use bench_rect for rectangular matmuls
        float *dummy = (float *)calloc((size_t)ci * co, sizeof(float));
        for (int j = 0; j < ci * co; j++) dummy[j] = 0.01f;
        ANEWeight w = ane_weight_fp16(wname, dummy, co, ci);
        char *mil = ane_mil_linear(ci, co, sp, wname);
        size_t in_bytes = (size_t)ci * sp * 4;
        size_t out_bytes = (size_t)co * sp * 4;
        ANEKernel *k = ane_compile(mil, strlen(mil), &w, 1,
                                    1, &in_bytes, 1, &out_bytes,
                                    ANE_QOS_BACKGROUND);
        free(dummy);
        double ms = -1;
        if (k) {
            float *inp = (float *)calloc(ci * sp, sizeof(float));
            for (int j = 0; j < ci * sp; j++) inp[j] = 0.5f;
            ane_write(k, 0, inp, in_bytes);
            free(inp);
            for (int j = 0; j < 5; j++) ane_eval(k, ANE_QOS_BACKGROUND);
            int iters = (ci >= 1024 && sp >= 128) ? 100 : 200;
            struct timespec t0, t1;
            clock_gettime(CLOCK_MONOTONIC, &t0);
            for (int j = 0; j < iters; j++) ane_eval(k, ANE_QOS_BACKGROUND);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            ms = ((t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6) / iters;
            ane_free(k);
        }
        free(mil); ane_weight_free(&w);

        double tflops = (ms > 0) ? gflop / ms : 0;
        results_single[i] = tflops;
        if (tflops > peak_single) { peak_single = tflops; best_single_idx = i; }

        char label[32];
        if (ci == co) snprintf(label, sizeof(label), "%dx%d sp%d", ci, co, sp);
        else          snprintf(label, sizeof(label), "%dx%d sp%d", ci, co, sp);
        if (ms > 0)
            printf("  %-22s %5.1f MB %6.2f  %6.3f ms  %6.2f\n", label, wt_mb, gflop, ms, tflops);
        else
            printf("  %-22s %5.1f MB %6.2f  FAILED\n", label, wt_mb, gflop);
        fflush(stdout);
    }

    // ===== Phase 2: Stacked (Peak Sustained) =====
    printf("\n  ---- Stacked Conv (amortize dispatch) ----\n\n");
    printf("  %-24s %7s %7s %9s %7s\n", "Config", "Weights", "GFLOP", "Latency", "TFLOPS");
    printf("  %-24s %7s %7s %9s %7s\n", "------------------------", "-------", "-------", "---------", "-------");

    struct { int ch; int sp; int depth; } stacks[] = {
        {512, 64, 32}, {512, 64, 64}, {512, 64, 128}
    };
    double peak_stacked = 0;
    double results_stacked[3];

    for (int i = 0; i < 3; i++) {
        int ch = stacks[i].ch, sp = stacks[i].sp, d = stacks[i].depth;
        double gflop = 2.0 * ch * ch * sp * d / 1e9;
        double wt_mb = (double)ch * ch * 2 * d / (1024 * 1024);
        double ms = bench_stacked(ch, sp, d, wname);
        double tflops = (ms > 0) ? gflop / ms : 0;
        results_stacked[i] = tflops;
        if (tflops > peak_stacked) peak_stacked = tflops;

        char label[48];
        snprintf(label, sizeof(label), "%dx conv %dch sp%d", d, ch, sp);
        if (ms > 0)
            printf("  %-24s %5.1f MB %6.2f  %6.3f ms  %6.2f\n", label, wt_mb, gflop, ms, tflops);
        else
            printf("  %-24s %5.1f MB %6.2f  FAILED\n", label, wt_mb, gflop);
        fflush(stdout);
    }

    // ===== Phase 3: 5-Second Sustained Peak =====
    printf("\n  ---- Sustained Peak (5 seconds) ----\n\n");
    double sustained_tflops = 0;
    double dispatch_overhead_ms = 0.25; // default, refined below
    {
        // Use the best single kernel shape for sustained test
        int ci = singles[best_single_idx].ch_in;
        int co = singles[best_single_idx].ch_out;
        int sp = singles[best_single_idx].sp;
        double gflop = 2.0 * ci * co * sp / 1e9;

        float *dummy = (float *)calloc((size_t)ci * co, sizeof(float));
        for (int j = 0; j < ci * co; j++) dummy[j] = 0.01f;
        ANEWeight w = ane_weight_fp16(wname, dummy, co, ci);
        char *mil = ane_mil_linear(ci, co, sp, wname);
        size_t in_bytes = (size_t)ci * sp * 4;
        size_t out_bytes = (size_t)co * sp * 4;
        ANEKernel *k = ane_compile(mil, strlen(mil), &w, 1,
                                    1, &in_bytes, 1, &out_bytes,
                                    ANE_QOS_BACKGROUND);
        free(dummy);

        if (k) {
            float *inp = (float *)calloc(ci * sp, sizeof(float));
            for (int j = 0; j < ci * sp; j++) inp[j] = 0.5f;
            ane_write(k, 0, inp, in_bytes);
            free(inp);

            // Warmup
            for (int j = 0; j < 10; j++) ane_eval(k, ANE_QOS_BACKGROUND);

            // 5 seconds sustained
            int count = 0;
            struct timespec t0, t1;
            clock_gettime(CLOCK_MONOTONIC, &t0);
            for (;;) {
                ane_eval(k, ANE_QOS_BACKGROUND);
                count++;
                if (count % 100 == 0) {
                    clock_gettime(CLOCK_MONOTONIC, &t1);
                    double elapsed = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
                    if (elapsed >= 5000.0) break;
                }
            }
            clock_gettime(CLOCK_MONOTONIC, &t1);
            double ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
            double ms_per = ms / count;
            sustained_tflops = gflop / ms_per;

            char label[32];
            snprintf(label, sizeof(label), "%dx%d sp%d", ci, co, sp);
            printf("  Kernel:    %s (best from sweep)\n", label);
            printf("  Evals:     %d in %.1fs\n", count, ms / 1000);
            printf("  Latency:   %.3f ms/eval\n", ms_per);
            printf("  Sustained: %.2f TFLOPS (fp16)\n", sustained_tflops);
            ane_free(k);
        } else {
            printf("  FAILED to compile sustained test kernel\n");
        }
        free(mil); ane_weight_free(&w);

        if (sustained_tflops > peak_single) peak_single = sustained_tflops;

        // Estimate dispatch overhead from smallest kernel (256x256 sp64, index 0)
        // dispatch = measured_latency - (gflop / peak_tflops)
        if (results_single[0] > 0 && peak_single > 0) {
            double tiny_gflop = 2.0 * 256 * 256 * 64 / 1e9;
            double tiny_latency = tiny_gflop / results_single[0]; // measured total ms
            double compute_only = tiny_gflop / peak_single;       // pure compute ms
            dispatch_overhead_ms = tiny_latency - compute_only;
            if (dispatch_overhead_ms < 0.05) dispatch_overhead_ms = 0.05;
        }
    }

    // ===== Phase 4: QoS Sweep =====
    printf("\n  ---- QoS Levels (512x512 sp64) ----\n\n");
    printf("  %-20s %6s %9s %7s\n", "QoS Level", "Value", "Latency", "TFLOPS");
    printf("  %-20s %6s %9s %7s\n", "--------------------", "------", "---------", "-------");

    struct { const char *name; ANEQoS qos; } qos_levels[] = {
        {"Background",       ANE_QOS_BACKGROUND},
        {"Utility",          ANE_QOS_UTILITY},
        {"Default",          ANE_QOS_DEFAULT},
        {"User Initiated",   ANE_QOS_USER_INITIATED},
        {"User Interactive", ANE_QOS_USER_INTERACTIVE},
    };
    double qos_tflops[5];
    double gflop_qos = 2.0 * 512 * 512 * 64 / 1e9;

    for (int i = 0; i < 5; i++) {
        double ms = bench_qos(512, 64, qos_levels[i].qos, wname);
        double tflops = (ms > 0) ? gflop_qos / ms : 0;
        qos_tflops[i] = tflops;

        if (ms > 0)
            printf("  %-20s %6d  %6.3f ms  %6.2f\n", qos_levels[i].name, (int)qos_levels[i].qos, ms, tflops);
        else
            printf("  %-20s %6d  FAILED\n", qos_levels[i].name, (int)qos_levels[i].qos);
    }

    // ===== Phase 5: ASCII Barchart =====
    double overall_peak = peak_single > peak_stacked ? peak_single : peak_stacked;

    printf("\n  ---- Performance Overview ----\n\n");

    for (int i = 0; i < n_singles; i++) {
        char label[32];
        int ci = singles[i].ch_in, co = singles[i].ch_out, sp = singles[i].sp;
        if (ci == co) snprintf(label, sizeof(label), "%dx%d sp%d", ci, co, sp);
        else          snprintf(label, sizeof(label), "%dx%d sp%d", ci, co, sp);
        bar(label, results_single[i], overall_peak * 1.2, 30);
    }

    printf("\n");
    for (int i = 0; i < 3; i++) {
        char label[32];
        snprintf(label, sizeof(label), "%dx stacked", stacks[i].depth);
        bar(label, results_stacked[i], overall_peak * 1.2, 30);
    }

    // ===== Phase 6: Reference Comparison =====
    printf("\n  ---- Chip Comparison (measured fp16 TFLOPS) ----\n\n");
    double max_ref = overall_peak;
    for (int i = 0; CHIPS[i].arch; i++)
        if (CHIPS[i].tflops > max_ref) max_ref = CHIPS[i].tflops;

    char your_label[48];
    snprintf(your_label, sizeof(your_label), ">> %s (%s)",
        info.arch ? info.arch : "?", name ? name : "This chip");
    bar(your_label, overall_peak, max_ref * 1.2, 30);

    for (int i = 0; CHIPS[i].arch; i++) {
        if (info.arch && strcmp(info.arch, CHIPS[i].arch) == 0) continue;
        char ref_label[48];
        snprintf(ref_label, sizeof(ref_label), "   %s (%s)", CHIPS[i].arch, CHIPS[i].name);
        bar(ref_label, CHIPS[i].tflops, max_ref * 1.2, 30);
    }

    // ===== Summary =====
    printf("\n  ---- Summary ----\n\n");
    printf("  Measured peak:       %.2f TFLOPS (fp16 matmul)\n", overall_peak);
    if (apple_tops > 0) {
        printf("  Apple marketing:     %.0f TOPS (INT8, theoretical)\n", apple_tops);
        printf("  Efficiency:          %.1f%% of Apple spec\n", 100.0 * overall_peak / apple_tops);
        printf("\n");
        printf("  Note: Apple's TOPS are INT8 peak throughput. Real fp16 matmul\n");
        printf("  throughput is lower due to dispatch overhead, memory bandwidth,\n");
        printf("  and the fp16 data path being narrower than INT8.\n");
    }
    printf("\n");
    printf("  Best QoS:            Background (9)\n");
    printf("  Dispatch overhead:   ~%.2f ms/kernel (minimum latency floor)\n", dispatch_overhead_ms);
    printf("  Compiles used:       %d / %d\n", ane_compile_count(), ANE_COMPILE_BUDGET);
    printf("\n");

    // ===== Save profile if requested =====
    if (profile_path) {
        // Find best QoS
        int best_qos_idx = 0;
        for (int i = 1; i < 5; i++)
            if (qos_tflops[i] > qos_tflops[best_qos_idx]) best_qos_idx = i;
        const char *qos_names[] = {"background", "utility", "default", "user_initiated", "user_interactive"};

        // Compute recommended accum_steps based on peak TFLOPS
        int rec_accum = 100;
        if (overall_peak < 4.0) rec_accum = 50;
        else if (overall_peak < 8.0) rec_accum = 100;
        else rec_accum = 200;

        // Best kernel shape
        int bci = singles[best_single_idx].ch_in;
        int bco = singles[best_single_idx].ch_out;
        int bsp = singles[best_single_idx].sp;

        FILE *fp = fopen(profile_path, "w");
        if (fp) {
            time_t now = time(NULL);
            struct tm *t = localtime(&now);
            char ts_str[64];
            strftime(ts_str, sizeof(ts_str), "%Y-%m-%dT%H:%M:%S%z", t);

            fprintf(fp, "# ANE Hardware Profile\n");
            fprintf(fp, "# Generated by: ./ane bench\n");
            fprintf(fp, "# Date: %s\n", ts_str);
            fprintf(fp, "#\n");
            fprintf(fp, "# This file is machine-specific and auto-generated.\n");
            fprintf(fp, "# Re-run ./ane bench to refresh.\n");
            fprintf(fp, "\n");
            fprintf(fp, "# Hardware\n");
            fprintf(fp, "chip='%s'\n", name ? name : "unknown");
            fprintf(fp, "arch='%s'\n", info.arch ? info.arch : "unknown");
            fprintf(fp, "cores=%d\n", info.num_cores);
            fprintf(fp, "build='%s'\n", info.build ? info.build : "unknown");
            fprintf(fp, "api_version=%d\n", ane_api_info().api_version);
            fprintf(fp, "api_classes=%d\n", ane_api_info().classes_found);
            fprintf(fp, "\n");
            fprintf(fp, "# Performance (measured)\n");
            fprintf(fp, "peak_tflops=%.2f\n", overall_peak);
            fprintf(fp, "sustained_tflops=%.2f\n", sustained_tflops);
            fprintf(fp, "dispatch_overhead_ms=%.2f\n", dispatch_overhead_ms);
            fprintf(fp, "best_kernel='%dx%dx%d'\n", bci, bco, bsp);
            fprintf(fp, "\n");
            fprintf(fp, "# Apple reference\n");
            fprintf(fp, "apple_tops=%.0f\n", apple_tops);
            fprintf(fp, "efficiency_pct=%.1f\n", apple_tops > 0 ? 100.0 * overall_peak / apple_tops : 0);
            fprintf(fp, "\n");
            fprintf(fp, "# QoS (measured TFLOPS per level)\n");
            fprintf(fp, "optimal_qos='%s'\n", qos_names[best_qos_idx]);
            fprintf(fp, "qos_background=%.2f\n", qos_tflops[0]);
            fprintf(fp, "qos_utility=%.2f\n", qos_tflops[1]);
            fprintf(fp, "qos_default=%.2f\n", qos_tflops[2]);
            fprintf(fp, "qos_user_initiated=%.2f\n", qos_tflops[3]);
            fprintf(fp, "qos_user_interactive=%.2f\n", qos_tflops[4]);
            fprintf(fp, "\n");
            fprintf(fp, "# Recommended training parameters\n");
            fprintf(fp, "recommended_accum_steps=%d\n", rec_accum);
            fprintf(fp, "recommended_max_compiles=1000\n");
            fprintf(fp, "compile_budget=%d\n", ANE_COMPILE_BUDGET);
            fprintf(fp, "\n");
            fprintf(fp, "# Timestamp (unix epoch, for cache expiry)\n");
            fprintf(fp, "timestamp=%ld\n", (long)now);

            fclose(fp);
            printf("  Profile saved: %s\n\n", profile_path);
        }
    }

    return 0;
}
