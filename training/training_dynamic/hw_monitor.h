// hw_monitor.h — Hardware utilization monitor for ANE training
// Header-only implementation. Samples CPU, GPU, memory, thermal metrics.
// Runs a background thread that writes structured CSV logs.
//
// Usage:
//   hw_monitor_start(1000, "hw_log.csv");   // sample every 1s
//   // ... training loop ...
//   hw_monitor_record_step(step, loss, ane_ms, cpu_ms, io_ms);  // per step
//   hw_monitor_stop();                       // prints summary
//
// Link: -framework Metal -framework IOKit
#pragma once

#import <Foundation/Foundation.h>
#import <mach/mach.h>
#import <mach/task_info.h>
#import <mach/mach_time.h>
#import <sys/sysctl.h>
#import <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include <unistd.h>

// Metal and IOKit are optional — only used if available
#ifdef __OBJC__
#import <Metal/Metal.h>
#import <IOKit/IOKitLib.h>
#endif

// ===== Snapshot struct =====

typedef struct {
    double timestamp_ms;       // wall clock since monitor start

    // CPU
    double cpu_user_ms;        // cumulative user time
    double cpu_sys_ms;         // cumulative system time
    int thread_count;          // threads in this process

    // GPU
    double gpu_util_pct;       // GPU utilization % (-1 if unavailable)
    double gpu_mem_mb;         // Metal allocated memory in MB (-1 if unavailable)

    // ANE
    double ane_util_pct;       // ANE utilization % (-1 = unavailable, always -1 on consumer macOS)

    // Memory
    double mem_rss_mb;         // resident set size in MB
    double mem_avail_mb;       // available memory in MB

    // Thermal
    int thermal_state;         // 0=nominal, 1=fair, 2=serious, 3=critical

    // Training metrics (filled by caller via hw_monitor_record_step)
    int step;
    float loss;
    double ane_ms;
    double cpu_ms;
    double io_ms;
} HWSnapshot;

// ===== Internal state =====

static struct {
    FILE *log_file;
    pthread_t thread;
    atomic_bool running;
    int interval_ms;
    uint64_t start_time;
    mach_timebase_info_data_t tb;

    // Accumulated snapshots for summary
    int n_snapshots;
    double sum_cpu_user, sum_cpu_sys;
    double sum_gpu_util, sum_gpu_mem;
    double sum_rss, sum_avail;
    int sum_thermal;
    int max_threads;
    double max_rss;
    double max_gpu_mem;
    int gpu_util_available;

    // Per-step training metrics (ring buffer of last value)
    atomic_int last_step;
    float last_loss;
    double last_ane_ms, last_cpu_ms, last_io_ms;

#ifdef __OBJC__
    id<MTLDevice> mtl_device;
#endif
} g_hwmon = {0};

// ===== Helper: elapsed ms =====

static double hwmon_elapsed_ms(void) {
    uint64_t now = mach_absolute_time();
    return (double)(now - g_hwmon.start_time) * g_hwmon.tb.numer / g_hwmon.tb.denom / 1e6;
}

// ===== CPU metrics via Mach task_info =====

static void hwmon_cpu_info(double *user_ms, double *sys_ms, int *threads) {
    // Cumulative CPU time for all threads in this process
    struct task_thread_times_info tti;
    mach_msg_type_number_t count = TASK_THREAD_TIMES_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_THREAD_TIMES_INFO,
                  (task_info_t)&tti, &count) == KERN_SUCCESS) {
        *user_ms = tti.user_time.seconds * 1000.0 + tti.user_time.microseconds / 1000.0;
        *sys_ms  = tti.system_time.seconds * 1000.0 + tti.system_time.microseconds / 1000.0;
    } else {
        *user_ms = *sys_ms = -1;
    }

    // Thread count via task_threads (most reliable method)
    thread_act_array_t thread_list;
    mach_msg_type_number_t thread_count;
    if (task_threads(mach_task_self(), &thread_list, &thread_count) == KERN_SUCCESS) {
        *threads = (int)thread_count;
        vm_deallocate(mach_task_self(), (vm_address_t)thread_list,
                      thread_count * sizeof(thread_act_t));
    } else {
        *threads = -1;
    }
}

// ===== Memory metrics =====

static void hwmon_mem_info(double *rss_mb, double *avail_mb) {
    struct mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  (task_info_t)&info, &count) == KERN_SUCCESS) {
        *rss_mb = (double)info.resident_size / (1024.0 * 1024.0);
    } else {
        *rss_mb = -1;
    }

    // System-wide memory pressure via host_statistics64
    vm_size_t page_size;
    host_page_size(mach_host_self(), &page_size);
    vm_statistics64_data_t vm_stat;
    mach_msg_type_number_t vm_count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                          (host_info64_t)&vm_stat, &vm_count) == KERN_SUCCESS) {
        // Free + purgeable is a reasonable "available" approximation
        uint64_t free_bytes = ((uint64_t)vm_stat.free_count + vm_stat.purgeable_count) * page_size;
        *avail_mb = (double)free_bytes / (1024.0 * 1024.0);
    } else {
        *avail_mb = -1;
    }
}

// ===== Thermal state =====

static int hwmon_thermal_state(void) {
#ifdef __OBJC__
    NSProcessInfoThermalState ts = [[NSProcessInfo processInfo] thermalState];
    switch (ts) {
        case NSProcessInfoThermalStateNominal:  return 0;
        case NSProcessInfoThermalStateFair:     return 1;
        case NSProcessInfoThermalStateSerious:  return 2;
        case NSProcessInfoThermalStateCritical: return 3;
        default: return 0;
    }
#else
    return -1;
#endif
}

static const char *hwmon_thermal_str(int state) {
    switch (state) {
        case 0: return "Nominal";
        case 1: return "Fair";
        case 2: return "Serious";
        case 3: return "Critical";
        default: return "Unknown";
    }
}

// ===== GPU utilization via IOKit =====

static double hwmon_gpu_util(void) {
#ifdef __OBJC__
    double util = -1.0;
    CFMutableDictionaryRef match = IOServiceMatching("IOAccelerator");
    io_iterator_t iter;
    if (IOServiceGetMatchingServices(kIOMainPortDefault, match, &iter) != kIOReturnSuccess)
        return -1.0;

    io_service_t service;
    while ((service = IOIteratorNext(iter)) != IO_OBJECT_NULL) {
        CFMutableDictionaryRef props = NULL;
        if (IORegistryEntryCreateCFProperties(service, &props, kCFAllocatorDefault, 0) == kIOReturnSuccess) {
            NSDictionary *dict = (__bridge NSDictionary *)props;
            NSDictionary *perfStats = dict[@"PerformanceStatistics"];
            if (perfStats) {
                // Try different keys for GPU utilization
                NSNumber *gpuUtil = perfStats[@"Device Utilization %"];
                if (!gpuUtil) gpuUtil = perfStats[@"GPU Activity(%)"];
                if (!gpuUtil) gpuUtil = perfStats[@"GPU Utilization %"];
                if (gpuUtil) {
                    util = [gpuUtil doubleValue];
                    g_hwmon.gpu_util_available = 1;
                }
            }
            CFRelease(props);
        }
        IOObjectRelease(service);
    }
    IOObjectRelease(iter);
    return util;
#else
    return -1.0;
#endif
}

// ===== GPU allocated memory via Metal =====

static double hwmon_gpu_mem_mb(void) {
#ifdef __OBJC__
    if (!g_hwmon.mtl_device) return -1.0;
    // currentAllocatedSize returns bytes allocated by this process on the GPU
    NSUInteger bytes = [g_hwmon.mtl_device currentAllocatedSize];
    return (double)bytes / (1024.0 * 1024.0);
#else
    return -1.0;
#endif
}

// ===== ANE utilization probe via IOKit =====

static double hwmon_ane_util(void) {
#ifdef __OBJC__
    // Probe IOKit for any ANE-related service with utilization stats.
    // On consumer macOS, this is not exposed — always returns -1.
    const char *service_names[] = {
        "AppleH13CamIn",    // ANE IO service on some chips
        "AppleNeuralEngine",
        "H11ANEIn",
        "ANEServices",
        NULL
    };
    for (int i = 0; service_names[i]; i++) {
        CFMutableDictionaryRef match = IOServiceMatching(service_names[i]);
        if (!match) continue;
        io_iterator_t iter;
        if (IOServiceGetMatchingServices(kIOMainPortDefault, match, &iter) != kIOReturnSuccess)
            continue;
        io_service_t svc;
        while ((svc = IOIteratorNext(iter)) != IO_OBJECT_NULL) {
            CFMutableDictionaryRef props = NULL;
            if (IORegistryEntryCreateCFProperties(svc, &props, kCFAllocatorDefault, 0) == kIOReturnSuccess) {
                NSDictionary *dict = (__bridge NSDictionary *)props;
                NSDictionary *perf = dict[@"PerformanceStatistics"];
                if (perf) {
                    NSNumber *u = perf[@"Device Utilization %"];
                    if (!u) u = perf[@"Neural Engine Utilization %"];
                    if (u) {
                        double val = [u doubleValue];
                        CFRelease(props);
                        IOObjectRelease(svc);
                        IOObjectRelease(iter);
                        return val;
                    }
                }
                CFRelease(props);
            }
            IOObjectRelease(svc);
        }
        IOObjectRelease(iter);
    }
#endif
    return -1.0;
}

// ===== Take a single snapshot =====

static HWSnapshot hw_snapshot(void) {
    HWSnapshot s = {0};
    s.timestamp_ms = hwmon_elapsed_ms();
    hwmon_cpu_info(&s.cpu_user_ms, &s.cpu_sys_ms, &s.thread_count);
    hwmon_mem_info(&s.mem_rss_mb, &s.mem_avail_mb);
    s.thermal_state = hwmon_thermal_state();
    s.gpu_util_pct = hwmon_gpu_util();
    s.gpu_mem_mb = hwmon_gpu_mem_mb();
    s.ane_util_pct = hwmon_ane_util();
    s.step = atomic_load(&g_hwmon.last_step);
    s.loss = g_hwmon.last_loss;
    s.ane_ms = g_hwmon.last_ane_ms;
    s.cpu_ms = g_hwmon.last_cpu_ms;
    s.io_ms = g_hwmon.last_io_ms;
    return s;
}

// ===== Write CSV header =====

static void hwmon_write_header(FILE *f) {
    fprintf(f, "timestamp_ms,step,loss,ane_ms,cpu_ms,io_ms,"
               "cpu_user_ms,cpu_sys_ms,threads,"
               "gpu_util,gpu_mem_mb,"
               "ane_util,"
               "mem_rss_mb,mem_avail_mb,thermal\n");
}

// ===== Write CSV row =====

static void hwmon_write_row(FILE *f, const HWSnapshot *s) {
    fprintf(f, "%.1f,%d,%.4f,%.1f,%.1f,%.1f,"
               "%.1f,%.1f,%d,"
               "%.1f,%.1f,"
               "%.1f,"
               "%.1f,%.1f,%d\n",
           s->timestamp_ms, s->step, s->loss,
           s->ane_ms, s->cpu_ms, s->io_ms,
           s->cpu_user_ms, s->cpu_sys_ms, s->thread_count,
           s->gpu_util_pct, s->gpu_mem_mb,
           s->ane_util_pct,
           s->mem_rss_mb, s->mem_avail_mb, s->thermal_state);
}

// ===== Update summary stats =====

static void hwmon_accumulate(const HWSnapshot *s) {
    g_hwmon.n_snapshots++;
    if (s->cpu_user_ms >= 0) g_hwmon.sum_cpu_user = s->cpu_user_ms;  // cumulative, take last
    if (s->cpu_sys_ms >= 0)  g_hwmon.sum_cpu_sys = s->cpu_sys_ms;
    if (s->gpu_util_pct >= 0) g_hwmon.sum_gpu_util += s->gpu_util_pct;
    if (s->gpu_mem_mb >= 0) {
        g_hwmon.sum_gpu_mem += s->gpu_mem_mb;
        if (s->gpu_mem_mb > g_hwmon.max_gpu_mem) g_hwmon.max_gpu_mem = s->gpu_mem_mb;
    }
    if (s->mem_rss_mb >= 0) {
        g_hwmon.sum_rss += s->mem_rss_mb;
        if (s->mem_rss_mb > g_hwmon.max_rss) g_hwmon.max_rss = s->mem_rss_mb;
    }
    if (s->mem_avail_mb >= 0) g_hwmon.sum_avail += s->mem_avail_mb;
    g_hwmon.sum_thermal += s->thermal_state;
    if (s->thread_count > g_hwmon.max_threads) g_hwmon.max_threads = s->thread_count;
}

// ===== Background sampling thread =====

static void *hwmon_thread_fn(void *arg) {
    (void)arg;
    while (atomic_load(&g_hwmon.running)) {
        @autoreleasepool {
            HWSnapshot s = hw_snapshot();
            if (g_hwmon.log_file) {
                hwmon_write_row(g_hwmon.log_file, &s);
                fflush(g_hwmon.log_file);
            }
            hwmon_accumulate(&s);
        }
        usleep(g_hwmon.interval_ms * 1000);
    }
    return NULL;
}

// ===== Public API =====

// Start background monitoring. Samples every interval_ms, writes CSV to log_path.
static void hw_monitor_start(int interval_ms, const char *log_path) {
    memset((void *)&g_hwmon, 0, sizeof(g_hwmon));
    mach_timebase_info(&g_hwmon.tb);
    g_hwmon.start_time = mach_absolute_time();
    g_hwmon.interval_ms = interval_ms;
    atomic_store(&g_hwmon.last_step, -1);
    g_hwmon.last_loss = 0;

#ifdef __OBJC__
    // Initialize Metal device for GPU memory tracking
    g_hwmon.mtl_device = MTLCreateSystemDefaultDevice();
#endif

    // Open CSV log
    g_hwmon.log_file = fopen(log_path, "w");
    if (g_hwmon.log_file) {
        hwmon_write_header(g_hwmon.log_file);
    } else {
        fprintf(stderr, "[hw_monitor] WARNING: cannot open %s for writing\n", log_path);
    }

    // Print detected capabilities
    printf("[hw_monitor] Starting hardware monitor (interval=%dms, log=%s)\n", interval_ms, log_path);

    // Probe what is actually available
    HWSnapshot probe = hw_snapshot();
    printf("[hw_monitor] Detected metrics:\n");
    printf("  CPU times:     %s\n", probe.cpu_user_ms >= 0 ? "YES (task_info)" : "NO");
    printf("  Thread count:  %s\n", probe.thread_count >= 0 ? "YES (task_threads)" : "NO");
    printf("  Memory RSS:    %s\n", probe.mem_rss_mb >= 0 ? "YES (mach_task_basic_info)" : "NO");
    printf("  Memory avail:  %s\n", probe.mem_avail_mb >= 0 ? "YES (host_statistics64)" : "NO");
    printf("  Thermal:       %s\n", probe.thermal_state >= 0 ? "YES (NSProcessInfo)" : "NO");
    printf("  GPU util:      %s\n", probe.gpu_util_pct >= 0 ? "YES (IOAccelerator)" : "NO (IOAccelerator not exposed)");
    printf("  GPU memory:    %s\n", probe.gpu_mem_mb >= 0 ? "YES (Metal)" : "NO");
    printf("  ANE util:      %s\n", probe.ane_util_pct >= 0 ? "YES (IOKit)" : "NO (not exposed on consumer macOS)");

    // Start background thread
    atomic_store(&g_hwmon.running, true);
    pthread_create(&g_hwmon.thread, NULL, hwmon_thread_fn, NULL);
}

// Record training step metrics (called from training loop).
// Thread-safe — updates atomics that the background sampler reads.
static void hw_monitor_record_step(int step, float loss, double ane_ms, double cpu_ms, double io_ms) {
    atomic_store(&g_hwmon.last_step, step);
    g_hwmon.last_loss = loss;
    g_hwmon.last_ane_ms = ane_ms;
    g_hwmon.last_cpu_ms = cpu_ms;
    g_hwmon.last_io_ms = io_ms;
}

// Stop background monitoring, close log, print summary.
static void hw_monitor_stop(void) {
    atomic_store(&g_hwmon.running, false);
    pthread_join(g_hwmon.thread, NULL);

    // Take one final snapshot
    HWSnapshot final_snap = hw_snapshot();
    if (g_hwmon.log_file) {
        hwmon_write_row(g_hwmon.log_file, &final_snap);
        fclose(g_hwmon.log_file);
        g_hwmon.log_file = NULL;
    }
    hwmon_accumulate(&final_snap);

    int n = g_hwmon.n_snapshots;
    if (n == 0) n = 1;

    printf("\n[hw_monitor] === Hardware Utilization Summary (%d samples) ===\n", g_hwmon.n_snapshots);
    printf("  CPU time:       user=%.0fms sys=%.0fms (cumulative)\n",
           g_hwmon.sum_cpu_user, g_hwmon.sum_cpu_sys);
    printf("  Peak threads:   %d\n", g_hwmon.max_threads);
    printf("  Memory RSS:     avg=%.1fMB  peak=%.1fMB\n",
           g_hwmon.sum_rss / n, g_hwmon.max_rss);
    printf("  Memory avail:   avg=%.0fMB\n", g_hwmon.sum_avail / n);
    printf("  Thermal:        avg=%.2f (%s)\n",
           (double)g_hwmon.sum_thermal / n,
           hwmon_thermal_str(g_hwmon.sum_thermal / n));
    if (g_hwmon.gpu_util_available) {
        printf("  GPU util:       avg=%.1f%%\n", g_hwmon.sum_gpu_util / n);
    } else {
        printf("  GPU util:       not available (IOAccelerator stats not exposed)\n");
    }
    if (g_hwmon.max_gpu_mem > 0) {
        printf("  GPU memory:     avg=%.1fMB  peak=%.1fMB\n",
               g_hwmon.sum_gpu_mem / n, g_hwmon.max_gpu_mem);
    } else {
        printf("  GPU memory:     no Metal allocations\n");
    }
    printf("  ANE util:       not available (no IOKit exposure on consumer macOS)\n");
    printf("[hw_monitor] ==========================================\n");
}
