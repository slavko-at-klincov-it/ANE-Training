#!/usr/bin/env python3
"""Parse ANE and GPU benchmark results and generate comparison table.
Supports multiple models (tiny_ane, stories110m)."""

import json
import os
import re
import sys
from datetime import datetime


def parse_ane_log(path):
    """Parse ANE training log for step times, loss, hardware breakdown, and config."""
    steps = []
    accum_steps = 10
    flops_total = 7851.7e6
    model_name = "Unknown"

    with open(path) as f:
        for line in f:
            m_name = re.search(r'Dynamic Training:\s+(.+?)\s+\(', line)
            if m_name:
                model_name = m_name.group(1)

            m_acc = re.search(r'Accum\s+(\d+)\s+steps', line)
            if m_acc:
                accum_steps = int(m_acc.group(1))

            m_flops = re.search(r'total=([\d.]+)M', line)
            if m_flops:
                flops_total = float(m_flops.group(1)) * 1e6

            m = re.search(r'step\s+(\d+)\s+loss=([\d.]+)\s+lr=([\d.e+-]+)\s+([\d.]+)ms/step', line)
            if m:
                steps.append({
                    'step': int(m.group(1)),
                    'loss': float(m.group(2)),
                    'lr': float(m.group(3)),
                    'step_ms': float(m.group(4)),
                })

            m_hw = re.search(r'hardware: ANE=([\d.]+)ms\s+CPU=([\d.]+)ms\s+IO=([\d.]+)ms', line)
            if m_hw and steps:
                steps[-1]['ane_ms'] = float(m_hw.group(1))
                steps[-1]['cpu_ms'] = float(m_hw.group(2))
                steps[-1]['io_ms'] = float(m_hw.group(3))

    return steps, accum_steps, flops_total, model_name


def compare_model(results_dir, model_key):
    """Compare ANE vs GPU for a single model. Returns summary dict."""
    # Load GPU results
    gpu_json = os.path.join(results_dir, f'{model_key}_gpu_results.json')
    gpu_data = None
    if os.path.exists(gpu_json):
        with open(gpu_json) as f:
            gpu_data = json.load(f)

    # Parse ANE log
    ane_log = os.path.join(results_dir, f'{model_key}_ane_log.txt')
    ane_steps = None
    ane_accum = 10
    ane_flops = 7851.7e6
    model_name = model_key
    if os.path.exists(ane_log):
        ane_steps, ane_accum, ane_flops, model_name = parse_ane_log(ane_log)

    if not gpu_data and not ane_steps:
        return None

    # ANE metrics
    ane_avg_ms = ane_micro_ms = ane_ane_only_ms = ane_cpu_ms = ane_io_ms = None
    ane_tflops = ane_loss = None
    if ane_steps and len(ane_steps) > 2:
        skip = ane_steps[2:]
        ane_micro_ms = sum(s['step_ms'] for s in skip) / len(skip)
        ane_avg_ms = ane_micro_ms * ane_accum
        ane_loss = ane_steps[-1]['loss']
        ane_tflops = (ane_flops * ane_accum) / (ane_avg_ms / 1000.0) / 1e12
        hw = [s for s in skip if 'ane_ms' in s]
        if hw:
            ane_ane_only_ms = sum(s['ane_ms'] for s in hw) / len(hw)
            ane_cpu_ms = sum(s['cpu_ms'] for s in hw) / len(hw)
            ane_io_ms = sum(s['io_ms'] for s in hw) / len(hw)

    # GPU metrics
    gpu_avg_ms = gpu_tflops = gpu_loss = None
    if gpu_data:
        per_step = gpu_data.get('per_step', [])
        if len(per_step) > 2:
            gpu_avg_ms = sum(s['step_ms'] for s in per_step[2:]) / len(per_step[2:])
            gpu_tflops = sum(s['tflops'] for s in per_step[2:]) / len(per_step[2:])
        else:
            gpu_avg_ms = gpu_data.get('avg_step_ms')
            gpu_tflops = gpu_data.get('avg_tflops')
        gpu_loss = gpu_data.get('final_loss')

    return {
        'model': model_name,
        'ane_adam_ms': ane_avg_ms,
        'ane_micro_ms': ane_micro_ms,
        'ane_hw_ms': ane_ane_only_ms,
        'ane_cpu_ms': ane_cpu_ms,
        'ane_io_ms': ane_io_ms,
        'ane_tflops': ane_tflops,
        'ane_loss': ane_loss,
        'gpu_adam_ms': gpu_avg_ms,
        'gpu_tflops': gpu_tflops,
        'gpu_loss': gpu_loss,
    }


def print_comparison(r):
    """Print comparison table for one model."""
    print(f"\n  Model: {r['model']}")
    print(f"  {'Metric':<30} {'ANE':>15} {'GPU (MPS)':>15} {'Winner':>10}")
    print("  " + "-" * 70)

    def row(name, a, g, lower_better=True, fmt='.1f', unit=''):
        if a is not None and g is not None:
            if lower_better:
                winner = "ANE" if a < g else "GPU"
                ratio = g / a if a != 0 else 0
            else:
                winner = "ANE" if a > g else "GPU"
                ratio = a / g if g != 0 else 0
            print(f"  {name:<30} {a:>12{fmt}}{unit:>3} {g:>12{fmt}}{unit:>3}  {winner} {ratio:.1f}x")
        elif a is not None:
            print(f"  {name:<30} {a:>12{fmt}}{unit:>3} {'—':>15}  {'':>10}")
        elif g is not None:
            print(f"  {name:<30} {'—':>15} {g:>12{fmt}}{unit:>3}  {'':>10}")

    row("Adam step (ms)", r['ane_adam_ms'], r['gpu_adam_ms'], lower_better=True)
    if r.get('ane_micro_ms'):
        print(f"  {'  Micro-step (ms)':<30} {r['ane_micro_ms']:>12.1f}{'ms':>3} {'—':>15}")
    if r.get('ane_hw_ms'):
        pct = r['ane_hw_ms'] / r['ane_micro_ms'] * 100 if r['ane_micro_ms'] else 0
        print(f"  {'  ANE hardware (ms)':<30} {r['ane_hw_ms']:>12.1f}{'ms':>3} {'':>15}  {pct:.0f}% of step")
    if r.get('ane_cpu_ms'):
        pct = r['ane_cpu_ms'] / r['ane_micro_ms'] * 100 if r['ane_micro_ms'] else 0
        print(f"  {'  CPU ops (ms)':<30} {r['ane_cpu_ms']:>12.1f}{'ms':>3} {'':>15}  {pct:.0f}% of step")
    if r.get('ane_io_ms'):
        pct = r['ane_io_ms'] / r['ane_micro_ms'] * 100 if r['ane_micro_ms'] else 0
        print(f"  {'  IOSurface I/O (ms)':<30} {r['ane_io_ms']:>12.1f}{'ms':>3} {'':>15}  {pct:.0f}% of step")
    row("Throughput (TFLOPS)", r['ane_tflops'], r['gpu_tflops'], lower_better=False, fmt='.3f')
    row("Final loss", r['ane_loss'], r['gpu_loss'], lower_better=True, fmt='.4f')


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else 'results'

    models = ['tiny_ane', 'stories110m']
    results = []

    for m in models:
        r = compare_model(results_dir, m)
        if r:
            results.append(r)

    # Also check for legacy filenames (gpu_results.json, ane_log.txt)
    if not results:
        gpu_json = os.path.join(results_dir, 'gpu_results.json')
        ane_log = os.path.join(results_dir, 'ane_log.txt')
        if os.path.exists(gpu_json) or os.path.exists(ane_log):
            # Copy to tiny_ane format and re-run
            if os.path.exists(gpu_json) and not os.path.exists(
                    os.path.join(results_dir, 'tiny_ane_gpu_results.json')):
                import shutil
                shutil.copy(gpu_json, os.path.join(results_dir, 'tiny_ane_gpu_results.json'))
            if os.path.exists(ane_log) and not os.path.exists(
                    os.path.join(results_dir, 'tiny_ane_ane_log.txt')):
                import shutil
                shutil.copy(ane_log, os.path.join(results_dir, 'tiny_ane_ane_log.txt'))
            r = compare_model(results_dir, 'tiny_ane')
            if r:
                results.append(r)

    if not results:
        print("No benchmark results found.")
        return

    print("\n" + "=" * 78)
    print("  ANE vs GPU (MPS) Training Benchmark")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("  Hardware: Apple M3 Pro (11-core CPU, 14-core GPU, 16-core ANE)")
    print("=" * 78)

    for r in results:
        print_comparison(r)

    # Summary table if multiple models
    if len(results) > 1:
        print(f"\n  {'='*74}")
        print(f"  Summary Table")
        print(f"  {'='*74}")
        print(f"  {'Model':<18} {'ANE ms':>10} {'GPU ms':>10} {'ANE TFLOPS':>12} {'GPU TFLOPS':>12} {'Speed':>10}")
        print(f"  {'-'*74}")
        for r in results:
            ane_ms = f"{r['ane_adam_ms']:.0f}" if r['ane_adam_ms'] else "—"
            gpu_ms = f"{r['gpu_adam_ms']:.0f}" if r['gpu_adam_ms'] else "—"
            ane_tf = f"{r['ane_tflops']:.3f}" if r['ane_tflops'] else "—"
            gpu_tf = f"{r['gpu_tflops']:.3f}" if r['gpu_tflops'] else "—"
            if r['ane_adam_ms'] and r['gpu_adam_ms']:
                ratio = r['gpu_adam_ms'] / r['ane_adam_ms']
                speed = f"{'ANE' if ratio > 1 else 'GPU'} {max(ratio, 1/ratio):.1f}x"
            else:
                speed = "—"
            print(f"  {r['model']:<18} {ane_ms:>10} {gpu_ms:>10} {ane_tf:>12} {gpu_tf:>12} {speed:>10}")

    # Verdict
    print(f"\n{'='*78}")
    for r in results:
        if r['ane_adam_ms'] and r['gpu_adam_ms']:
            ratio = r['gpu_adam_ms'] / r['ane_adam_ms']
            if ratio > 1:
                print(f"  {r['model']}: ANE is {ratio:.1f}x faster")
            else:
                print(f"  {r['model']}: GPU is {1/ratio:.1f}x faster")
            if r.get('ane_cpu_ms') and r.get('ane_micro_ms'):
                cpu_pct = r['ane_cpu_ms'] / r['ane_micro_ms'] * 100
                print(f"    ANE bottleneck: CPU ops = {cpu_pct:.0f}% of step time")
    print(f"{'='*78}")

    # Save combined results
    combined = {
        'date': datetime.now().isoformat(),
        'hardware': 'Apple M3 Pro',
        'models': results,
    }
    out = os.path.join(results_dir, 'comparison.json')
    with open(out, 'w') as f:
        json.dump(combined, f, indent=2)
    print(f"\n  Results saved to {out}")


if __name__ == '__main__':
    main()
