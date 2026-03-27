#!/usr/bin/env python3
"""Parse ANE and GPU inference benchmark results and generate comparison."""

import json
import os
import re
import sys
from datetime import datetime


def parse_ane_inference_log(path):
    """Parse ANE generate output for timing."""
    if not os.path.exists(path):
        return None

    tokens = 0
    total_ms = 0
    ms_per_token = 0
    tokens_per_sec = 0
    model_name = "Unknown"

    with open(path) as f:
        for line in f:
            m_name = re.search(r'Model:\s+(.+?)(?:\s*\(|$)', line)
            if m_name:
                model_name = m_name.group(1).strip()

            m = re.search(r'Tokens generated:\s+(\d+)', line)
            if m:
                tokens = int(m.group(1))

            m = re.search(r'Total time:\s+([\d.]+)ms\s+\(([\d.]+)\s*ms/token,\s*([\d.]+)\s*tokens/sec\)', line)
            if m:
                total_ms = float(m.group(1))
                ms_per_token = float(m.group(2))
                tokens_per_sec = float(m.group(3))

    if tokens == 0:
        return None

    return {
        'model': model_name,
        'tokens': tokens,
        'total_ms': total_ms,
        'ms_per_token': ms_per_token,
        'tokens_per_sec': tokens_per_sec,
    }


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else 'results'
    models = ['tiny_ane', 'stories110m']

    print(f"\n{'='*78}")
    print(f"  ANE vs GPU (MPS) Inference Benchmark")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Hardware: Apple M3 Pro (14-core GPU, 16-core ANE)")
    print(f"  Mode: Full-sequence recompute (no KV cache)")
    print(f"{'='*78}")

    all_results = []

    for model_key in models:
        # GPU results
        gpu_json = os.path.join(results_dir, f'{model_key}_gpu_inference.json')
        gpu = None
        if os.path.exists(gpu_json):
            with open(gpu_json) as f:
                gpu = json.load(f)

        # ANE results
        ane_log = os.path.join(results_dir, f'{model_key}_ane_inference_log.txt')
        ane = parse_ane_inference_log(ane_log)

        if not gpu and not ane:
            continue

        model_name = gpu.get('model', model_key) if gpu else (ane.get('model', model_key) if ane else model_key)

        gpu_ms = gpu.get('avg_ms_per_token') if gpu else None
        gpu_toks = gpu.get('avg_tokens_per_sec') if gpu else None
        gpu_tflops = gpu.get('avg_tflops') if gpu else None
        ane_ms = ane.get('ms_per_token') if ane else None
        ane_toks = ane.get('tokens_per_sec') if ane else None

        # Compute ANE TFLOPS (2 * params * seq_len / time)
        ane_tflops = None
        if ane_ms and gpu:
            flops_fwd = gpu.get('flops_per_forward', 0)
            if flops_fwd and ane_ms > 0:
                ane_tflops = flops_fwd / (ane_ms / 1000.0) / 1e12

        print(f"\n  Model: {model_name}")
        print(f"  {'Metric':<28} {'ANE':>15} {'GPU (MPS)':>15} {'Winner':>12}")
        print(f"  {'-'*70}")

        def row(name, a, g, lower_better=True, fmt='.1f', unit=''):
            if a is not None and g is not None:
                if lower_better:
                    winner = "ANE" if a < g else "GPU"
                    ratio = g / a if a != 0 else 0
                else:
                    winner = "ANE" if a > g else "GPU"
                    ratio = a / g if g != 0 else 0
                print(f"  {name:<28} {a:>12{fmt}}{unit:>3} {g:>12{fmt}}{unit:>3}  {winner} {ratio:.1f}x")
            elif a is not None:
                print(f"  {name:<28} {a:>12{fmt}}{unit:>3} {'—':>15}")
            elif g is not None:
                print(f"  {name:<28} {'—':>15} {g:>12{fmt}}{unit:>3}")

        row("ms/token", ane_ms, gpu_ms, lower_better=True)
        row("tokens/sec", ane_toks, gpu_toks, lower_better=False)
        row("TFLOPS", ane_tflops, gpu_tflops, lower_better=False, fmt='.2f')

        all_results.append({
            'model': model_name,
            'ane_ms': ane_ms,
            'ane_toks': ane_toks,
            'ane_tflops': ane_tflops,
            'gpu_ms': gpu_ms,
            'gpu_toks': gpu_toks,
            'gpu_tflops': gpu_tflops,
        })

    if len(all_results) > 1:
        print(f"\n  {'='*70}")
        print(f"  Summary")
        print(f"  {'='*70}")
        print(f"  {'Model':<18} {'ANE tok/s':>12} {'GPU tok/s':>12} {'ANE TFLOPS':>12} {'GPU TFLOPS':>12} {'Speed':>10}")
        print(f"  {'-'*70}")
        for r in all_results:
            at = f"{r['ane_toks']:.1f}" if r['ane_toks'] else "—"
            gt = f"{r['gpu_toks']:.1f}" if r['gpu_toks'] else "—"
            af = f"{r['ane_tflops']:.2f}" if r['ane_tflops'] else "—"
            gf = f"{r['gpu_tflops']:.2f}" if r['gpu_tflops'] else "—"
            if r['ane_toks'] and r['gpu_toks']:
                ratio = r['ane_toks'] / r['gpu_toks']
                speed = f"{'ANE' if ratio > 1 else 'GPU'} {max(ratio, 1/ratio):.1f}x"
            else:
                speed = "—"
            print(f"  {r['model']:<18} {at:>12} {gt:>12} {af:>12} {gf:>12} {speed:>10}")

    print(f"\n{'='*78}")
    for r in all_results:
        if r['ane_toks'] and r['gpu_toks']:
            ratio = r['ane_toks'] / r['gpu_toks']
            if ratio > 1:
                print(f"  {r['model']}: ANE is {ratio:.1f}x faster")
            else:
                print(f"  {r['model']}: GPU is {1/ratio:.1f}x faster")
    print(f"{'='*78}")

    out = os.path.join(results_dir, 'inference_comparison.json')
    with open(out, 'w') as f:
        json.dump({'date': datetime.now().isoformat(), 'results': all_results}, f, indent=2)
    print(f"\n  Results saved to {out}")


if __name__ == '__main__':
    main()
