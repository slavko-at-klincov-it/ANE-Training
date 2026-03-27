#!/usr/bin/env python3
"""Parse powermetrics output and generate ANE vs GPU power comparison."""

import json
import os
import re
import sys
from datetime import datetime


def parse_powermetrics(path):
    """Parse powermetrics text output for power readings (mW).

    powermetrics outputs blocks separated by '***' with lines like:
      ANE Power: 42 mW
      GPU Power: 1523 mW
      CPU Power: 3200 mW
      Combined Power (CPU + GPU + ANE): 4765 mW
      Package Power: 5200 mW
    """
    if not os.path.exists(path):
        return None

    samples = []
    current = {}

    with open(path) as f:
        for line in f:
            line = line.strip()

            # New sample block
            if line.startswith('***'):
                if current:
                    samples.append(current)
                current = {}
                continue

            # Parse power lines (various formats Apple uses)
            for pattern, key in [
                (r'ANE Power:\s+([\d.]+)\s*mW', 'ane_mw'),
                (r'GPU Power:\s+([\d.]+)\s*mW', 'gpu_mw'),
                (r'CPU Power:\s+([\d.]+)\s*mW', 'cpu_mw'),
                (r'Package Power:\s+([\d.]+)\s*mW', 'package_mw'),
                (r'Combined Power.*?:\s+([\d.]+)\s*mW', 'combined_mw'),
            ]:
                m = re.search(pattern, line)
                if m:
                    current[key] = float(m.group(1))

    if current:
        samples.append(current)

    if not samples:
        return None

    # Average all samples
    result = {}
    for key in ['ane_mw', 'gpu_mw', 'cpu_mw', 'package_mw', 'combined_mw']:
        vals = [s[key] for s in samples if key in s]
        if vals:
            result[key] = sum(vals) / len(vals)
    result['n_samples'] = len(samples)
    return result


def parse_inference_speed(log_path, json_path):
    """Get tokens/sec from inference logs."""
    # Try JSON first (GPU)
    if json_path and os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
        return {
            'tokens_per_sec': data.get('avg_tokens_per_sec'),
            'ms_per_token': data.get('avg_ms_per_token'),
            'tflops': data.get('avg_tflops'),
        }

    # Parse ANE log
    if log_path and os.path.exists(log_path):
        with open(log_path) as f:
            for line in f:
                m = re.search(r'([\d.]+)\s*ms/token,\s*([\d.]+)\s*tokens/sec', line)
                if m:
                    return {
                        'ms_per_token': float(m.group(1)),
                        'tokens_per_sec': float(m.group(2)),
                    }
    return None


def parse_training_speed(log_path, json_path):
    """Get step time from training logs."""
    if json_path and os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
        return {
            'avg_step_ms': data.get('avg_step_ms'),
            'avg_tflops': data.get('avg_tflops'),
        }

    if log_path and os.path.exists(log_path):
        step_times = []
        accum = 10  # default
        with open(log_path) as f:
            for line in f:
                m_acc = re.search(r'Accum\s+(\d+)\s+steps', line)
                if m_acc:
                    accum = int(m_acc.group(1))
                m = re.search(r'([\d.]+)ms/step', line)
                if m:
                    step_times.append(float(m.group(1)))
        if step_times and len(step_times) > 2:
            avg_micro = sum(step_times[2:]) / len(step_times[2:])
            # ANE reports per micro-step; Adam step = accum micro-steps
            return {'avg_step_ms': avg_micro * accum}
    return None


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else 'results'

    # Parse all power data
    idle = parse_powermetrics(os.path.join(results_dir, 'power_idle.txt'))
    gpu_inf = parse_powermetrics(os.path.join(results_dir, 'power_gpu_inference.txt'))
    ane_inf = parse_powermetrics(os.path.join(results_dir, 'power_ane_inference.txt'))
    gpu_train = parse_powermetrics(os.path.join(results_dir, 'power_gpu_training.txt'))
    ane_train = parse_powermetrics(os.path.join(results_dir, 'power_ane_training.txt'))

    # Parse speed data
    gpu_inf_speed = parse_inference_speed(
        None, os.path.join(results_dir, 'power_gpu_inference.json'))
    ane_inf_speed = parse_inference_speed(
        os.path.join(results_dir, 'power_ane_inference_log.txt'), None)
    gpu_train_speed = parse_training_speed(
        None, os.path.join(results_dir, 'power_gpu_training.json'))
    ane_train_speed = parse_training_speed(
        os.path.join(results_dir, 'power_ane_training_log.txt'), None)

    print(f"\n{'='*80}")
    print(f"  ANE vs GPU — Power & Energy Efficiency Benchmark")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Hardware: Apple M3 Pro")
    print(f"{'='*80}")

    # --- Idle baseline ---
    if idle:
        print(f"\n  Idle Baseline:")
        for k in ['cpu_mw', 'gpu_mw', 'ane_mw', 'package_mw']:
            if k in idle:
                label = k.replace('_mw', '').upper()
                print(f"    {label}: {idle[k]:.0f} mW")

    # --- Power rails comparison ---
    def print_power_table(label, ane_power, gpu_power):
        if not ane_power or not gpu_power:
            print(f"\n  {label}: No power data available")
            return

        print(f"\n  {label}")
        print(f"  {'Power Rail':<20} {'ANE run':>12} {'GPU run':>12} {'Diff':>12}")
        print(f"  {'-'*56}")

        for key, name in [('ane_mw', 'ANE'), ('gpu_mw', 'GPU'), ('cpu_mw', 'CPU'),
                          ('package_mw', 'Package')]:
            a = ane_power.get(key)
            g = gpu_power.get(key)
            if a is not None and g is not None:
                diff = a - g
                sign = "+" if diff > 0 else ""
                print(f"  {name + ' Power':<20} {a:>9.0f} mW {g:>9.0f} mW {sign}{diff:>8.0f} mW")
            elif a is not None:
                print(f"  {name + ' Power':<20} {a:>9.0f} mW {'—':>12}")
            elif g is not None:
                print(f"  {name + ' Power':<20} {'—':>12} {g:>9.0f} mW")

    print_power_table("Inference Power (Stories-110M)", ane_inf, gpu_inf)
    print_power_table("Training Power (Stories-110M)", ane_train, gpu_train)

    # --- Energy efficiency ---
    print(f"\n  {'='*56}")
    print(f"  Energy Efficiency")
    print(f"  {'='*56}")

    def energy_row(label, ane_power, gpu_power, ane_speed, gpu_speed, speed_key, speed_unit):
        if not all([ane_power, gpu_power, ane_speed, gpu_speed]):
            return

        ane_pkg = ane_power.get('package_mw') or ane_power.get('combined_mw')
        gpu_pkg = gpu_power.get('package_mw') or gpu_power.get('combined_mw')
        ane_s = ane_speed.get(speed_key)
        gpu_s = gpu_speed.get(speed_key)

        if not all([ane_pkg, gpu_pkg, ane_s, gpu_s]):
            # Fall back to sum of known rails
            if not ane_pkg:
                ane_pkg = sum(v for k, v in ane_power.items()
                              if k in ('cpu_mw', 'gpu_mw', 'ane_mw'))
            if not gpu_pkg:
                gpu_pkg = sum(v for k, v in gpu_power.items()
                              if k in ('cpu_mw', 'gpu_mw', 'ane_mw'))
            if not all([ane_pkg, gpu_pkg, ane_s, gpu_s]):
                return

        ane_w = ane_pkg / 1000.0
        gpu_w = gpu_pkg / 1000.0

        print(f"\n  {label}:")
        print(f"    {'Metric':<28} {'ANE':>12} {'GPU':>12} {'Winner':>12}")
        print(f"    {'-'*64}")
        print(f"    {'System power (W)':<28} {ane_w:>10.1f} W {gpu_w:>10.1f} W "
              f"{'ANE' if ane_w < gpu_w else 'GPU'}")

        # Speed
        print(f"    {'Speed (' + speed_unit + ')':<28} {ane_s:>10.1f}   {gpu_s:>10.1f}   "
              f"{'ANE' if ane_s > gpu_s else 'GPU'}")

        # Energy per unit of work
        if speed_key == 'tokens_per_sec':
            # mJ per token = (W * 1000) / tok_s
            ane_mj = (ane_w * 1000.0) / ane_s
            gpu_mj = (gpu_w * 1000.0) / gpu_s
            unit_label = "Energy/token (mJ)"
        else:
            # mJ per step = W * ms
            ane_mj = ane_w * ane_s
            gpu_mj = gpu_w * gpu_s
            unit_label = "Energy/step (mJ)"

        winner = "ANE" if ane_mj < gpu_mj else "GPU"
        ratio = max(ane_mj, gpu_mj) / min(ane_mj, gpu_mj) if min(ane_mj, gpu_mj) > 0 else 0
        print(f"    {unit_label:<28} {ane_mj:>9.1f} mJ {gpu_mj:>9.1f} mJ "
              f"{winner} {ratio:.1f}x better")

        # Tokens/Joule or Steps/Joule
        if speed_key == 'tokens_per_sec':
            ane_tj = ane_s / ane_w if ane_w > 0 else 0
            gpu_tj = gpu_s / gpu_w if gpu_w > 0 else 0
            eff_label = "Tokens/Joule"
        else:
            ane_tj = 1000.0 / (ane_w * ane_s) if (ane_w * ane_s) > 0 else 0
            gpu_tj = 1000.0 / (gpu_w * gpu_s) if (gpu_w * gpu_s) > 0 else 0
            eff_label = "Steps/Joule"

        winner = "ANE" if ane_tj > gpu_tj else "GPU"
        ratio = max(ane_tj, gpu_tj) / min(ane_tj, gpu_tj) if min(ane_tj, gpu_tj) > 0 else 0
        print(f"    {eff_label:<28} {ane_tj:>10.1f}   {gpu_tj:>10.1f}   "
              f"{winner} {ratio:.1f}x better")

    energy_row("Inference", ane_inf, gpu_inf,
               ane_inf_speed, gpu_inf_speed, 'tokens_per_sec', 'tok/s')
    energy_row("Training", ane_train, gpu_train,
               ane_train_speed, gpu_train_speed, 'avg_step_ms', 'ms/step')

    # --- Summary ---
    print(f"\n{'='*80}")
    summary_lines = []

    if ane_inf and gpu_inf:
        ane_pkg = ane_inf.get('package_mw', 0)
        gpu_pkg = gpu_inf.get('package_mw', 0)
        if ane_pkg and gpu_pkg:
            pct = (1 - ane_pkg / gpu_pkg) * 100
            if pct > 0:
                summary_lines.append(
                    f"  Inference: ANE uses {pct:.0f}% less system power")
            else:
                summary_lines.append(
                    f"  Inference: GPU uses {-pct:.0f}% less system power")

    if ane_train and gpu_train:
        ane_pkg = ane_train.get('package_mw', 0)
        gpu_pkg = gpu_train.get('package_mw', 0)
        if ane_pkg and gpu_pkg:
            pct = (1 - ane_pkg / gpu_pkg) * 100
            if pct > 0:
                summary_lines.append(
                    f"  Training: ANE uses {pct:.0f}% less system power")
            else:
                summary_lines.append(
                    f"  Training: GPU uses {-pct:.0f}% less system power")

    for line in summary_lines:
        print(line)
    print(f"{'='*80}")

    # Save results
    combined = {
        'date': datetime.now().isoformat(),
        'hardware': 'Apple M3 Pro',
        'idle': idle,
        'inference': {
            'ane_power': ane_inf,
            'gpu_power': gpu_inf,
            'ane_speed': ane_inf_speed,
            'gpu_speed': gpu_inf_speed,
        },
        'training': {
            'ane_power': ane_train,
            'gpu_power': gpu_train,
            'ane_speed': ane_train_speed,
            'gpu_speed': gpu_train_speed,
        },
    }
    out = os.path.join(results_dir, 'power_comparison.json')
    with open(out, 'w') as f:
        json.dump(combined, f, indent=2)
    print(f"\n  Results saved to {out}")


if __name__ == '__main__':
    main()
