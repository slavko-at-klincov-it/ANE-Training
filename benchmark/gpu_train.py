#!/usr/bin/env python3
"""GPU (MPS) training baseline — Tiny-ANE-15M Llama2-style model.
Exact architecture & hyperparameter match with ANE training pipeline.
Outputs JSON metrics for comparison with ANE benchmark.
"""

import argparse
import json
import math
import os
import struct
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Model Architecture (matches training/training_dynamic/models/tiny_ane.h)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).to(x.dtype) * self.weight


def precompute_rope(dim, seq_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    # x: [B, H, S, HD]
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    cos = cos[:x.shape[2], :d].unsqueeze(0).unsqueeze(0)
    sin = sin[:x.shape[2], :d].unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class Attention(nn.Module):
    def __init__(self, dim, n_heads, head_dim):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)

    def forward(self, x, cos, sin):
        B, S, _ = x.shape
        q = self.wq(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.wo(out)


class FFN(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, head_dim, hidden):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = Attention(dim, n_heads, head_dim)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = FFN(dim, hidden)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class TinyLlama(nn.Module):
    def __init__(self, vocab, dim, hidden, n_heads, head_dim, n_layers, seq_len):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, head_dim, hidden)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab, bias=False)
        # Tie weights
        self.output.weight = self.embed.weight

        self.seq_len = seq_len
        self.head_dim = head_dim
        cos, sin = precompute_rope(head_dim, seq_len)
        self.register_buffer('rope_cos', cos)
        self.register_buffer('rope_sin', sin)

    def forward(self, tokens):
        x = self.embed(tokens)
        for layer in self.layers:
            x = layer(x, self.rope_cos, self.rope_sin)
        x = self.norm(x)
        return self.output(x)


# ---------------------------------------------------------------------------
# Weight initialization (matches ANE: GPT-2 style init)
# ---------------------------------------------------------------------------

def init_weights(model, dim, hidden, n_layers):
    """Match ANE training init: uniform in [-scale, scale] with GPT-2 output scaling."""
    torch.manual_seed(42)
    scale_d = 1.0 / math.sqrt(dim)
    scale_h = 1.0 / math.sqrt(hidden)
    out_scale = 1.0 / math.sqrt(n_layers)

    with torch.no_grad():
        # Embedding
        model.embed.weight.uniform_(-0.02, 0.02)

        for i, layer in enumerate(model.layers):
            # Attention
            layer.attn.wq.weight.uniform_(-scale_d, scale_d)
            layer.attn.wk.weight.uniform_(-scale_d, scale_d)
            layer.attn.wv.weight.uniform_(-scale_d, scale_d)
            layer.attn.wo.weight.uniform_(-scale_d * out_scale, scale_d * out_scale)
            # FFN
            layer.ffn.w1.weight.uniform_(-scale_h, scale_h)
            layer.ffn.w3.weight.uniform_(-scale_h, scale_h)
            layer.ffn.w2.weight.uniform_(-scale_d * out_scale, scale_d * out_scale)
            # RMSNorm weights are already 1.0


# ---------------------------------------------------------------------------
# Data loading (matches ANE: binary uint16 tokens, random sampling)
# ---------------------------------------------------------------------------

def load_tokens(path):
    with open(path, 'rb') as f:
        data = f.read()
    return torch.frombuffer(bytearray(data), dtype=torch.int16).to(torch.long)


def get_batch(tokens, seq_len, device):
    n = len(tokens) - seq_len - 1
    pos = torch.randint(0, n, (1,)).item()
    x = tokens[pos:pos + seq_len].unsqueeze(0).to(device)
    y = tokens[pos + 1:pos + 1 + seq_len].unsqueeze(0).to(device)
    return x, y


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("mps")
    dtype = torch.float32  # MPS doesn't support float16 training well; use fp32

    # Model configs (match ANE training headers)
    MODELS = {
        'tiny_ane': dict(name='Tiny-ANE-15M', vocab=32000, dim=256, hidden=768,
                         n_heads=4, head_dim=64, n_layers=6, seq_len=256),
        'stories110m': dict(name='Stories-110M', vocab=32000, dim=768, hidden=2048,
                            n_heads=12, head_dim=64, n_layers=12, seq_len=256),
    }
    model_key = args.model
    if model_key not in MODELS:
        print(f"Unknown model: {model_key}. Available: {list(MODELS.keys())}")
        return
    cfg = {k: v for k, v in MODELS[model_key].items() if k != 'name'}
    model_name = MODELS[model_key]['name']

    model = TinyLlama(**cfg).to(device)
    init_weights(model, cfg['dim'], cfg['hidden'], cfg['n_layers'])

    n_params = sum(p.numel() for p in model.parameters())
    # Unique params (embedding tied with output)
    n_unique = n_params - model.output.weight.numel()
    print(f"Model: {model_name} on GPU (MPS)")
    print(f"Params: {n_unique/1e6:.1f}M unique ({n_params/1e6:.1f}M total with tied)")
    print(f"Config: dim={cfg['dim']} hidden={cfg['hidden']} heads={cfg['n_heads']} "
          f"layers={cfg['n_layers']} seq={cfg['seq_len']} vocab={cfg['vocab']}")

    # FLOPs per step (same formula as ANE: 6 * n_params * seq_len for fwd+bwd)
    flops_per_step = 6 * n_unique * cfg['seq_len']

    # Optimizer (matches ANE: AdamW, beta2=0.95)
    no_decay = {'attn_norm.weight', 'ffn_norm.weight', 'norm.weight'}
    param_groups = [
        {'params': [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95),
                                   eps=1e-8)

    # Load data
    tokens = load_tokens(args.data)
    print(f"Data: {len(tokens)} tokens from {args.data}")

    # Training
    results = []
    total_steps = args.steps
    warmup_steps = args.warmup
    accum_steps = args.accum

    print(f"\nTraining: {total_steps} steps, accum={accum_steps}, "
          f"lr={args.lr}, warmup={warmup_steps}")
    print(f"{'='*70}")

    optimizer.zero_grad()
    accum_loss = 0.0

    # Warmup
    for warmup_i in range(3):
        x, y = get_batch(tokens, cfg['seq_len'], device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, cfg['vocab']), y.view(-1))
        loss.backward()
        optimizer.zero_grad()
    torch.mps.synchronize()

    t_total_start = time.perf_counter()

    for step in range(total_steps):
        # Cosine LR schedule with warmup
        if step < warmup_steps:
            lr = args.lr * (step + 1) / warmup_steps
        else:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            lr = args.min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (args.lr - args.min_lr)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        t_step_start = time.perf_counter()

        # Gradient accumulation
        for micro in range(accum_steps):
            x, y = get_batch(tokens, cfg['seq_len'], device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, cfg['vocab']), y.view(-1))
            (loss / accum_steps).backward()
            accum_loss += loss.item() / accum_steps

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()
        optimizer.zero_grad()

        torch.mps.synchronize()
        t_step_end = time.perf_counter()
        step_ms = (t_step_end - t_step_start) * 1000.0

        # Compute TFLOPS
        total_flops = flops_per_step * accum_steps
        tflops = total_flops / (step_ms / 1000.0) / 1e12

        if step % args.log_every == 0 or step == 0:
            print(f"step {step:4d}  loss={accum_loss:.4f}  lr={lr:.2e}  "
                  f"{step_ms:.1f}ms/step  {tflops:.2f} TFLOPS")

        results.append({
            'step': step,
            'loss': accum_loss,
            'lr': lr,
            'step_ms': step_ms,
            'tflops': tflops,
        })
        accum_loss = 0.0

    t_total = time.perf_counter() - t_total_start

    # Summary
    avg_ms = sum(r['step_ms'] for r in results) / len(results)
    avg_tflops = sum(r['tflops'] for r in results) / len(results)
    final_loss = results[-1]['loss']

    print(f"\n{'='*70}")
    print(f"GPU (MPS) Training Complete")
    print(f"  Steps: {total_steps}, Accum: {accum_steps}")
    print(f"  Avg step: {avg_ms:.1f} ms")
    print(f"  Avg TFLOPS: {avg_tflops:.2f}")
    print(f"  Final loss: {final_loss:.4f}")
    print(f"  Total time: {t_total:.1f}s")

    # Save results
    summary = {
        'device': 'GPU (MPS)',
        'model': model_name,
        'params_m': n_unique / 1e6,
        'steps': total_steps,
        'accum_steps': accum_steps,
        'avg_step_ms': avg_ms,
        'avg_tflops': avg_tflops,
        'final_loss': final_loss,
        'total_time_s': t_total,
        'flops_per_step': flops_per_step,
        'per_step': results,
    }
    out_path = args.output or 'gpu_results.json'
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Results saved to {out_path}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='GPU (MPS) Training Benchmark')
    p.add_argument('--data', default='../training/tinystories_data00.bin',
                   help='Path to tokenized training data (uint16 binary)')
    p.add_argument('--steps', type=int, default=100, help='Training steps')
    p.add_argument('--accum', type=int, default=10, help='Gradient accumulation steps')
    p.add_argument('--lr', type=float, default=3e-4, help='Peak learning rate')
    p.add_argument('--min-lr', type=float, default=3e-5, help='Min learning rate')
    p.add_argument('--warmup', type=int, default=10, help='Warmup steps')
    p.add_argument('--weight-decay', type=float, default=0.1, help='Weight decay')
    p.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clip norm')
    p.add_argument('--log-every', type=int, default=10, help='Log every N steps')
    p.add_argument('--model', default='tiny_ane',
                   choices=['tiny_ane', 'stories110m'],
                   help='Model architecture (default: tiny_ane)')
    p.add_argument('--output', default=None, help='Output JSON path')
    train(p.parse_args())
