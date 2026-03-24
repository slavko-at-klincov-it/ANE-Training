#!/usr/bin/env python3
"""GPU (MPS) inference benchmark — matches ANE generate pipeline.
No KV cache (fair comparison: ANE also recomputes full sequence each token).
Measures tokens/sec, ms/token for direct ANE vs GPU comparison.
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
# Model (same as gpu_train.py)
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
        self.output.weight = self.embed.weight
        self.seq_len = seq_len
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
# Checkpoint loading (matches ANE BLZT format)
# ---------------------------------------------------------------------------

def load_checkpoint(model, path, cfg):
    """Load ANE checkpoint (BLZT format) into PyTorch model."""
    dim = cfg['dim']
    hidden = cfg['hidden']
    n_heads = cfg['n_heads']
    head_dim = cfg['head_dim']
    n_layers = cfg['n_layers']
    vocab = cfg['vocab']
    q_dim = n_heads * head_dim
    kv_dim = n_heads * head_dim  # MHA

    with open(path, 'rb') as f:
        # Header: magic, version, step, total, nlayers, vocab, dim, hidden,
        #         nheads, seq, lr(float), loss(float),
        #         cum_train(double), cum_wall(double), cum_steps, adam_t
        #         kv_heads, head_dim, q_dim
        hdr = f.read(4 * 10 + 4 * 2 + 8 * 2 + 4 * 2 + 4 * 3)
        magic = struct.unpack('<I', hdr[:4])[0]
        if magic != 0x424C5A54:
            raise ValueError(f"Bad magic: {magic:#x}, expected BLZT (0x424C5A54)")

        # Read layer weights
        with torch.no_grad():
            for L in range(n_layers):
                layer = model.layers[L]

                wq = torch.frombuffer(bytearray(f.read(q_dim * dim * 4)),
                                       dtype=torch.float32).reshape(q_dim, dim)
                wk = torch.frombuffer(bytearray(f.read(kv_dim * dim * 4)),
                                       dtype=torch.float32).reshape(kv_dim, dim)
                wv = torch.frombuffer(bytearray(f.read(kv_dim * dim * 4)),
                                       dtype=torch.float32).reshape(kv_dim, dim)
                wo = torch.frombuffer(bytearray(f.read(dim * q_dim * 4)),
                                       dtype=torch.float32).reshape(dim, q_dim)

                layer.attn.wq.weight.copy_(wq)
                layer.attn.wk.weight.copy_(wk)
                layer.attn.wv.weight.copy_(wv)
                layer.attn.wo.weight.copy_(wo)

                w1 = torch.frombuffer(bytearray(f.read(hidden * dim * 4)),
                                       dtype=torch.float32).reshape(hidden, dim)
                w2 = torch.frombuffer(bytearray(f.read(dim * hidden * 4)),
                                       dtype=torch.float32).reshape(dim, hidden)
                w3 = torch.frombuffer(bytearray(f.read(hidden * dim * 4)),
                                       dtype=torch.float32).reshape(hidden, dim)

                layer.ffn.w1.weight.copy_(w1)
                layer.ffn.w2.weight.copy_(w2)
                layer.ffn.w3.weight.copy_(w3)

                rms_att = torch.frombuffer(bytearray(f.read(dim * 4)),
                                            dtype=torch.float32)
                rms_ffn = torch.frombuffer(bytearray(f.read(dim * 4)),
                                            dtype=torch.float32)
                layer.attn_norm.weight.copy_(rms_att)
                layer.ffn_norm.weight.copy_(rms_ffn)

            # Final norm + embedding
            rms_final = torch.frombuffer(bytearray(f.read(dim * 4)),
                                          dtype=torch.float32)
            model.norm.weight.copy_(rms_final)

            embed = torch.frombuffer(bytearray(f.read(vocab * dim * 4)),
                                      dtype=torch.float32).reshape(vocab, dim)
            model.embed.weight.copy_(embed)
            # output.weight is tied to embed.weight, already set


# ---------------------------------------------------------------------------
# Inference benchmark
# ---------------------------------------------------------------------------

def benchmark_inference(args):
    device = torch.device("mps")

    MODELS = {
        'tiny_ane': dict(name='Tiny-ANE-15M', vocab=32000, dim=256, hidden=768,
                         n_heads=4, head_dim=64, n_layers=6, seq_len=256),
        'stories110m': dict(name='Stories-110M', vocab=32000, dim=768, hidden=2048,
                            n_heads=12, head_dim=64, n_layers=12, seq_len=256),
    }

    model_key = args.model
    cfg = {k: v for k, v in MODELS[model_key].items() if k != 'name'}
    model_name = MODELS[model_key]['name']

    model = TinyLlama(**cfg).to(device)

    # Load checkpoint if available
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        load_checkpoint(model, args.checkpoint, cfg)
        print("  Checkpoint loaded")
    else:
        print("  No checkpoint — using random weights (timing still valid)")

    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) - model.output.weight.numel()
    print(f"Model: {model_name} on GPU (MPS)")
    print(f"Params: {n_params/1e6:.1f}M")

    seq_len = cfg['seq_len']
    max_tokens = args.tokens

    # FLOPs per forward pass: 2 * n_params * seq_len (inference = forward only)
    flops_per_fwd = 2 * n_params * seq_len

    print(f"\nInference benchmark: {max_tokens} tokens, seq_len={seq_len}")
    print(f"Mode: Full-sequence recompute (no KV cache — matches ANE)")
    print(f"{'='*60}")

    # Build initial context (BOS + some tokens)
    context = [1]  # BOS token
    token_times = []

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            x = torch.tensor([context + [0] * (seq_len - len(context))],
                             dtype=torch.long, device=device)
            model(x)
        torch.mps.synchronize()

    t_total_start = time.perf_counter()

    with torch.no_grad():
        for i in range(max_tokens):
            # Pad context to seq_len
            ctx = context[-seq_len:]  # keep last seq_len tokens
            padded = ctx + [0] * (seq_len - len(ctx))
            x = torch.tensor([padded], dtype=torch.long, device=device)

            t0 = time.perf_counter()
            logits = model(x)
            torch.mps.synchronize()
            t1 = time.perf_counter()

            # Sample from last context position
            last_pos = min(len(ctx), seq_len) - 1
            next_logits = logits[0, last_pos, :]

            # Argmax (deterministic for benchmarking)
            next_token = next_logits.argmax().item()

            token_ms = (t1 - t0) * 1000.0
            token_times.append(token_ms)

            # Append to context
            context.append(next_token)

            if (i + 1) % 20 == 0 or i == 0:
                tok_s = 1000.0 / token_ms
                tflops = flops_per_fwd / (token_ms / 1000.0) / 1e12
                print(f"  token {i+1:4d}  {token_ms:.1f} ms  {tok_s:.1f} tok/s  {tflops:.2f} TFLOPS")

    t_total = time.perf_counter() - t_total_start

    # Skip first token (cold start)
    steady = token_times[1:] if len(token_times) > 1 else token_times
    avg_ms = sum(steady) / len(steady)
    avg_toks = 1000.0 / avg_ms
    avg_tflops = flops_per_fwd / (avg_ms / 1000.0) / 1e12

    print(f"\n{'='*60}")
    print(f"GPU (MPS) Inference Complete — {model_name}")
    print(f"  Tokens: {max_tokens}")
    print(f"  Avg: {avg_ms:.1f} ms/token  ({avg_toks:.1f} tokens/sec)")
    print(f"  Throughput: {avg_tflops:.2f} TFLOPS")
    print(f"  Total: {t_total:.1f}s")

    summary = {
        'device': 'GPU (MPS)',
        'model': model_name,
        'model_key': model_key,
        'params_m': n_params / 1e6,
        'tokens': max_tokens,
        'avg_ms_per_token': avg_ms,
        'avg_tokens_per_sec': avg_toks,
        'avg_tflops': avg_tflops,
        'total_time_s': t_total,
        'flops_per_forward': flops_per_fwd,
        'per_token_ms': token_times,
    }

    out_path = args.output or f'results/{model_key}_gpu_inference.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved to {out_path}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='GPU (MPS) Inference Benchmark')
    p.add_argument('--model', default='tiny_ane',
                   choices=['tiny_ane', 'stories110m'])
    p.add_argument('--checkpoint', default=None,
                   help='ANE checkpoint file (BLZT format)')
    p.add_argument('--tokens', type=int, default=100,
                   help='Number of tokens to generate')
    p.add_argument('--output', default=None)
    benchmark_inference(p.parse_args())
