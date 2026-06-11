# -*- coding: utf-8 -*-
"""
DEMO VISÍVEL: mini modelo de linguagem (nível de caractere) com o backbone híbrido
Mamba/SWA/MoE, treinado em texto público (Machado de Assis) — 100% em CPU.

Uso:
    python train_text.py                                   # treina e gera amostras
    python train_text.py --steps 600                       # treino mais curto
    python train_text.py sample --prompt "Eu era "         # gera a partir do checkpoint

O modelo aprende, caractere por caractere e do zero, a estatística do português de
1881. A geração usa o caminho recorrente do SSM: custo O(1) por token — repare que a
velocidade de geração NÃO cai conforme o texto cresce (no Transformer, cai).
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import asdict

import torch

from hjepa.config import LMConfig
from hjepa.data import CharVocab, get_batch, load_char_corpus
from hjepa.lm import HybridCharLM
from hjepa.train import build_lr_lambda, build_optimizer, count_params


def evaluate(model: HybridCharLM, data: torch.Tensor, seq_len: int,
             batch_size: int, device: torch.device, n_batches: int = 5) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = get_batch(data, seq_len, batch_size, device)
            _, ce, _ = model.loss(x, y)
            losses.append(ce.item())
    model.train()
    return sum(losses) / len(losses)


def sample_and_print(model: HybridCharLM, vocab: CharVocab, device: torch.device,
                     prompt: str, n_tokens: int, temperature: float, top_k: int) -> None:
    idx = vocab.encode(prompt).unsqueeze(0).to(device)
    if idx.shape[1] == 0:
        idx = vocab.encode("\n").unsqueeze(0).to(device)
    t0 = time.perf_counter()
    out = model.generate(idx, n_tokens, temperature=temperature, top_k=top_k)
    dt = time.perf_counter() - t0
    text = vocab.decode(out[0])
    print(f"┌─ amostra ({n_tokens} chars em {dt:.1f}s ≈ {n_tokens/dt:.0f} chars/s, "
          f"inferência recorrente O(1)/token) " + "─" * 8)
    for line in text.splitlines():
        print(f"│ {line}")
    print("└" + "─" * 72)


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text = load_char_corpus(args.data)
    vocab = CharVocab(text)
    data = vocab.encode(text)                                   # [n_chars] int64
    n_val = max(1000, int(0.1 * data.shape[0]))
    train_data, val_data = data[:-n_val], data[-n_val:]
    print(f"[dados] {data.shape[0]:,} chars | vocab={len(vocab)} | "
          f"treino={train_data.shape[0]:,} | val={val_data.shape[0]:,}")

    cfg = LMConfig(vocab_size=len(vocab), d_model=args.d_model, depth=args.depth,
                   n_heads=4, window=args.window, d_state=8, ff_mult=2,
                   attn_every=10, ssm_version=args.ssm_version)
    model = HybridCharLM(cfg).to(device).train()
    swa = [i + 1 for i in range(cfg.depth) if (i + 1) % cfg.attn_every == 0]
    print(f"[modelo] {count_params(model)/1e6:.2f}M parâmetros | "
          f"Mamba-{cfg.ssm_version} | D={cfg.d_model} | depth={cfg.depth} "
          f"(SWA nas camadas {swa}) | {cfg.n_experts} experts top-{cfg.top_k}")

    optimizer = build_optimizer(model, args.lr, weight_decay=0.05, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, build_lr_lambda(warmup_steps=max(10, args.steps // 30),
                                   total_steps=args.steps, min_lr_ratio=0.1))

    print(f"[treino] {args.steps} passos | B={args.batch_size} | L={args.seq_len} | "
          f"device={device.type} | loss inicial esperada ≈ ln({len(vocab)}) = "
          f"{math.log(len(vocab)):.2f}")
    t_prev = time.perf_counter()
    for step in range(args.steps):
        x, y = get_batch(train_data, args.seq_len, args.batch_size, device)  # [B, L]
        loss, ce, aux = model.loss(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % args.log_every == 0:
            t_now = time.perf_counter()
            cps = args.batch_size * args.seq_len * args.log_every / max(1e-9, t_now - t_prev)
            t_prev = t_now
            print(f"iter {step:05d}/{args.steps:05d} | ce {ce.item():6.4f} "
                  f"(ppl {math.exp(min(20, ce.item())):7.2f}) | moe_lb {aux.item():8.6f} | "
                  f"lr {scheduler.get_last_lr()[0]:.2e} | {cps:7.0f} chars/s")
        if args.eval_every and step and step % args.eval_every == 0:
            val_ce = evaluate(model, val_data, args.seq_len, args.batch_size, device)
            print(f"  └─ val_ce {val_ce:.4f} (ppl {math.exp(val_ce):.2f})")
        if args.sample_every and step and step % args.sample_every == 0:
            sample_and_print(model, vocab, device, args.prompt, 200,
                             args.temperature, args.top_k)

    val_ce = evaluate(model, val_data, args.seq_len, args.batch_size, device)
    print(f"[final] val_ce {val_ce:.4f} (ppl {math.exp(val_ce):.2f})")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(), "config": asdict(cfg),
                "itos": vocab.itos}, args.out)
    print(f"[checkpoint] salvo em {args.out}")
    sample_and_print(model, vocab, device, args.prompt, args.tokens,
                     args.temperature, args.top_k)


def sample_from_ckpt(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    cfg = LMConfig(**ckpt["config"])
    model = HybridCharLM(cfg).to(device).eval()
    model.load_state_dict(ckpt["model"])
    vocab = CharVocab.__new__(CharVocab)
    vocab.itos = ckpt["itos"]
    vocab.stoi = {ch: i for i, ch in enumerate(vocab.itos)}
    sample_and_print(model, vocab, device, args.prompt, args.tokens,
                     args.temperature, args.top_k)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mini-LM híbrido Mamba/SWA/MoE (demo)")
    parser.add_argument("mode", nargs="?", choices=("train", "sample"), default="train")
    parser.add_argument("--data", type=str, default=None,
                        help="arquivo .txt local ou URL (padrão: Machado de Assis)")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=192)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--ssm-version", type=int, choices=(1, 2), default=2,
                        help="2 = Mamba-2/SSD (padrão); 1 = Mamba-1/S6")
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--sample-every", type=int, default=300)
    parser.add_argument("--out", type=str, default="out/lm_machado.pt")
    parser.add_argument("--ckpt", type=str, default="out/lm_machado.pt")
    parser.add_argument("--prompt", type=str, default="Ao verme que ")
    parser.add_argument("--tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.mode == "sample":
        sample_from_ckpt(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
