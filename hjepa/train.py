# -*- coding: utf-8 -*-
"""Utilidades de treino (otimizador, schedules, AMP) e o loop de treino do H-JEPA."""

from __future__ import annotations

import math
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import HJepaConfig
from .data import SyntheticSequenceDataset
from .jepa import HJEPASSMMoE
from .scan import HAS_TRITON


def ema_momentum_schedule(step: int, total_steps: int, base: float) -> float:
    """m(t): base → 1.0 em cosseno (estilo BYOL/I-JEPA)."""
    cos_t = (math.cos(math.pi * step / max(1, total_steps)) + 1.0) / 2.0
    return 1.0 - (1.0 - base) * cos_t


def build_optimizer(model: nn.Module, lr: float, weight_decay: float,
                    betas: tuple[float, float]) -> torch.optim.AdamW:
    """AdamW em grupos: sem weight decay em vetores, normas, A_log, D, pos/mask tokens."""
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:                                  # ex.: target encoder EMA
            continue
        if p.ndim < 2 or getattr(p, "_no_weight_decay", False):
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=1e-8)


def build_lr_lambda(warmup_steps: int, total_steps: int, min_lr_ratio: float):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)             # warmup linear
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cos = 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cos
    return lr_lambda


def make_grad_scaler(enabled: bool):
    """Compat: torch.amp.GradScaler (≥2.3) com fallback para torch.cuda.amp."""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):                          # pragma: no cover
        return torch.cuda.amp.GradScaler(enabled=enabled)


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def train_jepa(cfg: HJepaConfig, steps: int, batch_size: int, device: torch.device,
               use_amp: bool, num_workers: int, log_every: int, seed: int) -> None:
    """Loop de treino do H-JEPA com AMP, EMA agendado e logs discriminados por iteração."""
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    dataset = SyntheticSequenceDataset(
        num_samples=max(2048, steps * batch_size), seq_len=cfg.seq_len,
        input_dim=cfg.input_dim, action_dim=cfg.action_dim, seed=seed,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                        num_workers=num_workers, pin_memory=(device.type == "cuda"),
                        persistent_workers=(num_workers > 0))

    model = HJEPASSMMoE(cfg).to(device).train()
    print(f"[modelo] context_encoder: {count_params(model.context_encoder)/1e6:.2f}M | "
          f"target_encoder (EMA, congelado): {count_params(model.target_encoder)/1e6:.2f}M | "
          f"predictor: {count_params(model.predictor)/1e6:.2f}M")
    print(f"[modelo] treináveis: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M | "
          f"razão híbrida: {cfg.attn_every - 1} SSM : 1 SWA | "
          f"experts: {cfg.n_experts} (top-{cfg.top_k})")

    optimizer = build_optimizer(model, cfg.lr, cfg.weight_decay, (cfg.beta1, cfg.beta2))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, build_lr_lambda(cfg.warmup_steps, steps, cfg.min_lr_ratio))

    amp_dtype = (torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported())
                 else torch.float16) if device.type == "cuda" else torch.bfloat16
    use_amp = use_amp and device.type == "cuda"
    scaler = make_grad_scaler(enabled=use_amp and amp_dtype == torch.float16)
    print(f"[runtime] device={device.type} | "
          f"amp={'on (' + str(amp_dtype).split('.')[-1] + ')' if use_amp else 'off'} | "
          f"triton_scan={'disponível' if (HAS_TRITON and device.type == 'cuda') else 'indisponível'}")

    step = 0
    data_iter = iter(loader)
    t_prev = time.perf_counter()
    while step < steps:
        try:
            x, action = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, action = next(data_iter)
        x = x.to(device, non_blocking=True)                      # [B, L, D_in]
        action = action.to(device, non_blocking=True)            # [B, A]

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            out = model(x, action)
            loss = out["loss"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        m = ema_momentum_schedule(step, steps, cfg.ema_base)     # EMA após o otimizador
        model.update_target_encoder(m)

        if step % log_every == 0:
            t_now = time.perf_counter()
            tok_s = batch_size * cfg.seq_len * log_every / max(1e-9, t_now - t_prev)
            t_prev = t_now
            print(f"iter {step:05d}/{steps:05d} | loss {out['loss'].item():9.4f} | "
                  f"vicreg[inv {out['inv'].item():7.4f} | var {out['var'].item():7.4f} | "
                  f"cov {out['cov'].item():7.4f} | wtot {out['vic_total'].item():9.4f}] | "
                  f"moe_lb {out['moe_aux'].item():8.6f} | lr {scheduler.get_last_lr()[0]:.2e} | "
                  f"ema_m {m:.5f} | grad {grad_norm.item():7.3f} | "
                  f"n_tgt {out['n_targets']:5d} | {tok_s:9.1f} tok/s")
        step += 1

    print("[treino] concluído.")
