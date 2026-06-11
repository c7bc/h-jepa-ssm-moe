# -*- coding: utf-8 -*-
"""
Espinha dorsal híbrida (Samba): blocos pré-norm residuais com mixer temporal
(Mamba ×9 : SWA ×1) seguido de FFN esparso (MoE). Suporta o caminho paralelo de
treino (forward) e o caminho recorrente de inferência (init_cache/step, O(1)/token).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .layers import LocalWindowAttention, MambaBlock, RMSNorm
from .mamba2 import Mamba2Block
from .moe import SparseMoE


class HybridBlock(nn.Module):
    """x ← x + Mixer(RMS(x));  x ← x + MoE(RMS(x))."""

    def __init__(self, cfg, d_model: int, use_attention: bool):
        super().__init__()
        self.use_attention = use_attention
        self.norm1 = RMSNorm(d_model)
        if use_attention:
            self.mixer: nn.Module = LocalWindowAttention(d_model, cfg.n_heads, cfg.window)
        elif getattr(cfg, "ssm_version", 2) == 2:
            self.mixer = Mamba2Block(d_model, cfg.d_state, cfg.d_conv, cfg.expand,
                                     cfg.headdim, cfg.dt_min, cfg.dt_max)
        else:
            self.mixer = MambaBlock(d_model, cfg.d_state, cfg.d_conv,
                                    cfg.expand, cfg.dt_min, cfg.dt_max)
        self.norm2 = RMSNorm(d_model)
        self.moe = SparseMoE(d_model, cfg.ff_mult * d_model, cfg.n_experts,
                             cfg.top_k, cfg.w_balance, cfg.router_noise)

    def forward(self, x: torch.Tensor):
        # x: [B, L, D]
        x = x + self.mixer(self.norm1(x))                       # mistura temporal
        ff, aux = self.moe(self.norm2(x))                       # mistura de canais
        x = x + ff
        return x, aux

    def step(self, x: torch.Tensor, cache: dict) -> torch.Tensor:
        # x: [B, D] — um token; MoE roteia normalmente (T = B tokens)
        x = x + self.mixer.step(self.norm1(x), cache)
        ff, _ = self.moe(self.norm2(x).unsqueeze(1))            # [B, 1, D]
        return x + ff.squeeze(1)


class HybridStack(nn.Module):
    """depth blocos; a cada `attn_every` camadas a última é SWA ⇒ (attn_every−1):1."""

    def __init__(self, cfg, d_model: int, depth: int):
        super().__init__()
        self.blocks = nn.ModuleList(
            HybridBlock(cfg, d_model, use_attention=((i + 1) % cfg.attn_every == 0))
            for i in range(depth)
        )

    def forward(self, x: torch.Tensor):
        # x: [B, L, D] → (h: [B, L, D], aux_total: escalar fp32)
        aux_total = torch.zeros((), device=x.device, dtype=torch.float32)
        for block in self.blocks:
            x, aux = block(x)
            aux_total = aux_total + aux
        return x, aux_total

    def init_cache(self, batch: int, device: torch.device) -> list[dict]:
        return [blk.mixer.init_cache(batch, device) for blk in self.blocks]

    def step(self, x: torch.Tensor, caches: list[dict]) -> torch.Tensor:
        # x: [B, D] → [B, D]
        for block, cache in zip(self.blocks, caches):
            x = block.step(x, cache)
        return x
