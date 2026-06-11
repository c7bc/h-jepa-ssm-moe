# -*- coding: utf-8 -*-
"""
Camadas da espinha dorsal: RMSNorm, RoPE, Atenção Local de Janela (SWA) e Bloco Mamba.

Cada mixer expõe DOIS caminhos:
  • forward(x [B,L,D])          — treino / avaliação em lote (paralelo no tempo);
  • init_cache + step(x [B,D])  — inferência recorrente token a token, custo O(1)
    por token (a vantagem decisiva do SSM sobre o Transformer na geração).
A equivalência exata entre os dois caminhos é verificada na suíte de testes.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .scan import _SMALL, selective_scan


class RMSNorm(nn.Module):
    """RMSNorm com acumulação em fp32 (estável sob AMP bf16/fp16)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., D] → [..., D]
        x32 = x.float()
        normed = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (normed * self.weight.float()).to(x.dtype)


# ═════════════════════════════════════════════════════════════════════════════════════
# RoPE — só as camadas de atenção carregam posição explícita (o SSM é recorrente)
# ═════════════════════════════════════════════════════════════════════════════════════


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)  # [hd/2]

    def forward(self, L: int, device: torch.device):
        t = torch.arange(L, device=device, dtype=torch.float32)       # [L]
        ang = torch.outer(t, self.inv_freq)                           # [L, hd/2]
        return ang.cos(), ang.sin()

    def at(self, pos: int, device: torch.device):
        """cos/sin de UMA posição absoluta (inferência recorrente). [hd/2] cada."""
        ang = pos * self.inv_freq.to(device)
        return ang.cos(), ang.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, L, H, hd]; cos/sin: [L, hd/2] — rotação por pares (metades inferior/superior)
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def _rope_single(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, hd]; cos/sin: [hd/2]
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


# ═════════════════════════════════════════════════════════════════════════════════════
# Atenção Local Esparsa de Janela Causal (camada de recall do Samba)
# ═════════════════════════════════════════════════════════════════════════════════════


class LocalWindowAttention(nn.Module):
    """
    Atenção causal de janela K em TILES de bloco — o padrão de acesso à memória do
    FlashAttention com sliding window: a matriz L×L nunca é materializada; cada bloco
    de K queries visita só 2K chaves (bloco anterior + atual), custo O(L·K), softmax
    em fp32. Papel na topologia híbrida: recuperação exata ("agulha no palheiro") que
    a compressão recorrente do SSM não garante.
    """

    def __init__(self, d_model: int, n_heads: int, window: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window = window                                     # K
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        Bsz, L, Dm = x.shape
        H, hd, K = self.n_heads, self.head_dim, self.window

        qkv = self.qkv(x).view(Bsz, L, 3, H, hd)                 # [B, L, 3, H, hd]
        q, k, v = qkv.unbind(dim=2)                              # [B, L, H, hd]

        cos, sin = self.rope(L, x.device)
        q = apply_rope(q.float(), cos, sin)
        k = apply_rope(k.float(), cos, sin)
        v = v.float()

        pad = (K - L % K) % K                                    # completa múltiplo de K
        if pad:
            q = F.pad(q, (0, 0, 0, 0, 0, pad))
            k = F.pad(k, (0, 0, 0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, 0, 0, pad))
        Lp = L + pad
        nb = Lp // K

        # tiling: [B, L, H, hd] → [B, H, nb, K, hd]
        q = q.view(Bsz, nb, K, H, hd).permute(0, 3, 1, 2, 4)
        k = k.view(Bsz, nb, K, H, hd).permute(0, 3, 1, 2, 4)
        v = v.view(Bsz, nb, K, H, hd).permute(0, 3, 1, 2, 4)

        # chaves do bloco i: [bloco i−1 ‖ bloco i] — cobre a janela causal K
        k_prev = F.pad(k, (0, 0, 0, 0, 1, 0))[:, :, :-1]
        v_prev = F.pad(v, (0, 0, 0, 0, 1, 0))[:, :, :-1]
        kk = torch.cat([k_prev, k], dim=3)                       # [B, H, nb, 2K, hd]
        vv = torch.cat([v_prev, v], dim=3)

        scores = torch.einsum("bhnqd,bhnkd->bhnqk", q, kk) / math.sqrt(hd)  # [B,H,nb,K,2K]

        # g_q − g_k = q − k + K ∈ [0, K−1]  ⇔  q < k ≤ q + K  (causal + janela K)
        qi = torch.arange(K, device=x.device)[:, None]
        ki = torch.arange(2 * K, device=x.device)[None, :]
        allowed = (ki > qi) & (ki <= qi + K)                     # [K, 2K]
        first_ok = allowed & (ki >= K)                           # bloco 0 não tem anterior
        blk = torch.arange(nb, device=x.device)[:, None, None]
        mask = torch.where(blk == 0, first_ok[None], allowed[None])  # [nb, K, 2K]
        scores = scores.masked_fill(~mask[None, None], float("-inf"))

        probs = torch.softmax(scores, dim=-1)                    # fp32
        out = torch.einsum("bhnqk,bhnkd->bhnqd", probs, vv)      # [B, H, nb, K, hd]
        out = out.permute(0, 2, 3, 1, 4).reshape(Bsz, Lp, H * hd)[:, :L]
        return self.out_proj(out.to(x.dtype))

    # ---- inferência recorrente: cache deslizante de K−1 chaves/valores ----

    def init_cache(self, batch: int, device: torch.device) -> dict:
        H, hd = self.n_heads, self.head_dim
        return {
            "k": torch.zeros(batch, H, 0, hd, device=device),
            "v": torch.zeros(batch, H, 0, hd, device=device),
            "pos": 0,
        }

    def step(self, x: torch.Tensor, cache: dict) -> torch.Tensor:
        # x: [B, D] → [B, D]  (atende às últimas ≤K posições, incluindo a atual)
        Bsz = x.shape[0]
        H, hd, K = self.n_heads, self.head_dim, self.window
        q, k, v = self.qkv(x).view(Bsz, 3, H, hd).unbind(dim=1)  # [B, H, hd]
        cos, sin = self.rope.at(cache["pos"], x.device)
        q = _rope_single(q.float(), cos, sin)
        k = _rope_single(k.float(), cos, sin)

        ks = torch.cat([cache["k"], k.unsqueeze(2)], dim=2)      # [B, H, t≤K, hd]
        vs = torch.cat([cache["v"], v.float().unsqueeze(2)], dim=2)
        att = (q.unsqueeze(2) * ks).sum(-1) / math.sqrt(hd)      # [B, H, t]
        probs = torch.softmax(att, dim=-1)
        out = (probs.unsqueeze(-1) * vs).sum(2)                  # [B, H, hd]

        cache["k"] = ks[:, :, -(K - 1):] if K > 1 else ks[:, :, :0]
        cache["v"] = vs[:, :, -(K - 1):] if K > 1 else vs[:, :, :0]
        cache["pos"] += 1
        return self.out_proj(out.reshape(Bsz, H * hd).to(x.dtype))


# ═════════════════════════════════════════════════════════════════════════════════════
# Bloco Mamba (S6: SSM seletivo)
# ═════════════════════════════════════════════════════════════════════════════════════


class MambaBlock(nn.Module):
    """
    x ─ in_proj ─┬─ conv1d causal ─ SiLU ─ S6(Δ(x), B(x), C(x)) ─┐
                 │                                               ⊙ ─ out_proj
                 └────────────────── SiLU (porta z) ─────────────┘
    """

    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int,
                 dt_min: float, dt_max: float):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state                                   # N
        self.d_conv = d_conv
        self.d_inner = expand * d_model                          # Di
        self.dt_rank = math.ceil(d_model / 16)                   # R

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                groups=self.d_inner, padding=d_conv - 1, bias=True)
        # projeção seletiva: x ↦ (Δ_raw ∈ R^R, B ∈ R^N, C ∈ R^N)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # init de Δ: softplus(dt_bias) ~ LogUniforme[dt_min, dt_max]
        dt_init_std = self.dt_rank**-0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(torch.rand(self.d_inner)
                       * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        inv_dt = dt + torch.log(-torch.expm1(-dt))               # softplus⁻¹(dt)
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # init S4D-real: A[d, n] = −(n+1), parametrizada em log (negatividade garantida)
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        A = A.unsqueeze(0).expand(self.d_inner, -1).contiguous() # [Di, N]
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        Bsz, L, _ = x.shape
        xz = self.in_proj(x)                                     # [B, L, 2·Di]
        xa, z = xz.chunk(2, dim=-1)                              # [B, L, Di]

        xa = xa.transpose(1, 2)                                  # [B, Di, L]
        xa = self.conv1d(xa)[..., :L]                            # causal: corta lookahead
        xa = xa.transpose(1, 2)                                  # [B, L, Di]
        xa = F.silu(xa)

        dbc = self.x_proj(xa)                                    # [B, L, R + 2N]
        dt_raw, Bm, Cm = torch.split(
            dbc, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt_raw))                    # Δ > 0  [B, L, Di]

        A = -torch.exp(self.A_log.float())                       # A ≺ 0  [Di, N]
        y = selective_scan(xa, dt, A, Bm, Cm, self.D)            # [B, L, Di]

        y = y * F.silu(z)
        return self.out_proj(y)                                  # [B, L, D]

    # ---- inferência recorrente: estado h [B,Di,N] + janela da conv [B,Di,k−1] ----

    def init_cache(self, batch: int, device: torch.device) -> dict:
        return {
            "conv": torch.zeros(batch, self.d_inner, self.d_conv - 1, device=device),
            "h": torch.zeros(batch, self.d_inner, self.d_state, device=device),
        }

    def step(self, x: torch.Tensor, cache: dict) -> torch.Tensor:
        # x: [B, D] → [B, D] — UM passo da recorrência, custo O(1) no comprimento
        xz = self.in_proj(x)                                     # [B, 2·Di]
        xa, z = xz.chunk(2, dim=-1)                              # [B, Di]

        win = torch.cat([cache["conv"], xa.float().unsqueeze(-1)], dim=-1)  # [B, Di, k]
        cache["conv"] = win[:, :, 1:]
        w = self.conv1d.weight.squeeze(1).float()                # [Di, k]
        xa = F.silu((win * w).sum(-1) + self.conv1d.bias.float())  # [B, Di]

        dbc = self.x_proj(xa.to(self.x_proj.weight.dtype))
        dt_raw, Bm, Cm = torch.split(
            dbc, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )                                                        # [B,R], [B,N], [B,N]
        dt = F.softplus(self.dt_proj(dt_raw)).float()            # [B, Di]

        A = -torch.exp(self.A_log.float())                       # [Di, N]
        dA = dt.unsqueeze(-1) * A                                # [B, Di, N]
        a_bar = torch.exp(dA)
        safe_A = torch.where(A == 0, torch.ones_like(A), A)
        r = torch.where(dA.abs() < _SMALL, dt.unsqueeze(-1).expand_as(dA),
                        torch.expm1(dA) / safe_A)
        h = a_bar * cache["h"] + r * Bm.float().unsqueeze(1) * xa.unsqueeze(-1)
        cache["h"] = h                                           # [B, Di, N]

        y = (h * Cm.float().unsqueeze(1)).sum(-1) + self.D.float() * xa  # [B, Di]
        y = y.to(x.dtype) * F.silu(z)
        return self.out_proj(y)                                  # [B, D]
