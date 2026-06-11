# -*- coding: utf-8 -*-
"""
Bloco Mamba-2 (SSD — State Space Duality; Dao & Gu, 2024).

Diferenças estruturais vs Mamba-1 (S6, em layers.py):
  • A deixa de ser uma diagonal [Di, N] e vira UM ESCALAR POR CABEÇA [H]
    ⇒ o decaimento a_t = exp(Δ_t·A_h) é escalar por (cabeça, passo);
  • o canal interno é dividido em H cabeças de dimensão P (headdim), como na atenção;
    B_t e C_t são compartilhados entre as cabeças (análogo de multi-value attention);
  • discretização: ā = exp(Δ·A) (ZOH exata em A) e B̄x = Δ·B⊗x — a forma simplificada
    é a DEFINIÇÃO oficial do Mamba-2 (no Mamba-1 deste repo usamos a ZOH exata em B);
  • projeções paralelas: z, x, B, C e Δ saem todos de UMA projeção de entrada,
    com a convolução causal aplicada ao grupo concatenado [x ‖ B ‖ C];
  • normalização com porta (NormFormer): y = RMSNorm(y ⊙ SiLU(z)) antes do out_proj.

Por que isso importa: com decaimento escalar, a matriz de mistura temporal fatoriza em
    M[t,s] = (Π_{s<i≤t} a_i) · ⟨C_t, B_s⟩       (s ≤ t)
— uma máscara 1-semiseparável ⊙ um produto externo C·Bᵀ. O algoritmo SSD por blocos
abaixo computa isso com GEMMs densos (amigo de tensor core na GPU e de BLAS na CPU),
sem nenhum laço por passo de tempo, e o autograd diferencia direto (sem backward
custom: não há recorrência explícita no grafo).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RMSNorm

_CHUNK = 32      # tamanho do bloco do algoritmo SSD


def ssd_chunked(x: torch.Tensor, dt: torch.Tensor, A: torch.Tensor,
                Bm: torch.Tensor, Cm: torch.Tensor) -> torch.Tensor:
    """
    Núcleo SSD por blocos (fp32, diferenciável por autograd).

    x:  [B, L, H, P]  valores por cabeça        dt: [B, L, H]  passo Δ > 0
    A:  [H]           decaimento escalar (< 0)  Bm/Cm: [B, L, N]  (grupo único, G=1)
    Retorna y: [B, L, H, P]  (sem o caminho-D e sem a porta; o caller aplica).

    Por bloco de comprimento c:
        cs_t  = Σ_{i≤t} Δ_i·A                                  [B, c, H]
        L[t,s] = exp(cs_t − cs_s)  (s ≤ t; −inf antes do exp)  [B, c, c, H]
        y_intra[t] = Σ_{s≤t} L[t,s]·⟨C_t, B_s⟩·Δ_s·x_s          (GEMMs densos)
        y_inter[t] = exp(cs_t) · ⟨C_t, h_entrada⟩               (estado dos blocos prévios)
        h_saída    = Σ_s exp(cs_last − cs_s)·Δ_s·(x_s ⊗ B_s) + exp(cs_last)·h_entrada
    Expoentes sempre ≤ 0 (A < 0) ⇒ exp ∈ (0, 1]: numericamente estável.
    """
    Bsz, L, H, P = x.shape
    N = Bm.shape[-1]
    dA = dt * A                                                  # [B, L, H]
    dtx = dt.unsqueeze(-1) * x                                   # Δ·x   [B, L, H, P]

    h_carry = x.new_zeros(Bsz, H, P, N)                          # estado entre blocos
    outs = []
    for s0 in range(0, L, _CHUNK):
        dA_seg = dA[:, s0:s0 + _CHUNK]                           # [B, c, H]
        dtx_seg = dtx[:, s0:s0 + _CHUNK]                         # [B, c, H, P]
        B_seg = Bm[:, s0:s0 + _CHUNK]                            # [B, c, N]
        C_seg = Cm[:, s0:s0 + _CHUNK]                            # [B, c, N]
        c = dA_seg.shape[1]

        cs = torch.cumsum(dA_seg, dim=1)                         # [B, c, H]
        E = cs.unsqueeze(2) - cs.unsqueeze(1)                    # E[b,t,s,h] = cs_t − cs_s
        tril = torch.ones(c, c, dtype=torch.bool, device=x.device).tril()
        Lmat = E.masked_fill(~tril.view(1, c, c, 1), float("-inf")).exp()  # [B,c,c,H]

        CB = torch.einsum("btn,bsn->bts", C_seg, B_seg)          # ⟨C_t, B_s⟩  [B, c, c]
        M = CB.unsqueeze(-1) * Lmat                              # máscara semiseparável
        y_intra = torch.einsum("btsh,bshp->bthp", M, dtx_seg)    # [B, c, H, P]

        decay_in = torch.exp(cs)                                 # [B, c, H]
        y_inter = decay_in.unsqueeze(-1) * torch.einsum(
            "btn,bhpn->bthp", C_seg, h_carry)                    # [B, c, H, P]
        outs.append(y_intra + y_inter)

        decay_out = torch.exp(cs[:, -1:] - cs)                   # exp(cs_last − cs_s)
        h_carry = (torch.exp(cs[:, -1]).unsqueeze(-1).unsqueeze(-1) * h_carry
                   + torch.einsum("bsh,bshp,bsn->bhpn", decay_out, dtx_seg, B_seg))

    return torch.cat(outs, dim=1)                                # [B, L, H, P]


class Mamba2Block(nn.Module):
    """
    x ─ in_proj ─┬─ [x‖B‖C] ─ conv1d causal ─ SiLU ─ SSD(Δ, A_h, B, C) ── +D·x ─┐
                 ├─ Δ_raw (por cabeça)                                          │
                 └─ z ──────────────── SiLU ────────────── ⊙ ── RMSNorm ── out_proj
    """

    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int,
                 headdim: int, dt_min: float, dt_max: float):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state                                   # N
        self.d_conv = d_conv
        self.d_inner = expand * d_model                          # Di
        assert self.d_inner % headdim == 0, "d_inner deve ser múltiplo de headdim"
        self.headdim = headdim                                   # P
        self.n_ssm_heads = self.d_inner // headdim               # H
        self.conv_dim = self.d_inner + 2 * d_state               # canais de [x‖B‖C]

        # projeção única de entrada: [z ‖ x ‖ B ‖ C ‖ Δ_raw]
        self.in_proj = nn.Linear(
            d_model, 2 * self.d_inner + 2 * d_state + self.n_ssm_heads, bias=False)
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, kernel_size=d_conv,
                                groups=self.conv_dim, padding=d_conv - 1, bias=True)

        # Δ: viés por cabeça com softplus(dt_bias) ~ LogUniforme[dt_min, dt_max]
        dt = torch.exp(torch.rand(self.n_ssm_heads)
                       * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))  # softplus⁻¹
        self.dt_bias._no_weight_decay = True

        # A escalar por cabeça: A ~ −Uniforme[1, 16]  (init oficial do Mamba-2)
        A = torch.empty(self.n_ssm_heads).uniform_(1.0, 16.0)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.n_ssm_heads))      # skip por cabeça
        self.D._no_weight_decay = True

        self.norm = RMSNorm(self.d_inner)                        # norma com porta
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def _split(self, zxbcdt: torch.Tensor):
        """Separa a projeção concatenada em (z, xBC, Δ_raw)."""
        return torch.split(
            zxbcdt, [self.d_inner, self.conv_dim, self.n_ssm_heads], dim=-1)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # u: [B, L, D]
        Bsz, L, _ = u.shape
        H, P, N = self.n_ssm_heads, self.headdim, self.d_state

        z, xBC, dt_raw = self._split(self.in_proj(u))            # [B,L,Di],[B,L,Di+2N],[B,L,H]
        xBC = self.conv1d(xBC.transpose(1, 2))[..., :L].transpose(1, 2)  # conv causal
        xBC = F.silu(xBC)
        x, Bm, Cm = torch.split(xBC, [self.d_inner, N, N], dim=-1)

        x = x.view(Bsz, L, H, P)                                 # cabeças  [B, L, H, P]
        dt = F.softplus(dt_raw.float() + self.dt_bias.float())   # Δ > 0    [B, L, H]
        A = -torch.exp(self.A_log.float())                       # A ≺ 0    [H]

        y = ssd_chunked(x.float(), dt, A, Bm.float(), Cm.float())  # [B, L, H, P]
        y = y + self.D.float().view(1, 1, H, 1) * x.float()      # caminho de salto
        y = y.reshape(Bsz, L, self.d_inner).to(u.dtype)

        y = self.norm(y * F.silu(z))                             # porta + RMSNorm
        return self.out_proj(y)                                  # [B, L, D]

    # ---- inferência recorrente: h [B,H,P,N] + janela da conv [B,Di+2N,k−1] ----

    def init_cache(self, batch: int, device: torch.device) -> dict:
        return {
            "conv": torch.zeros(batch, self.conv_dim, self.d_conv - 1, device=device),
            "h": torch.zeros(batch, self.n_ssm_heads, self.headdim, self.d_state,
                             device=device),
        }

    def step(self, u: torch.Tensor, cache: dict) -> torch.Tensor:
        # u: [B, D] → [B, D] — um passo da recorrência, custo O(1) no comprimento
        Bsz = u.shape[0]
        H, P, N = self.n_ssm_heads, self.headdim, self.d_state

        z, xBC, dt_raw = self._split(self.in_proj(u))            # [B,Di],[B,Di+2N],[B,H]
        win = torch.cat([cache["conv"], xBC.float().unsqueeze(-1)], dim=-1)
        cache["conv"] = win[:, :, 1:]
        w = self.conv1d.weight.squeeze(1).float()                # [Di+2N, k]
        xBC = F.silu((win * w).sum(-1) + self.conv1d.bias.float())
        x, Bm, Cm = torch.split(xBC, [self.d_inner, N, N], dim=-1)

        x = x.view(Bsz, H, P)                                    # [B, H, P]
        dt = F.softplus(dt_raw.float() + self.dt_bias.float())   # [B, H]
        A = -torch.exp(self.A_log.float())                       # [H]

        a_bar = torch.exp(dt * A)                                # [B, H]
        dtx = dt.unsqueeze(-1) * x                               # [B, H, P]
        cache["h"] = (a_bar.unsqueeze(-1).unsqueeze(-1) * cache["h"]
                      + dtx.unsqueeze(-1) * Bm.unsqueeze(1).unsqueeze(1))  # [B,H,P,N]

        y = torch.einsum("bn,bhpn->bhp", Cm, cache["h"]) \
            + self.D.float().view(1, H, 1) * x                   # [B, H, P]
        y = y.reshape(Bsz, self.d_inner).to(u.dtype)
        y = self.norm(y * F.silu(z))
        return self.out_proj(y)                                  # [B, D]
