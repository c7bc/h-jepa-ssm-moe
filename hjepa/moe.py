# -*- coding: utf-8 -*-
"""
Mistura de Especialistas esparsa, dropless (MegaBlocks) com roteador Noisy Top-k.

O conjunto {(token t, expert e, porta g)} é um grafo bipartido dinâmico, materializado
como lista de arestas COO [T·k]; `argsort` por expert produz segmentos contíguos por
especialista — o análogo exato dos blocos densos de um Block-Sparse GEMM (grouped
GEMM), sem padding nem descarte por fator de capacidade.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """Especialista: MLP independente D → D_ff → D com SiLU."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [n_e, D] → [n_e, D]
        return self.w2(F.silu(self.w1(x)))


class SparseMoE(nn.Module):
    """
    1. Roteador Noisy Top-k (Shazeer 2017): H(x) = x·W_g + ε ⊙ softplus(x·W_noise),
       ε ~ N(0, I), só em treino. Roteador SEMPRE em fp32 (estabilidade sob AMP).
    2. Despacho por lista de arestas ordenada (segmentos contíguos = Block-Sparse GEMM).
    3. Recombinação por index_add_ (scatter-add) ponderada pelo softmax das rotas.
    4. Perda de balanceamento (Switch): L_aux = w · E · Σ_e f_e · P_e
       (f_e = fração de arestas despachadas a e, sem grad; P_e = prob. média, com grad).
    """

    def __init__(self, d_model: int, d_ff: int, n_experts: int, top_k: int,
                 w_balance: float, router_noise: float):
        super().__init__()
        self.n_experts = n_experts                                  # E
        self.top_k = top_k                                          # k
        self.w_balance = w_balance
        self.router_noise = router_noise
        self.experts = nn.ModuleList(Expert(d_model, d_ff) for _ in range(n_experts))
        self.w_gate = nn.Linear(d_model, n_experts, bias=False)
        self.w_noise = nn.Linear(d_model, n_experts, bias=False)
        nn.init.normal_(self.w_gate.weight, std=0.02)
        nn.init.zeros_(self.w_noise.weight)                         # σ inicial = softplus(0)

    def forward(self, x: torch.Tensor):
        # x: [B, L, D] → (y: [B, L, D], aux: escalar fp32)
        Bsz, L, Dm = x.shape
        xf = x.reshape(-1, Dm)                                      # [T, D], T = B·L
        T = xf.shape[0]
        E, k = self.n_experts, self.top_k

        # ---- 1) roteador em fp32 puro (fora do autocast) ----
        with torch.autocast(device_type=xf.device.type, enabled=False):
            xr = xf.float()
            clean_logits = F.linear(xr, self.w_gate.weight.float()) # [T, E]
            if self.training and self.router_noise > 0:
                noise_std = F.softplus(
                    F.linear(xr, self.w_noise.weight.float())
                ) * self.router_noise                               # [T, E]
                noisy_logits = clean_logits + torch.randn_like(clean_logits) * noise_std
            else:
                noisy_logits = clean_logits

            top_val, top_idx = noisy_logits.topk(k, dim=-1)         # [T, k]
            gates = torch.softmax(top_val, dim=-1)                  # softmax nas rotas [T, k]

            probs = torch.softmax(clean_logits, dim=-1)             # [T, E]
            P = probs.mean(dim=0)                                   # [E]
            with torch.no_grad():
                f = F.one_hot(top_idx, num_classes=E).sum(dim=(0, 1)).float() / (T * k)
            aux = self.w_balance * E * torch.sum(f * P)             # escalar fp32

        # ---- 2) grafo de despacho: lista de arestas (token → expert) ----
        flat_e = top_idx.reshape(-1)                                # [T·k]
        flat_t = torch.arange(T, device=xf.device).repeat_interleave(k)
        flat_g = gates.reshape(-1)                                  # [T·k]

        order = torch.argsort(flat_e, stable=True)                  # agrupa por expert
        sorted_t = flat_t[order]
        sorted_g = flat_g[order]
        counts = torch.bincount(flat_e, minlength=E)                # [E]

        gathered = torch.gather(
            xf, 0, sorted_t.unsqueeze(-1).expand(-1, Dm)
        )                                                           # [T·k, D]

        # ---- 3) "Block-Sparse GEMM": um GEMM denso por segmento contíguo ----
        out_buf = torch.zeros_like(gathered)
        offset = 0
        for e in range(E):
            n_e = int(counts[e])
            if n_e == 0:
                continue                                            # dropless
            seg = gathered[offset:offset + n_e]                     # [n_e, D]
            out_buf[offset:offset + n_e] = self.experts[e](seg)
            offset += n_e

        out_buf = out_buf * sorted_g.unsqueeze(-1).to(out_buf.dtype)

        y = torch.zeros_like(xf)                                    # [T, D]
        y.index_add_(0, sorted_t, out_buf)                          # Σ das k rotas
        return y.view(Bsz, L, Dm), aux
