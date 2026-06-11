# -*- coding: utf-8 -*-
"""
H-JEPA: encoders (contexto + alvo EMA), preditor latente e perda VICReg.

Predição em espaço de embeddings — sem decodificador generativo de pixels/tokens.
O Target Encoder é uma cópia EMA do Context Encoder com gradientes estritamente
congelados; o anticolapso vem da regularização geométrica não-contrastiva (VICReg).
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import HybridStack
from .config import HJepaConfig
from .layers import RMSNorm


class SequenceEncoder(nn.Module):
    """Projeção de entrada + pilha híbrida + RMSNorm final."""

    def __init__(self, cfg: HJepaConfig):
        super().__init__()
        self.embed = nn.Linear(cfg.input_dim, cfg.d_model)          # [.., D_in] → [.., D]
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        nn.init.normal_(self.mask_token, std=0.02)
        self.mask_token._no_weight_decay = True
        self.stack = HybridStack(cfg, cfg.d_model, cfg.depth)
        self.norm_f = RMSNorm(cfg.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        # x: [B, L, D_in]; mask: [B, L] bool (True = posição-alvo OCULTADA do contexto)
        h = self.embed(x)                                           # [B, L, D]
        if mask is not None:
            # SSMs exigem alinhamento temporal: posições-alvo são substituídas pelo
            # token de máscara (em vez de removidas, como num ViT do I-JEPA).
            h = torch.where(mask.unsqueeze(-1), self.mask_token.to(h.dtype), h)
        h, aux = self.stack(h)
        return self.norm_f(h), aux


class LatentPredictor(nn.Module):
    """
    Preditor g_φ: recebe z_ctx + variável de posição (mask token + embedding posicional
    nos slots-alvo) + variável de ação (condicionamento global) e prediz o embedding-
    alvo NO ESPAÇO LATENTE. Estreito (D_pred < D): gargalo informacional deliberado.
    """

    def __init__(self, cfg: HJepaConfig):
        super().__init__()
        dp = cfg.pred_d_model
        self.max_seq_len = cfg.max_seq_len
        self.in_proj = nn.Linear(cfg.d_model, dp)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, dp)            # variável de POSIÇÃO
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        self.pos_emb.weight._no_weight_decay = True
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dp))
        nn.init.normal_(self.mask_token, std=0.02)
        self.mask_token._no_weight_decay = True
        self.action_proj = nn.Linear(cfg.action_dim, dp)            # variável de AÇÃO
        self.stack = HybridStack(cfg, dp, cfg.pred_depth)
        self.norm_f = RMSNorm(dp)
        self.out_proj = nn.Linear(dp, cfg.d_model)

    def forward(self, z_ctx: torch.Tensor, target_mask: torch.Tensor,
                action: torch.Tensor | None = None):
        # z_ctx: [B, L, D]; target_mask: [B, L] bool; action: [B, A] ou None
        Bsz, L, _ = z_ctx.shape
        assert L <= self.max_seq_len, "seq_len excede max_seq_len do preditor"

        h = self.in_proj(z_ctx)                                     # [B, L, D_pred]
        h = torch.where(target_mask.unsqueeze(-1), self.mask_token.to(h.dtype), h)
        pos = self.pos_emb(torch.arange(L, device=z_ctx.device))    # [L, D_pred]
        h = h + pos.unsqueeze(0).to(h.dtype)
        if action is not None:
            h = h + self.action_proj(action).unsqueeze(1).to(h.dtype)

        h, aux = self.stack(h)
        return self.out_proj(self.norm_f(h)), aux                   # [B, L, D]


class VICRegLoss(nn.Module):
    """
    L = λ·Inv + μ·[Var(ẑ) + Var(z̄)] + ν·[Cov(ẑ) + Cov(z̄)]   (Bardes et al., 2022)

    Inv = MSE(ẑ, z̄);  Var = (1/D) Σ_d max(0, γ − √(Var_N(z_d) + ε));
    Cov = (1/D) Σ_{i≠j} [Cov(Z)]²_{ij}  — zera os termos fora da diagonal (anticolapso
    dimensional). Os termos do ramo-alvo são constantes em θ (alvo EMA/no_grad):
    contribuem para o valor, mas o gradiente flui só pelo ramo de predição.
    """

    def __init__(self, sim_w: float, std_w: float, cov_w: float,
                 gamma: float = 1.0, eps: float = 1e-4):
        super().__init__()
        self.sim_w, self.std_w, self.cov_w = sim_w, std_w, cov_w
        self.gamma, self.eps = gamma, eps

    def _variance_term(self, z: torch.Tensor) -> torch.Tensor:
        # z: [N, D] → escalar
        std = torch.sqrt(z.var(dim=0, unbiased=True) + self.eps)
        return F.relu(self.gamma - std).mean()

    def _covariance_term(self, z: torch.Tensor) -> torch.Tensor:
        # z: [N, D] → escalar
        N, D = z.shape
        zc = z - z.mean(dim=0, keepdim=True)
        cov = (zc.T @ zc) / (N - 1)                                 # [D, D]
        off_diag_sq = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
        return off_diag_sq / D

    def forward(self, pred: torch.Tensor, tgt: torch.Tensor):
        # pred/tgt: [N_t, D] (tokens-alvo empilhados do lote), fp32 interno
        pred, tgt = pred.float(), tgt.float()
        assert pred.shape[0] >= 2, "VICReg requer ≥ 2 tokens-alvo no lote"
        inv = F.mse_loss(pred, tgt)
        var = self._variance_term(pred) + self._variance_term(tgt)
        cov = self._covariance_term(pred) + self._covariance_term(tgt)
        total = self.sim_w * inv + self.std_w * var + self.cov_w * cov
        return total, inv, var, cov


def sample_target_mask(Bsz: int, L: int, n_blocks: int, min_ratio: float,
                       max_ratio: float, device: torch.device) -> torch.Tensor:
    """Máscara multi-bloco temporal (I-JEPA): True = posição-alvo. [B, L] bool."""
    mask = torch.zeros(Bsz, L, dtype=torch.bool, device=device)
    for b in range(Bsz):
        for _ in range(n_blocks):
            ratio = min_ratio + (max_ratio - min_ratio) * torch.rand(()).item()
            blk = max(1, int(L * ratio))
            start = int(torch.randint(0, L - blk + 1, ()).item())
            mask[b, start:start + blk] = True
        guard = max(1, L // 10)
        if bool(mask[b].all()):
            mask[b, :guard] = False
        if not bool(mask[b].any()):
            mask[b, L // 2:L // 2 + guard] = True
    return mask


class HJEPASSMMoE(nn.Module):
    """
    context_encoder f_θ — treinado por gradiente (vê contexto mascarado)
    target_encoder  f_θ̄ — EMA de f_θ, congelado (no_grad, eval perpétuo)
    predictor       g_φ — prediz embeddings-alvo (posição + ação como condicionamento)
    """

    def __init__(self, cfg: HJepaConfig):
        super().__init__()
        self.cfg = cfg
        self.context_encoder = SequenceEncoder(cfg)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.predictor = LatentPredictor(cfg)
        self.vicreg = VICRegLoss(cfg.vic_sim, cfg.vic_std, cfg.vic_cov,
                                 cfg.vic_gamma, cfg.vic_eps)

    def train(self, mode: bool = True):
        # o ramo-alvo é SEMPRE determinístico: sem ruído de roteador no EMA
        super().train(mode)
        self.target_encoder.eval()
        return self

    @torch.no_grad()
    def update_target_encoder(self, momentum: float) -> None:
        """θ̄ ← m·θ̄ + (1−m)·θ — média móvel exponencial dos pesos do contexto."""
        for p_t, p_c in zip(self.target_encoder.parameters(),
                            self.context_encoder.parameters()):
            p_t.mul_(momentum).add_(p_c.detach(), alpha=1.0 - momentum)
        for b_t, b_c in zip(self.target_encoder.buffers(),
                            self.context_encoder.buffers()):
            b_t.copy_(b_c)

    def forward(self, x: torch.Tensor, action: torch.Tensor | None = None) -> dict:
        # x: [B, L, D_in]; action: [B, A]
        Bsz, L, _ = x.shape
        tmask = sample_target_mask(Bsz, L, self.cfg.mask_n_blocks,
                                   self.cfg.mask_min_ratio, self.cfg.mask_max_ratio,
                                   x.device)                        # [B, L]

        z_ctx, aux_ctx = self.context_encoder(x, mask=tmask)        # [B, L, D]

        with torch.no_grad():
            z_tgt, _ = self.target_encoder(x, mask=None)            # [B, L, D]
            z_tgt = z_tgt.detach()

        z_pred, aux_pred = self.predictor(z_ctx, tmask, action)     # [B, L, D]

        pred_t = z_pred[tmask]                                      # [N_t, D]
        tgt_t = z_tgt[tmask]                                        # [N_t, D]
        vic_total, inv, var, cov = self.vicreg(pred_t, tgt_t)

        moe_aux = aux_ctx + aux_pred
        loss = vic_total + moe_aux
        return dict(loss=loss, vic_total=vic_total, inv=inv, var=var, cov=cov,
                    moe_aux=moe_aux, n_targets=int(tmask.sum()))
