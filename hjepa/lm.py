# -*- coding: utf-8 -*-
"""
Mini modelo de linguagem causal (nível de caractere) sobre o MESMO backbone híbrido
Mamba/SWA/MoE — o demo "visível" da arquitetura.

Nota de projeto: a filosofia JEPA evita decodificação auto-regressiva; este módulo
existe para DEMONSTRAR o backbone temporal gerando texto. A geração usa o caminho
recorrente dos mixers (init_cache/step): custo O(1) por token e memória constante —
a vantagem estrutural do SSM sobre o Transformer (que paga O(L) por token no KV-cache
completo). A equivalência passo-a-passo ≡ paralelo é verificada na suíte de testes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import HybridStack
from .config import LMConfig
from .layers import RMSNorm


class HybridCharLM(nn.Module):
    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)      # [V] → [D]
        nn.init.normal_(self.embed.weight, std=0.02)
        self.stack = HybridStack(cfg, cfg.d_model, cfg.depth)
        self.norm_f = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight                        # amarração de pesos

    def forward(self, idx: torch.Tensor):
        # idx: [B, L] int64 → (logits: [B, L, V], aux: escalar fp32)
        h = self.embed(idx)                                         # [B, L, D]
        h, aux = self.stack(h)
        return self.head(self.norm_f(h)), aux

    def loss(self, idx: torch.Tensor, targets: torch.Tensor):
        # idx/targets: [B, L] — next-char: targets[t] = idx[t+1] (deslocado no caller)
        logits, aux = self.forward(idx)                             # [B, L, V]
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return ce + aux, ce, aux

    def _step(self, token: torch.Tensor, caches: list[dict]) -> torch.Tensor:
        # token: [B] int64 → logits [B, V]  (um passo recorrente, O(1) no comprimento)
        x = self.embed(token)                                       # [B, D]
        x = self.stack.step(x, caches)
        return self.head(self.norm_f(x))

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 0.8, top_k: int = 40) -> torch.Tensor:
        """idx: [B, T0] prompt → [B, T0 + max_new_tokens]. Inferência recorrente."""
        was_training = self.training
        self.eval()
        caches = self.stack.init_cache(idx.shape[0], idx.device)

        logits = None
        for t in range(idx.shape[1]):                               # prefill do prompt
            logits = self._step(idx[:, t], caches)

        out = [idx]
        for _ in range(max_new_tokens):
            lg = logits / max(1e-5, temperature)                    # [B, V]
            if top_k:
                kth = torch.topk(lg, min(top_k, lg.shape[-1]), dim=-1).values[..., -1:]
                lg = lg.masked_fill(lg < kth, float("-inf"))
            probs = torch.softmax(lg, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)           # [B, 1]
            out.append(nxt)
            logits = self._step(nxt.squeeze(1), caches)

        if was_training:
            self.train()
        return torch.cat(out, dim=1)
