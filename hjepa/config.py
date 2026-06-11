# -*- coding: utf-8 -*-
"""Configurações da arquitetura H-JEPA-SSM-MoE (backbone compartilhado + variantes)."""

from dataclasses import dataclass


@dataclass
class BackboneConfig:
    """Campos compartilhados pela espinha dorsal híbrida (Mamba 9 : 1 SWA + MoE)."""

    ssm_version: int = 2         # 2 = Mamba-2/SSD (padrão); 1 = Mamba-1/S6
    attn_every: int = 10         # a cada 10 camadas, a 10ª é atenção ⇒ 9 SSM : 1 SWA
    n_heads: int = 8             # H: cabeças da atenção local
    window: int = 64             # K: janela causal local (bloco fixo)
    d_state: int = 16            # N: dimensão do estado do SSM
    d_conv: int = 4              # kernel da convolução causal depthwise
    expand: int = 2              # D_inner = expand · D
    headdim: int = 32            # P: dimensão da cabeça do SSM (só Mamba-2)
    dt_min: float = 1e-3         # faixa de init do passo Δ (log-uniforme)
    dt_max: float = 1e-1
    n_experts: int = 8           # E: pool de especialistas do MoE
    top_k: int = 2               # k: rotas por token
    ff_mult: int = 4             # D_ff = ff_mult · D
    w_balance: float = 0.01      # coeficiente da perda de balanceamento (Switch)
    router_noise: float = 1.0    # multiplicador do ruído gaussiano do roteador


@dataclass
class HJepaConfig(BackboneConfig):
    """Modelo completo H-JEPA (auto-supervisão por predição latente)."""

    # dados
    input_dim: int = 64          # D_in: canais da observação
    action_dim: int = 8          # A: variável de ação/condicionamento
    seq_len: int = 512           # L

    # encoder
    d_model: int = 256           # D
    depth: int = 20

    # preditor latente
    pred_d_model: int = 128      # D_pred < D (gargalo informacional)
    pred_depth: int = 10
    max_seq_len: int = 4096

    # máscara multi-bloco (I-JEPA temporal)
    mask_n_blocks: int = 4
    mask_min_ratio: float = 0.10
    mask_max_ratio: float = 0.25

    # VICReg
    vic_sim: float = 25.0        # λ (invariância)
    vic_std: float = 25.0        # μ (variância)
    vic_cov: float = 1.0         # ν (covariância)
    vic_gamma: float = 1.0       # γ: desvio padrão mínimo
    vic_eps: float = 1e-4

    # EMA do target encoder
    ema_base: float = 0.996

    # otimização
    lr: float = 3e-4
    weight_decay: float = 0.05
    beta1: float = 0.9
    beta2: float = 0.95
    warmup_steps: int = 30
    min_lr_ratio: float = 0.10
    grad_clip: float = 1.0

    def __post_init__(self) -> None:
        assert self.d_model % self.n_heads == 0, "d_model deve ser múltiplo de n_heads"
        assert self.pred_d_model % self.n_heads == 0, "pred_d_model deve ser múltiplo de n_heads"
        assert self.action_dim % 2 == 0, "action_dim deve ser par (freqs ‖ amplitudes)"
        assert self.top_k <= self.n_experts
        assert 0.0 < self.mask_min_ratio <= self.mask_max_ratio < 1.0
        assert self.ssm_version in (1, 2)
        if self.ssm_version == 2:
            assert (self.expand * self.d_model) % self.headdim == 0, \
                "expand·d_model deve ser múltiplo de headdim (Mamba-2)"
            assert (self.expand * self.pred_d_model) % self.headdim == 0, \
                "expand·pred_d_model deve ser múltiplo de headdim (Mamba-2)"


@dataclass
class LMConfig(BackboneConfig):
    """Mini modelo de linguagem causal (demo) reutilizando o MESMO backbone híbrido."""

    vocab_size: int = 128
    d_model: int = 96
    depth: int = 10

    def __post_init__(self) -> None:
        assert self.d_model % self.n_heads == 0, "d_model deve ser múltiplo de n_heads"
        assert self.top_k <= self.n_experts
        assert self.ssm_version in (1, 2)
        if self.ssm_version == 2:
            assert (self.expand * self.d_model) % self.headdim == 0, \
                "expand·d_model deve ser múltiplo de headdim (Mamba-2)"


PRESETS = {
    # GPU HPC: ~105M parâmetros treináveis (+ cópia EMA congelada)
    "full": dict(),
    # CPU / smoke-test: roda em segundos
    "tiny": dict(
        input_dim=32, action_dim=8, seq_len=64,
        d_model=64, depth=10, n_heads=4, window=16, d_state=8,
        ff_mult=2, pred_d_model=32, pred_depth=2,
        mask_n_blocks=3, mask_max_ratio=0.20,
        lr=1e-3, warmup_steps=5,
    ),
}
