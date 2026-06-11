# -*- coding: utf-8 -*-
"""
H-JEPA-SSM-MoE — Joint Embedding Predictive Architecture com backbone híbrido
Mamba/Atenção-Local (Samba, 9:1) e Mistura de Especialistas esparsa (MegaBlocks).

Módulos:
    config    — dataclasses de configuração (backbone, H-JEPA, mini-LM) e presets
    scan      — varredura seletiva: blocos paralelos + backward adjunto + kernel Triton
    layers    — RMSNorm, RoPE, atenção local de janela, bloco Mamba (S6)
    moe       — roteador Noisy Top-k + despacho dropless por lista de arestas
    backbone  — blocos híbridos e pilha 9 SSM : 1 SWA (forward paralelo + step O(1))
    jepa      — encoders contexto/alvo-EMA, preditor latente, VICReg
    lm        — mini modelo de linguagem causal (demo) sobre o mesmo backbone
    data      — dataset sintético e corpus de texto público
    train     — otimizador, schedules, loop de treino do H-JEPA
"""

from .config import HJepaConfig, LMConfig, PRESETS
from .jepa import HJEPASSMMoE, VICRegLoss
from .lm import HybridCharLM
from .scan import selective_scan, selective_scan_reference

__all__ = [
    "HJepaConfig", "LMConfig", "PRESETS",
    "HJEPASSMMoE", "VICRegLoss", "HybridCharLM",
    "selective_scan", "selective_scan_reference",
]
