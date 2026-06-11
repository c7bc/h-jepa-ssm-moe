# -*- coding: utf-8 -*-
"""
Converse com o mini-modelo NO SEU TERMINAL — geração em streaming, O(1) por token.

    .venv/bin/python interactive.py

Importante calibrar a expectativa: isto é um COMPLETADOR de texto de 1,59M de
parâmetros treinado num único romance de 1881 — ele não responde perguntas, ele
CONTINUA o que você escrever, no estilo (e na ortografia!) do Machado de Assis.
Escreva o começo de uma frase e veja onde ele leva.

Comandos:  /temp 0.9   /tokens 300   /sair
"""

from __future__ import annotations

import os
import sys
import time

import torch

from hjepa.config import LMConfig
from hjepa.data import CharVocab
from hjepa.lm import HybridCharLM

CKPT_LOCAL = "out/lm_machado.pt"
CKPT_HF = ("dnnxzz/h-jepa-ssm-moe", "lm_machado.pt")


def load_checkpoint() -> str:
    """Usa o checkpoint local; se não existir, baixa do Hugging Face (repo público)."""
    if os.path.exists(CKPT_LOCAL):
        return CKPT_LOCAL
    print(f"[setup] checkpoint local não encontrado; baixando de "
          f"huggingface.co/{CKPT_HF[0]} …")
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=CKPT_HF[0], filename=CKPT_HF[1])
    return path


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(load_checkpoint(), map_location=device, weights_only=True)
    cfg = LMConfig(**ckpt["config"])
    model = HybridCharLM(cfg).to(device).eval()
    model.load_state_dict(ckpt["model"])
    vocab = CharVocab.__new__(CharVocab)
    vocab.itos = ckpt["itos"]
    vocab.stoi = {ch: i for i, ch in enumerate(vocab.itos)}

    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    temperature, n_tokens = 0.8, 250

    print("┌" + "─" * 74)
    print(f"│ H-JEPA-SSM-MoE · mini-LM {n_par:.2f}M params (Mamba-2 ×9 + atenção local ×1 + MoE)")
    print("│ Treinado em 'Memórias Póstumas de Brás Cubas' (1881) — ele COMPLETA texto,")
    print("│ não conversa. Escreva o começo de uma frase e veja a continuação.")
    print(f"│ Comandos: /temp X (atual {temperature}) · /tokens N (atual {n_tokens}) · /sair")
    print("└" + "─" * 74)

    while True:
        try:
            prompt = input("\nvocê>  ")
        except (EOFError, KeyboardInterrupt):
            print("\naté mais!")
            break

        cmd = prompt.strip()
        if cmd in ("/sair", "/quit", "/exit"):
            print("até mais!")
            break
        if cmd.startswith("/temp"):
            try:
                temperature = float(cmd.split()[1])
                print(f"[ok] temperatura = {temperature}")
            except (IndexError, ValueError):
                print("[uso] /temp 0.9")
            continue
        if cmd.startswith("/tokens"):
            try:
                n_tokens = max(1, int(cmd.split()[1]))
                print(f"[ok] tokens = {n_tokens}")
            except (IndexError, ValueError):
                print("[uso] /tokens 300")
            continue
        if not cmd:
            continue

        idx = vocab.encode(prompt)
        if idx.numel() == 0:
            print("[aviso] nenhum caractere do prompt existe no vocabulário do corpus.")
            continue
        dropped = len(prompt) - idx.numel()
        if dropped:
            print(f"[aviso] {dropped} caractere(s) fora do vocabulário de 1881 foram ignorados.")

        sys.stdout.write(f"\nmodelo> {vocab.decode(idx)}")
        sys.stdout.flush()
        t0 = time.perf_counter()
        count = 0
        try:
            for tok in model.generate_stream(idx.unsqueeze(0).to(device), n_tokens,
                                             temperature=temperature):
                sys.stdout.write(vocab.itos[tok])
                sys.stdout.flush()
                count += 1
        except KeyboardInterrupt:
            pass                                         # Ctrl+C corta a geração
        dt = time.perf_counter() - t0
        print(f"\n[{count} chars em {dt:.1f}s ≈ {count / max(dt, 1e-9):.0f} chars/s, "
              f"O(1)/token]")


if __name__ == "__main__":
    main()
