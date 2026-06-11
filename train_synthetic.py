# -*- coding: utf-8 -*-
"""
Treino auto-supervisionado H-JEPA-SSM-MoE em dados sintéticos (sem rótulos).

Uso:
    python train_synthetic.py --preset tiny --steps 30      # CPU / desenvolvimento
    python train_synthetic.py --preset full --steps 1000    # GPU HPC (AMP automático)
"""

from __future__ import annotations

import argparse

import torch

from hjepa.config import HJepaConfig, PRESETS
from hjepa.train import train_jepa


def main() -> None:
    parser = argparse.ArgumentParser(description="H-JEPA-SSM-MoE — treino sintético")
    parser.add_argument("--preset", choices=("auto", "tiny", "full"), default="auto")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--pred-depth", type=int, default=None)
    parser.add_argument("--ssm-version", type=int, choices=(1, 2), default=2,
                        help="2 = Mamba-2/SSD (padrão); 1 = Mamba-1/S6")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cuda_ok = torch.cuda.is_available()
    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if cuda_ok else "cpu"))
    if device.type == "cuda" and not cuda_ok:
        raise SystemExit("CUDA solicitada mas indisponível neste host.")

    preset = args.preset if args.preset != "auto" else ("full" if device.type == "cuda" else "tiny")
    overrides = dict(PRESETS[preset])
    for key, val in (("seq_len", args.seq_len), ("d_model", args.d_model),
                     ("depth", args.depth), ("pred_depth", args.pred_depth)):
        if val is not None:
            overrides[key] = val
    overrides["ssm_version"] = args.ssm_version
    cfg = HJepaConfig(**overrides)

    steps = args.steps if args.steps is not None else (300 if preset == "full" else 30)
    batch_size = args.batch_size if args.batch_size is not None else (8 if preset == "full" else 4)

    swa_layers = [i + 1 for i in range(cfg.depth) if (i + 1) % cfg.attn_every == 0]
    print(f"[config] preset={preset} | L={cfg.seq_len} | D={cfg.d_model} | "
          f"depth={cfg.depth} (SWA nas camadas {swa_layers}) | N={cfg.d_state} | "
          f"E={cfg.n_experts} top-{cfg.top_k} | B={batch_size} | steps={steps}")
    train_jepa(cfg, steps=steps, batch_size=batch_size, device=device,
               use_amp=not args.no_amp, num_workers=args.workers,
               log_every=args.log_every, seed=args.seed)


if __name__ == "__main__":
    main()
