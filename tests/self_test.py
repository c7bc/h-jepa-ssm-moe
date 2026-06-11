# -*- coding: utf-8 -*-
"""
Suíte de autoverificação — cada subsistema é comparado a um oráculo independente.

    python tests/self_test.py
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from hjepa.config import HJepaConfig, LMConfig, PRESETS
from hjepa.jepa import HJEPASSMMoE, VICRegLoss
from hjepa.layers import LocalWindowAttention, apply_rope
from hjepa.lm import HybridCharLM
from hjepa.mamba2 import ssd_chunked
from hjepa.moe import SparseMoE
from hjepa.scan import (HAS_TRITON, _SelectiveScanFn, selective_scan,
                        selective_scan_reference, selective_scan_triton)


# ─────────────────────────────────────────────────────────────────────────────────────
# 1) Varredura seletiva: forward (blocos paralelos) vs oráculo escalar
# ─────────────────────────────────────────────────────────────────────────────────────


def _scan_naive_loops(u, dt, A, Bm, Cm, Dp) -> torch.Tensor:
    """Oráculo em laços escalares puros da MESMA matemática ZOH."""
    Bsz, L, Di = u.shape
    N = A.shape[1]
    y = torch.zeros(Bsz, L, Di)
    for b in range(Bsz):
        h = [[0.0] * N for _ in range(Di)]
        for t in range(L):
            for d in range(Di):
                acc = 0.0
                for n in range(N):
                    a = float(A[d, n])
                    da = float(dt[b, t, d]) * a
                    a_bar = math.exp(da)
                    ratio = float(dt[b, t, d]) if abs(da) < 1e-6 else math.expm1(da) / a
                    h[d][n] = a_bar * h[d][n] + ratio * float(Bm[b, t, n]) * float(u[b, t, d])
                    acc += float(Cm[b, t, n]) * h[d][n]
                y[b, t, d] = acc + float(Dp[d]) * float(u[b, t, d])
    return y


def _random_scan_inputs(Bsz=2, L=24, Di=5, N=4, seed=0, requires_grad=False):
    torch.manual_seed(seed)
    u = torch.randn(Bsz, L, Di, requires_grad=requires_grad)
    dt = F.softplus(torch.randn(Bsz, L, Di)).detach().requires_grad_(requires_grad)
    A = -torch.exp(torch.randn(Di, N))
    A[0, 0] = -1e-12                                  # exercita o ramo-limite Δ·a → 0
    A = A.detach().requires_grad_(requires_grad)
    Bm = torch.randn(Bsz, L, N, requires_grad=requires_grad)
    Cm = torch.randn(Bsz, L, N, requires_grad=requires_grad)
    Dp = torch.randn(Di, requires_grad=requires_grad)
    return u, dt, A, Bm, Cm, Dp


def test_scan_forward_vs_naive() -> None:
    u, dt, A, Bm, Cm, Dp = _random_scan_inputs(L=37)  # L não-múltiplo do chunk (16)
    with torch.no_grad():
        y_par = selective_scan(u, dt, A, Bm, Cm, Dp)              # blocos paralelos
        y_ref = selective_scan_reference(u, dt, A, Bm, Cm, Dp)    # recorrência explícita
    y_naive = _scan_naive_loops(u, dt, A, Bm, Cm, Dp)
    e1 = (y_par - y_naive).abs().max().item()
    e2 = (y_ref - y_naive).abs().max().item()
    assert e1 < 1e-4 and e2 < 1e-5, f"scan ≠ oráculo (par {e1:.2e}, ref {e2:.2e})"
    print(f"[PASS] scan paralelo em blocos ≡ recorrência ≡ oráculo escalar "
          f"(err {e1:.2e} / {e2:.2e})")


def test_scan_backward_vs_autograd() -> None:
    """Backward ADJUNTO analítico vs autograd através da recorrência explícita."""
    ins_a = _random_scan_inputs(L=37, seed=7, requires_grad=True)
    ins_b = tuple(t.detach().clone().requires_grad_(True) for t in ins_a)
    torch.manual_seed(99)
    w = torch.randn(2, 37, 5)                                     # gradiente upstream

    y_a = _SelectiveScanFn.apply(*ins_a)                          # caminho de produção
    (y_a * w).sum().backward()
    y_b = selective_scan_reference(*ins_b)                        # oráculo autograd
    (y_b * w).sum().backward()

    err_y = (y_a - y_b).abs().max().item()
    assert err_y < 1e-4, f"forward divergiu ({err_y:.2e})"
    names = ("u", "dt", "A", "B", "C", "D")
    worst = 0.0
    for name, ta, tb in zip(names, ins_a, ins_b):
        scale = tb.grad.abs().max().clamp_min(1.0)
        err = ((ta.grad - tb.grad).abs().max() / scale).item()
        worst = max(worst, err)
        assert err < 1e-4, f"grad_{name} divergiu (rel err {err:.2e})"
    print(f"[PASS] backward adjunto analítico ≡ autograd (pior grad rel err {worst:.2e})")


# ─────────────────────────────────────────────────────────────────────────────────────
# 1b) Mamba-2 / SSD: algoritmo por blocos vs recorrência escalar-por-cabeça
# ─────────────────────────────────────────────────────────────────────────────────────


def test_ssd_vs_naive_and_grads() -> None:
    """SSD por blocos (forward E gradientes via autograd) vs recorrência explícita."""
    torch.manual_seed(11)
    Bsz, L, H, P, N = 2, 37, 3, 4, 5                  # L não-múltiplo do chunk (32)

    def make_inputs():
        torch.manual_seed(12)
        x = torch.randn(Bsz, L, H, P, requires_grad=True)
        dt = F.softplus(torch.randn(Bsz, L, H)).detach().requires_grad_(True)
        A = (-torch.exp(torch.randn(H))).detach().requires_grad_(True)
        Bm = torch.randn(Bsz, L, N, requires_grad=True)
        Cm = torch.randn(Bsz, L, N, requires_grad=True)
        return x, dt, A, Bm, Cm

    ins_a = make_inputs()
    ins_b = make_inputs()
    torch.manual_seed(13)
    w = torch.randn(Bsz, L, H, P)                     # gradiente upstream

    y_a = ssd_chunked(*ins_a)                         # algoritmo SSD por blocos
    (y_a * w).sum().backward()

    x, dt, A, Bm, Cm = ins_b                          # oráculo: recorrência por passo
    h = torch.zeros(Bsz, H, P, N)
    ys = []
    for t in range(L):
        a_bar = torch.exp(dt[:, t] * A)               # decaimento ESCALAR/cabeça [B, H]
        dtx = dt[:, t].unsqueeze(-1) * x[:, t]        # Δ·x  [B, H, P]
        h = a_bar.unsqueeze(-1).unsqueeze(-1) * h \
            + dtx.unsqueeze(-1) * Bm[:, t].unsqueeze(1).unsqueeze(1)
        ys.append(torch.einsum("bn,bhpn->bhp", Cm[:, t], h))
    y_b = torch.stack(ys, dim=1)                      # [B, L, H, P]
    (y_b * w).sum().backward()

    err_y = (y_a - y_b).abs().max().item()
    assert err_y < 1e-4, f"SSD forward divergiu ({err_y:.2e})"
    worst = 0.0
    for name, ta, tb in zip(("x", "dt", "A", "B", "C"), ins_a, ins_b):
        scale = tb.grad.abs().max().clamp_min(1.0)
        err = ((ta.grad - tb.grad).abs().max() / scale).item()
        worst = max(worst, err)
        assert err < 1e-4, f"SSD grad_{name} divergiu (rel err {err:.2e})"
    print(f"[PASS] Mamba-2/SSD por blocos ≡ recorrência escalar-por-cabeça "
          f"(fwd {err_y:.2e}; pior grad {worst:.2e})")


# ─────────────────────────────────────────────────────────────────────────────────────
# 2) Atenção local em tiles vs atenção densa mascarada
# ─────────────────────────────────────────────────────────────────────────────────────


def test_local_attention_vs_dense() -> None:
    torch.manual_seed(1)
    d_model, H, K, Bsz, L = 64, 4, 16, 2, 50          # L não-múltiplo de K (testa o pad)
    attn = LocalWindowAttention(d_model, H, K)
    x = torch.randn(Bsz, L, d_model)
    y_blocked = attn(x)

    hd = d_model // H
    qkv = attn.qkv(x).view(Bsz, L, 3, H, hd)
    q, k, v = qkv.unbind(dim=2)
    cos, sin = attn.rope(L, x.device)
    q, k, v = apply_rope(q.float(), cos, sin), apply_rope(k.float(), cos, sin), v.float()
    q, k, v = (t.permute(0, 2, 1, 3) for t in (q, k, v))          # [B, H, L, hd]
    scores = q @ k.transpose(-1, -2) / math.sqrt(hd)              # [B, H, L, L]
    dist = torch.arange(L)[:, None] - torch.arange(L)[None, :]
    scores = scores.masked_fill(~((dist >= 0) & (dist < K)), float("-inf"))
    out = torch.softmax(scores, dim=-1) @ v
    y_dense = attn.out_proj(out.permute(0, 2, 1, 3).reshape(Bsz, L, d_model).to(x.dtype))

    err = (y_blocked - y_dense).abs().max().item()
    assert err < 1e-4, f"atenção em blocos ≠ densa mascarada ({err:.2e})"
    print(f"[PASS] atenção local em tiles ≡ atenção densa mascarada (err {err:.2e})")


# ─────────────────────────────────────────────────────────────────────────────────────
# 3) MoE esparso vs referência densa por token
# ─────────────────────────────────────────────────────────────────────────────────────


def test_moe_vs_dense() -> None:
    torch.manual_seed(2)
    d_model, d_ff, E, k, Bsz, L = 32, 64, 4, 2, 2, 16
    moe = SparseMoE(d_model, d_ff, E, k, w_balance=0.01, router_noise=1.0).eval()
    x = torch.randn(Bsz, L, d_model)
    y_sparse, aux = moe(x)

    xf = x.reshape(-1, d_model)
    logits = F.linear(xf, moe.w_gate.weight)                      # eval ⇒ sem ruído
    top_val, top_idx = logits.topk(k, dim=-1)
    gates = torch.softmax(top_val, dim=-1)
    y_ref = torch.zeros_like(xf)
    for t_i in range(xf.shape[0]):
        for j in range(k):
            y_ref[t_i] += gates[t_i, j] * moe.experts[int(top_idx[t_i, j])](xf[t_i])
    err = (y_sparse.reshape(-1, d_model) - y_ref).abs().max().item()
    assert err < 1e-5, f"roteamento esparso ≠ referência densa ({err:.2e})"

    probs = torch.softmax(logits, dim=-1).mean(0)
    f = F.one_hot(top_idx, num_classes=E).sum(dim=(0, 1)).float() / (xf.shape[0] * k)
    aux_ref = 0.01 * E * torch.sum(f * probs)
    assert abs(aux.item() - aux_ref.item()) < 1e-6
    print(f"[PASS] MoE gather/segment-GEMM/index_add ≡ referência densa "
          f"(err {err:.2e}; aux {aux.item():.6f})")


# ─────────────────────────────────────────────────────────────────────────────────────
# 4) VICReg, 5) modelo JEPA fim-a-fim, 6) LM passo-a-passo, 7) Triton
# ─────────────────────────────────────────────────────────────────────────────────────


def test_vicreg_sanity() -> None:
    torch.manual_seed(3)
    vic = VICRegLoss(25.0, 25.0, 1.0)
    z = torch.randn(256, 32)
    _, inv, _, cov = vic(z, z.clone())
    assert inv.item() < 1e-10, "invariância deveria ser 0 para entradas idênticas"
    zc = torch.zeros(256, 32)
    _, _, var_c, _ = vic(zc, zc)
    assert abs(var_c.item() - 2 * vic.gamma) < 0.05, "hinge deveria saturar em 2γ"
    print(f"[PASS] VICReg: inv(z,z)={inv.item():.1e}; var(colapso)={var_c.item():.4f}≈2γ; "
          f"cov(iid)={cov.item():.4f}")


def test_jepa_end_to_end() -> None:
    torch.manual_seed(4)
    cfg = HJepaConfig(**{**PRESETS["tiny"], "seq_len": 48, "depth": 10, "pred_depth": 2})
    model = HJEPASSMMoE(cfg).train()
    x = torch.randn(2, cfg.seq_len, cfg.input_dim)
    a = torch.rand(2, cfg.action_dim)
    out = model(x, a)
    assert math.isfinite(out["loss"].item()), "loss não-finita"
    out["loss"].backward()

    assert all(p.grad is None for p in model.target_encoder.parameters()), \
        "target encoder recebeu gradiente (violação do congelamento JEPA)"
    g_router = model.context_encoder.stack.blocks[0].moe.w_gate.weight.grad
    assert g_router is not None and torch.isfinite(g_router).all() \
        and g_router.abs().sum() > 0, "roteador sem gradiente"
    assert model.predictor.mask_token.grad is not None, "mask token sem gradiente"

    model.update_target_encoder(momentum=0.0)
    for p_t, p_c in zip(model.target_encoder.parameters(),
                        model.context_encoder.parameters()):
        assert torch.equal(p_t, p_c), "EMA com m=0 deveria copiar exatamente"
    snap = model.target_encoder.embed.weight.clone()
    model.update_target_encoder(momentum=1.0)
    assert torch.equal(snap, model.target_encoder.embed.weight), "EMA m=1 deveria congelar"
    print(f"[PASS] JEPA fim-a-fim: loss={out['loss'].item():.4f} finita; alvo sem grad; "
          f"roteador/preditor com grad; EMA correto nos extremos")


def test_lm_stepwise_equivalence() -> None:
    """Geração recorrente O(1)/token ≡ forward paralelo, para Mamba-1 E Mamba-2."""
    for version in (1, 2):
        torch.manual_seed(5)
        cfg = LMConfig(vocab_size=23, d_model=32, depth=3, attn_every=3, n_heads=4,
                       window=8, d_state=4, d_conv=4, ff_mult=2, n_experts=4,
                       top_k=2, ssm_version=version, headdim=32)
        model = HybridCharLM(cfg).eval()              # inclui 1 camada SWA (3ª)
        idx = torch.randint(0, cfg.vocab_size, (2, 20))   # [B, L]

        with torch.no_grad():
            logits_full, _ = model(idx)               # [B, L, V] caminho paralelo
            caches = model.stack.init_cache(2, idx.device)
            logits_step = torch.stack(
                [model._step(idx[:, t], caches) for t in range(idx.shape[1])], dim=1)

        err = (logits_full - logits_step).abs().max().item()
        assert err < 1e-4, f"Mamba-{version}: recorrente ≠ paralelo ({err:.2e})"
        print(f"[PASS] LM Mamba-{version}: inferência recorrente O(1)/token ≡ "
              f"forward paralelo (err {err:.2e})")


def test_triton_vs_reference() -> None:
    if not (HAS_TRITON and torch.cuda.is_available()):
        print("[SKIP] kernel Triton (requer GPU CUDA + triton)")
        return
    torch.manual_seed(6)
    dev = torch.device("cuda")
    Bsz, L, Di, N = 2, 96, 10, 16
    u = torch.randn(Bsz, L, Di, device=dev)
    dt = F.softplus(torch.randn(Bsz, L, Di, device=dev))
    A = -torch.exp(torch.randn(Di, N, device=dev))
    Bm, Cm = torch.randn(Bsz, L, N, device=dev), torch.randn(Bsz, L, N, device=dev)
    Dp = torch.randn(Di, device=dev)
    with torch.no_grad():
        y_tri = selective_scan_triton(u, dt, A, Bm, Cm, Dp)
        y_ref = selective_scan_reference(u, dt, A, Bm, Cm, Dp)
    err = (y_tri - y_ref).abs().max().item()
    assert err < 1e-3, f"kernel Triton ≠ referência ({err:.2e})"
    print(f"[PASS] kernel Triton (varredura fundida) ≡ referência (err {err:.2e})")


if __name__ == "__main__":
    print("── suíte de autoverificação H-JEPA-SSM-MoE ──")
    test_scan_forward_vs_naive()
    test_scan_backward_vs_autograd()
    test_ssd_vs_naive_and_grads()
    test_local_attention_vs_dense()
    test_moe_vs_dense()
    test_vicreg_sanity()
    test_jepa_end_to_end()
    test_lm_stepwise_equivalence()
    test_triton_vs_reference()
    print("── todos os testes passaram ──")
