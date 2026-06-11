# -*- coding: utf-8 -*-
"""
Varredura seletiva (S6) — núcleo numérico da arquitetura.

Matemática (discretização Zero-Order Hold exata, A diagonal real negativa):
    Ā[b,l,d,n] = exp(Δ[b,l,d] · A[d,n])
    B̄[b,l,d,n] = expm1(Δ[b,l,d] · A[d,n]) / A[d,n] · B[b,l,n]
    h_t = Ā_t ⊙ h_{t−1} + B̄_t · u_t          (recorrência linear de 1ª ordem)
    y_t = Σ_n C_t[n] · h_t[:, :, n] + D · u_t

Três realizações da MESMA matemática, todas verificadas entre si na suíte de testes:

  1. `_chunked_linear_scan` — varredura paralela EM BLOCOS (decomposição da recorrência
     em chunks: matriz de decaimento triangular dentro do bloco + carry entre blocos).
     É a formulação usada em TREINO e inferência CPU/GPU, sem laço por passo de tempo.
  2. `_SelectiveScanFn` — torch.autograd.Function com BACKWARD ADJUNTO ANALÍTICO:
     o gradiente da recorrência é OUTRA recorrência linear (reversa no tempo),
        λ_t = C_t·g_t + Ā_{t+1} ⊙ λ_{t+1},
     resolvida pela mesma rotina de blocos sobre a sequência invertida. Treina pelos
     mesmos caminhos rápidos do forward — nada de autograd através de laço Python.
  3. `_selective_scan_fwd_kernel` — kernel Triton fundido (HBM→SRAM, tl.associative_scan)
     para inferência CUDA; e `selective_scan_reference` — recorrência explícita passo a
     passo, mantida como ORÁCULO de teste e documentação executável.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover - ambiente sem triton
    triton = None
    tl = None
    HAS_TRITON = False

_SMALL = 1e-6        # limiar do ramo-limite Δ·a → 0 (onde B̄/B → Δ, a forma de Euler)
_CHUNK = 16          # tamanho do bloco da varredura paralela em PyTorch


# ═════════════════════════════════════════════════════════════════════════════════════
# 1) Varredura paralela em blocos (forward e backward usam esta rotina)
# ═════════════════════════════════════════════════════════════════════════════════════


def _chunked_linear_scan(dA: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Resolve h_t = exp(dA_t) ⊙ h_{t−1} + b_t   sem laço por passo de tempo.

    dA, b: [B, L, D, N] fp32 (dA = log do fator de decaimento; ≤ 0 nos usos reais).
    Retorna h: [B, L, D, N].

    Dentro de cada bloco de comprimento c (forma fechada da recorrência):
        cum_t = Σ_{i≤t} dA_i
        h_t   = Σ_{s≤t} exp(cum_t − cum_s) · b_s  +  exp(cum_t) · h_entrada
    Para s ≤ t o expoente cum_t − cum_s = Σ_{s<i≤t} dA_i ≤ 0 ⇒ exp ∈ (0,1]: estável.
    Posições s > t recebem −inf ANTES do exp (exp(−inf) = 0; evita inf·0 = NaN).
    Entre blocos, o estado final propaga como carry — O(L/c) iterações de Python,
    todo o trabalho pesado em GEMMs batched.
    """
    Bsz, L, Dm, N = dA.shape
    h_carry = dA.new_zeros(Bsz, Dm, N)                          # [B, D, N]
    outs = []
    for s0 in range(0, L, _CHUNK):
        seg_dA = dA[:, s0:s0 + _CHUNK]                          # [B, c, D, N]
        seg_b = b[:, s0:s0 + _CHUNK]                            # [B, c, D, N]
        c = seg_dA.shape[1]
        cum = torch.cumsum(seg_dA, dim=1)                       # [B, c, D, N]
        # E[b,t,s,d,n] = cum_t − cum_s  (decaimento acumulado de s+1 até t)
        E = cum.unsqueeze(2) - cum.unsqueeze(1)                 # [B, c, c, D, N]
        tril = torch.ones(c, c, dtype=torch.bool, device=dA.device).tril()
        W = E.masked_fill(~tril.view(1, c, c, 1, 1), float("-inf")).exp()
        h_seg = torch.einsum("btsdn,bsdn->btdn", W, seg_b)      # contribuição intra-bloco
        h_seg = h_seg + torch.exp(cum) * h_carry.unsqueeze(1)   # + carry dos blocos prévios
        h_carry = h_seg[:, -1]                                  # [B, D, N]
        outs.append(h_seg)
    return torch.cat(outs, dim=1)                               # [B, L, D, N]


def _discretize_fp32(u32, dt32, A32, Bm32):
    """ZOH exata. Retorna (dA, small, r, bx) — todos [B, L, D, N] fp32."""
    dA = dt32.unsqueeze(-1) * A32                               # [B,L,D,1]·[D,N]
    safe_A = torch.where(A32 == 0, torch.ones_like(A32), A32)
    small = dA.abs() < _SMALL
    r = torch.where(small, dt32.unsqueeze(-1).expand_as(dA),    # limite: B̄/B → Δ
                    torch.expm1(dA) / safe_A)                   # exato: expm1(Δ·a)/a
    bx = r * Bm32.unsqueeze(2) * u32.unsqueeze(-1)              # B̄ · u
    return dA, small, r, bx


def _scan_forward_fp32(u32, dt32, A32, Bm32, Cm32, Dp32):
    """Forward completo em fp32 via varredura em blocos. Retorna (y, h)."""
    dA, _, _, bx = _discretize_fp32(u32, dt32, A32, Bm32)
    h = _chunked_linear_scan(dA, bx)                            # [B, L, D, N]
    y = torch.einsum("bldn,bln->bld", h, Cm32) + u32 * Dp32     # [B, L, D]
    return y, h


# ═════════════════════════════════════════════════════════════════════════════════════
# 2) autograd.Function com backward adjunto analítico
# ═════════════════════════════════════════════════════════════════════════════════════


class _SelectiveScanFn(torch.autograd.Function):
    """
    Backward derivado à mão (método adjunto). Dado g_t = ∂L/∂y_t:

      λ_t ≡ ∂L/∂h_t = C_t·g_t + Ā_{t+1} ⊙ λ_{t+1}      ← recorrência REVERSA: é o mesmo
                                                           scan linear sobre a sequência
                                                           invertida (multiplicador
                                                           deslocado em 1 passo).
      ∂L/∂Ā_t = λ_t ⊙ h_{t−1}            ∂L/∂B̄u_t = λ_t
      ∂L/∂Δ   = Σ_n [∂L/∂(Δ·a)]·a  (+ ramo-limite onde B̄/B = Δ: ∂r/∂Δ = 1)
      ∂L/∂A   = Σ_{b,l} [∂L/∂(Δ·a)]·Δ + Σ gr·(−expm1(Δ·a)/a²)   (dependência explícita 1/a)
      ∂L/∂B_t = Σ_d λ_t·r·u_t            ∂L/∂C_t = Σ_d g_t·h_t
      ∂L/∂u_t = Σ_n λ_t·r·B_t + g_t·D    ∂L/∂D   = Σ_{b,l} g_t·u_t
      onde ∂L/∂(Δ·a) = λ_t·h_{t−1}·Ā + gr·Ā/a (ramo suave),  gr ≡ λ_t·B_t·u_t = ∂L/∂r.

    Memória: salva apenas as entradas (fp32) e os estados h — Ā, r e B̄u são
    recomputados no backward (recomputação elemento a elemento, baratíssima).
    Equivalência com autograd puro é verificada numericamente na suíte de testes.
    """

    @staticmethod
    def forward(ctx, u, dt, A, Bm, Cm, Dp):
        u32, dt32, A32 = u.float(), dt.float(), A.float()
        Bm32, Cm32, Dp32 = Bm.float(), Cm.float(), Dp.float()
        y32, h = _scan_forward_fp32(u32, dt32, A32, Bm32, Cm32, Dp32)
        ctx.save_for_backward(u32, dt32, A32, Bm32, Cm32, Dp32, h)
        ctx.in_dtypes = (u.dtype, dt.dtype, A.dtype, Bm.dtype, Cm.dtype, Dp.dtype)
        return y32.to(u.dtype)

    @staticmethod
    def backward(ctx, grad_y):
        u, dt, A, Bm, Cm, Dp, h = ctx.saved_tensors             # tudo fp32
        g = grad_y.float()                                      # [B, L, D]

        # recomputação barata das quantidades da discretização
        dA = dt.unsqueeze(-1) * A                               # [B, L, D, N]
        a_bar = torch.exp(dA)
        safe_A = torch.where(A == 0, torch.ones_like(A), A)
        small = dA.abs() < _SMALL
        r = torch.where(small, dt.unsqueeze(-1).expand_as(dA), (a_bar - 1.0) / safe_A)

        # ---- estado adjunto λ via scan REVERSO (mesma rotina de blocos) ----
        G = torch.einsum("bld,bln->bldn", g, Cm)                # ∂L/∂h direto  [B,L,D,N]
        dA_rev = torch.flip(dA, dims=[1])
        # multiplicador do scan invertido: ā'_k = rev(ā)_{k−1};  ā'_0 = exp(0) = 1
        dA_shift = F.pad(dA_rev[:, :-1], (0, 0, 0, 0, 1, 0))
        lam = torch.flip(
            _chunked_linear_scan(dA_shift, torch.flip(G, dims=[1])), dims=[1]
        )                                                       # λ  [B, L, D, N]

        h_prev = F.pad(h[:, :-1], (0, 0, 0, 0, 1, 0))           # h_{t−1} (h_{−1} = 0)
        g_abar = lam * h_prev                                   # ∂L/∂Ā
        gr = lam * Bm.unsqueeze(2) * u.unsqueeze(-1)            # ∂L/∂r

        zeros = torch.zeros((), dtype=gr.dtype, device=gr.device)
        g_dA = g_abar * a_bar + torch.where(small, zeros, gr * a_bar / safe_A)

        grad_dt = (g_dA * A).sum(-1) + (gr * small).sum(-1)     # [B, L, D]
        grad_A = (g_dA * dt.unsqueeze(-1)).sum(dim=(0, 1)) + torch.where(
            small, zeros, gr * (-(a_bar - 1.0) / safe_A.pow(2))
        ).sum(dim=(0, 1))                                       # [D, N]
        grad_B = (lam * r * u.unsqueeze(-1)).sum(2)             # [B, L, N]
        grad_u = (lam * r * Bm.unsqueeze(2)).sum(-1) + g * Dp   # [B, L, D]
        grad_C = torch.einsum("bld,bldn->bln", g, h)            # [B, L, N]
        grad_D = (g * u).sum(dim=(0, 1))                        # [D]

        dts = ctx.in_dtypes
        grads = (grad_u.to(dts[0]), grad_dt.to(dts[1]), grad_A.to(dts[2]),
                 grad_B.to(dts[3]), grad_C.to(dts[4]), grad_D.to(dts[5]))
        return tuple(gr_ if need else None
                     for gr_, need in zip(grads, ctx.needs_input_grad))


# ═════════════════════════════════════════════════════════════════════════════════════
# 3) Kernel Triton fundido (inferência CUDA) + oráculo de referência
# ═════════════════════════════════════════════════════════════════════════════════════

if HAS_TRITON:

    @triton.jit
    def _first_order_combine(a_l, b_l, a_r, b_r):
        """Monóide da recorrência h ← a·h + b: (a₁,b₁)∘(a₂,b₂) = (a₂a₁, a₂b₁ + b₂)."""
        return a_r * a_l, a_r * b_l + b_r

    @triton.jit
    def _selective_scan_fwd_kernel(
        u_ptr, dt_ptr, A_ptr, Bm_ptr, Cm_ptr, Dp_ptr, y_ptr,
        L, Dm, N,
        s_ub, s_ul, s_ud,
        s_tb, s_tl, s_td,
        s_ad, s_an,
        s_bb, s_bl, s_bn,
        s_cb, s_cl, s_cn,
        s_yb, s_yl, s_yd,
        BLOCK_L: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        """
        Kernel fundido: discretização ZOH + varredura associativa paralela + contração
        de C em uma única passagem HBM→SRAM. Programa (pid_b, pid_d) é dono do lote
        pid_b e de BLOCK_D canais; o tempo anda em chunks de BLOCK_L com
        tl.associative_scan intra-chunk e carry em registradores entre chunks.
        """
        pid_b = tl.program_id(0)
        pid_d = tl.program_id(1)

        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        offs_n = tl.arange(0, BLOCK_N)
        m_d = offs_d < Dm
        m_n = offs_n < N
        m_dn = m_d[:, None] & m_n[None, :]

        a = tl.load(A_ptr + offs_d[:, None] * s_ad + offs_n[None, :] * s_an,
                    mask=m_dn, other=-1.0).to(tl.float32)               # [BD, BN]
        d_skip = tl.load(Dp_ptr + offs_d, mask=m_d, other=0.0).to(tl.float32)

        h = tl.zeros((BLOCK_D, BLOCK_N), dtype=tl.float32)
        last_row = tl.arange(0, BLOCK_L) == (BLOCK_L - 1)

        for start in range(0, L, BLOCK_L):
            offs_l = start + tl.arange(0, BLOCK_L)
            m_l = offs_l < L
            m_ld = m_l[:, None] & m_d[None, :]
            m_ln = m_l[:, None] & m_n[None, :]

            dt_c = tl.load(dt_ptr + pid_b * s_tb + offs_l[:, None] * s_tl
                           + offs_d[None, :] * s_td, mask=m_ld, other=0.0).to(tl.float32)
            u_c = tl.load(u_ptr + pid_b * s_ub + offs_l[:, None] * s_ul
                          + offs_d[None, :] * s_ud, mask=m_ld, other=0.0).to(tl.float32)
            b_c = tl.load(Bm_ptr + pid_b * s_bb + offs_l[:, None] * s_bl
                          + offs_n[None, :] * s_bn, mask=m_ln, other=0.0).to(tl.float32)
            c_c = tl.load(Cm_ptr + pid_b * s_cb + offs_l[:, None] * s_cl
                          + offs_n[None, :] * s_cn, mask=m_ln, other=0.0).to(tl.float32)

            dA = dt_c[:, :, None] * a[None, :, :]                       # [BL, BD, BN]
            a_bar = tl.exp(dA)
            denom = tl.where(a == 0.0, 1.0, a)[None, :, :]
            ratio = tl.where(tl.abs(dA) < 1e-6, dt_c[:, :, None],
                             (a_bar - 1.0) / denom)
            bx = ratio * b_c[:, None, :] * u_c[:, :, None]

            a_bar = tl.where(m_l[:, None, None], a_bar, 1.0)            # pad = identidade
            bx = tl.where(m_l[:, None, None], bx, 0.0)

            a_cum, h_scan = tl.associative_scan(
                (a_bar, bx), axis=0, combine_fn=_first_order_combine
            )
            h_all = h_scan + a_cum * h[None, :, :]

            y_c = tl.sum(h_all * c_c[:, None, :], axis=2) + u_c * d_skip[None, :]
            tl.store(y_ptr + pid_b * s_yb + offs_l[:, None] * s_yl
                     + offs_d[None, :] * s_yd, y_c, mask=m_ld)

            h = tl.sum(tl.where(last_row[:, None, None], h_all, 0.0), axis=0)


_TRITON_STATE = {"ok": None}


def selective_scan_triton(u, dt, A, Bm, Cm, Dp, BLOCK_L: int = 32, BLOCK_D: int = 4):
    """Lançamento do kernel fundido (saída fp32; o caller faz o cast)."""
    Bsz, L, Di = u.shape
    N = A.shape[1]
    u, dt, Bm, Cm = u.contiguous(), dt.contiguous(), Bm.contiguous(), Cm.contiguous()
    A32, Dp32 = A.float().contiguous(), Dp.float().contiguous()
    y = torch.empty(Bsz, L, Di, device=u.device, dtype=torch.float32)
    grid = (Bsz, triton.cdiv(Di, BLOCK_D))
    _selective_scan_fwd_kernel[grid](
        u, dt, A32, Bm, Cm, Dp32, y,
        L, Di, N,
        *u.stride(), *dt.stride(), *A32.stride(), *Bm.stride(), *Cm.stride(), *y.stride(),
        BLOCK_L=BLOCK_L, BLOCK_D=BLOCK_D, BLOCK_N=triton.next_power_of_2(N),
        num_warps=2,
    )
    return y


def selective_scan_reference(u, dt, A, Bm, Cm, Dp) -> torch.Tensor:
    """
    ORÁCULO: recorrência explícita passo a passo (diferenciável via autograd).
    Não é usado em produção — serve de fonte de verdade nos testes e como
    documentação executável da matemática.
    """
    in_dtype = u.dtype
    u32, dt32, Bm32, Cm32 = u.float(), dt.float(), Bm.float(), Cm.float()
    A32, Dp32 = A.float(), Dp.float()
    dA, _, _, bx = _discretize_fp32(u32, dt32, A32, Bm32)
    A_bar = torch.exp(dA)
    Bsz, L, Di, N = dA.shape
    h = u32.new_zeros(Bsz, Di, N)
    ys = []
    for t in range(L):                                          # laço temporal explícito
        h = A_bar[:, t] * h + bx[:, t]                          # h_t = Ā_t·h_{t−1} + B̄_t·u_t
        ys.append(torch.einsum("bdn,bn->bd", h, Cm32[:, t]))    # y_t = ⟨C_t, h_t⟩
    y = torch.stack(ys, dim=1) + u32 * Dp32
    return y.to(in_dtype)


# ═════════════════════════════════════════════════════════════════════════════════════
# 4) Despachante de produção
# ═════════════════════════════════════════════════════════════════════════════════════


def selective_scan(u, dt, A, Bm, Cm, Dp) -> torch.Tensor:
    """
    • COM gradiente (treino, CPU ou GPU): _SelectiveScanFn — varredura paralela em
      blocos no forward + backward adjunto analítico (sem autograd através de laços).
    • SEM gradiente em CUDA: kernel Triton fundido; falha de runtime ⇒ fallback
      memoizado com aviso único.
    • SEM gradiente em CPU: varredura paralela em blocos, fp32.
    """
    need_grad = torch.is_grad_enabled() and any(
        t.requires_grad for t in (u, dt, A, Bm, Cm, Dp)
    )
    if need_grad:
        return _SelectiveScanFn.apply(u, dt, A, Bm, Cm, Dp)

    if HAS_TRITON and u.is_cuda and _TRITON_STATE["ok"] is not False:
        try:
            y = selective_scan_triton(u, dt, A, Bm, Cm, Dp)
            _TRITON_STATE["ok"] = True
            return y.to(u.dtype)
        except Exception as exc:  # pragma: no cover
            _TRITON_STATE["ok"] = False
            warnings.warn(f"Kernel Triton indisponível ({exc!r}); usando varredura PyTorch.")

    y32, _ = _scan_forward_fp32(u.float(), dt.float(), A.float(),
                                Bm.float(), Cm.float(), Dp.float())
    return y32.to(u.dtype)
