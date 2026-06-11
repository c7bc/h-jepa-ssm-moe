# H-JEPA-SSM-MoE: An Oracle-Verified, CPU-Trainable Reference Implementation of a Hybrid Mamba/Attention Backbone with Sparse Mixture-of-Experts and a Latent-Predictive Objective

**dnnxzz** — Independent
*Technical report, June 2026*

Code: https://github.com/c7bc/h-jepa-ssm-moe · Model: https://huggingface.co/dnnxzz/h-jepa-ssm-moe

---

## Abstract

We present a self-contained, pure-PyTorch reference implementation of a post-Transformer
architecture that combines four lines of research: (i) selective state-space sequence
mixing (Mamba-1/S6 with exact zero-order-hold discretization, and Mamba-2/SSD), (ii) a
Samba-style hybrid topology that interleaves nine SSM layers with one sliding-window
attention layer, (iii) dropless sparse Mixture-of-Experts feed-forward blocks with noisy
top-2 routing and a Switch-style load-balancing loss, and (iv) a Joint-Embedding
Predictive Architecture (JEPA) training objective with an EMA target encoder and VICReg
anti-collapse regularization. The distinguishing feature of this implementation is its
**verification methodology**: every numerical subsystem — including a hand-derived
adjoint backward pass for the selective scan — is tested against an independent oracle
implementation, with worst-case discrepancies between 10⁻⁷ and 10⁻⁵. All components
run and train on commodity CPUs. We report two small-scale empirical observations:
(1) on a 16-core CPU, the Mamba-2/SSD block formulation trains **7.3× faster** than an
equivalently-sized Mamba-1/S6 stack (3,550 vs. 484 characters/second) at identical loss
trajectories, confirming that SSD's matmul-friendly structure benefits BLAS-bound CPU
execution and not only GPU tensor cores; and (2) a 1.59M-parameter character-level
instance of the hybrid backbone trained for 18 minutes on a single public-domain
Portuguese novel reaches a validation perplexity of 5.3 per character (train ≈3.1) and
generates locally coherent 19th-century Portuguese, with O(1) per-token recurrent
inference whose throughput does not degrade with sequence length. We release the code, the trained checkpoint, and the
full test suite. This work makes no state-of-the-art claims; its intended contributions
are pedagogical clarity, verifiability, and accessibility on minimal hardware.

## Resumo (Português)

Apresentamos uma implementação de referência, autocontida e em PyTorch puro, de uma
arquitetura pós-Transformer que combina: (i) mistura sequencial por modelos de espaço de
estados seletivos (Mamba-1/S6 com discretização ZOH exata, e Mamba-2/SSD), (ii) topologia
híbrida estilo Samba (9 camadas SSM : 1 camada de atenção local de janela deslizante),
(iii) blocos *feed-forward* de Mistura de Especialistas esparsa *dropless* com roteamento
top-2 ruidoso e perda de balanceamento estilo Switch, e (iv) o objetivo de treino JEPA
(predição em espaço latente) com *encoder*-alvo EMA e regularização anticolapso VICReg.
O diferencial é a **metodologia de verificação**: cada subsistema numérico — incluindo um
*backward* adjunto derivado à mão para o *scan* seletivo — é testado contra um oráculo
independente, com discrepâncias máximas entre 10⁻⁷ e 10⁻⁵. Tudo roda e treina em CPU
comum. Observações empíricas em pequena escala: (1) num CPU de 16 núcleos, o bloco
Mamba-2/SSD treina **7,3× mais rápido** que um Mamba-1/S6 de tamanho equivalente (3.550
vs. 484 caracteres/s) com trajetória de perda idêntica; (2) uma instância de 1,59M de
parâmetros, treinada por 18 minutos em um único romance de domínio público, atinge
perplexidade de validação 5,3 por caractere (treino ≈3,1) e gera português oitocentista
localmente coerente, com inferência recorrente O(1) por token. Não reivindicamos estado-da-arte: as contribuições
pretendidas são clareza pedagógica, verificabilidade e acessibilidade em hardware mínimo.

---

## 1. Introduction

Transformers pay O(L²) attention cost in sequence length and, in language modeling,
are typically trained with token-level generative objectives. Two research programs
attack these limitations from different angles: **selective state-space models** (Mamba
[1], Mamba-2 [2]) replace quadratic attention with a linear-time recurrence whose
parameters are functions of the input; **joint-embedding predictive architectures**
(JEPA [8, 9]) replace token/pixel reconstruction with prediction in representation
space, regularized against collapse. Orthogonally, **hybrid topologies** (Samba [3])
restore exact-recall ability that recurrent compression loses by interleaving local
attention, and **sparse Mixture-of-Experts** (MegaBlocks [4], Switch [5], noisy top-k
[6]) decouple parameter count from per-token compute.

This report documents a reference implementation that composes all four. It is written
for readability and verification rather than peak performance: every tensor shape is
annotated, every mathematical step is explicit, and — unusually for reference code —
**every subsystem ships with an oracle test**, including the analytic gradient of the
selective scan. The full system trains on a laptop CPU without a GPU.

We are explicit about what this work is not. It is not a state-of-the-art system, it
reports no large-scale benchmarks, and its empirical section is limited to what can be
measured honestly on a single CPU box: relative throughput of the two SSM formulations
under identical conditions, and a character-level language-modeling demo on a single
public-domain novel.

## 2. Architecture

The backbone is a pre-norm residual stack of `depth` blocks. Block *i* applies a
temporal **mixer** followed by a sparse **MoE feed-forward**:

```
x ← x + Mixer(RMSNorm(x))        Mixer ∈ {Mamba-2 (default), Mamba-1, SWA}
x ← x + SparseMoE(RMSNorm(x))
```

Every 10th mixer is sliding-window attention (SWA); the other nine are SSM blocks —
the Samba recipe [3] at a 9:1 ratio, with the dense MLP replaced by MoE.

### 2.1 Selective SSM, exact ZOH discretization (Mamba-1 / S6)

The continuous system `h′(t) = A h(t) + B x(t)`, `y(t) = C h(t) + D x(t)` is
discretized per step Δ with a zero-order hold. With diagonal negative `A ∈ ℝ^{D×N}`
and input-dependent `Δ(x) ∈ ℝ^{B×L×D}`, `B(x), C(x) ∈ ℝ^{B×L×N}`:

$$\bar A = e^{\Delta A}, \qquad
\bar B = (\Delta A)^{-1}\left(e^{\Delta A}-I\right)\Delta B
       = A^{-1}\left(e^{\Delta A}-I\right)B
\;\;\xrightarrow{\;\Delta A\to 0\;}\;\Delta B .$$

We implement the **exact** form elementwise, `B̄ = expm1(Δa)/a · B`, with a guarded
branch that returns the Euler limit `ΔB` when `|Δa| < 10⁻⁶` (the official Mamba-1
kernel uses the Euler simplification throughout; the two coincide in that limit). The
recurrence `h_t = Ā_t ⊙ h_{t−1} + B̄_t x_t`, `y_t = ⟨C_t, h_t⟩ + D x_t` is evaluated by
a **chunked parallel scan**: within a chunk of length *c*, with `cs_t = Σ_{i≤t} Δ_i a`,

$$h_t = \sum_{s\le t} e^{\,cs_t - cs_s}\, b_s \;+\; e^{\,cs_t}\, h_{\text{in}},$$

where all exponents are ≤ 0 (hence numerically stable), masked positions receive −∞
*before* exponentiation, and the inter-chunk state is carried sequentially. No
per-timestep Python loop appears in training.

**Analytic adjoint backward.** Autograd through a scan loop is memory- and
overhead-expensive. We instead derive the gradient by the adjoint method. Given
`g_t = ∂L/∂y_t`, the adjoint state obeys a *reversed* linear recurrence,

$$\lambda_t \;=\; C_t\, g_t \;+\; \bar A_{t+1} \odot \lambda_{t+1},$$

which is the same first-order scan run on the time-reversed sequence with a one-step
shifted multiplier — so forward and backward share one chunked-scan routine. The
remaining gradients are elementwise contractions (Sec. A in code, `hjepa/scan.py`):
`∂L/∂Ā_t = λ_t ⊙ h_{t−1}`, `∂L/∂B̄x_t = λ_t`, with the chain through
`Ā = e^{ΔA}` and `B̄/B = expm1(ΔA)/A` handled in both the smooth and guarded branches.
Only the inputs and the state sequence `h` are saved; `Ā, B̄` are recomputed. The
implementation is wrapped in a `torch.autograd.Function` and validated against
autograd-through-the-explicit-recurrence to a worst-case relative error of
**1.9 × 10⁻⁷** over all six input gradients (Sec. 3).

A fused Triton kernel implements the same forward (ZOH + first-order associative scan
via `tl.associative_scan` + output contraction in one HBM→SRAM pass) and is dispatched
automatically for no-grad CUDA paths; on CPU-only hosts all paths use the chunked scan.

### 2.2 Mamba-2 / SSD (default mixer)

Mamba-2 [2] restricts the decay to a **scalar per head**, `a_t = e^{Δ_t A_h}` with
`A_h < 0`, splits channels into H heads of dimension P, and shares `B_t, C_t ∈ ℝ^N`
across heads. The sequence-mixing operator then factorizes into a 1-semiseparable mask
times an outer product:

$$M[t,s] \;=\; \Big(\textstyle\prod_{s<i\le t} a_i\Big)\,\langle C_t, B_s\rangle,
\qquad y = (L \odot CB^\top)\,(\Delta x),$$

so a chunk of the sequence is processed with **dense matmuls** (the "duality" with
masked attention), plus an inter-chunk state of shape `[B, H, P, N]`. Our
`ssd_chunked` routine implements exactly this block decomposition in fp32; because the
graph contains no explicit recurrence, plain autograd differentiates it efficiently and
no custom backward is required. Following [2], the input discretization is `B̄x = Δ·B⊗x`
(this is Mamba-2's definition, not a simplification of it), projections for
`z, x, B, C, Δ` are emitted by a single input projection, the causal depthwise
convolution is applied to the concatenated `[x‖B‖C]` group, and a gated RMSNorm
(`y = RMSNorm(y ⊙ SiLU(z))`) precedes the output projection.

### 2.3 Sliding-window attention (recall layer)

Every 10th mixer is causal local attention with window K, computed in block tiles:
queries of a K-block attend to the concatenated `[previous block ‖ current block]`
(2K keys), masked to `0 ≤ g_q − g_k < K`. The L×L score matrix is never materialized
— the memory access pattern of FlashAttention's sliding window [10, 11] — and softmax
runs in fp32. RoPE [13] is applied only in attention layers; the SSM layers carry
position implicitly through recurrence. Functionally, the SSM compresses history into
a fixed-size state (lossy), while SWA provides exact retrieval over the recent window
("needle-in-a-haystack" recall), as analyzed in Samba [3].

### 2.4 Dropless sparse MoE

Each block's FFN is a pool of E = 8 independent SiLU MLPs. Routing follows noisy
top-k [6] with k = 2: `H(x) = xW_g + ε ⊙ softplus(xW_noise)`, ε ~ 𝒩(0, I) (training
only), with the router computed in fp32. Gates are the softmax over the two selected
logits. Dispatch materializes the token→expert bipartite graph as a COO edge list,
stable-sorts it by expert, and runs **one dense GEMM per contiguous expert segment** —
the dropless grouped-GEMM scheme of MegaBlocks [4]: no capacity factor, no token
dropping, no padding. Recombination is a gate-weighted `index_add_`. The auxiliary
load-balancing loss is Switch's [5] `L_aux = w·E·Σ_e f_e·P_e` with w = 0.01, where
`f_e` is the (non-differentiable) fraction of routed pairs and `P_e` the mean router
probability. In all reported runs the per-layer loss sits at ≈ 0.0100–0.0102, i.e.
within ~2% of the perfectly balanced value `w·E·(1/E²·E) = w`.

### 2.5 JEPA head and VICReg objective

For self-supervised training, a **context encoder** sees the input sequence with
multi-block target spans replaced by a learned mask token (positions are kept, not
dropped, to preserve SSM time alignment); a **target encoder** — a frozen EMA copy,
`θ̄ ← m θ̄ + (1−m) θ` with m annealed 0.996 → 1.0 — sees the full sequence under
`no_grad`. A **narrow latent predictor** (`D_pred < D`) receives the context
representation with mask-token+positional-embedding queries at target slots and an
additive projection of a global action/conditioning vector, and predicts the target
embeddings. The energy is VICReg [7] evaluated on target positions only:

$$\mathcal{L} = \lambda\,\mathrm{MSE}(\hat z, \bar z)
+ \mu \sum_{branch}\frac{1}{D}\sum_d \max\!\big(0,\,\gamma - \sqrt{\mathrm{Var}(z_d)+\epsilon}\big)
+ \nu \sum_{branch}\frac{1}{D}\sum_{i\ne j} \mathrm{Cov}(Z)_{ij}^2,$$

with λ = μ = 25, ν = 1, γ = 1. The variance hinge prevents point collapse; zeroing
off-diagonal covariance prevents dimensional collapse; no negatives are needed. There
is no generative decoder anywhere in the JEPA path.

## 3. Verification methodology

Every subsystem is tested against an independently-written oracle (different algorithm,
same mathematics). The suite runs in seconds on CPU (`tests/self_test.py`):

| # | Subsystem under test | Oracle | Max discrepancy |
|---|---|---|---|
| 1 | Chunked parallel scan (fwd) + explicit recurrence | Pure-Python scalar quadruple loop | 3.8 × 10⁻⁶ |
| 2 | **Analytic adjoint backward** (all 6 input grads) | Autograd through explicit recurrence | 1.9 × 10⁻⁷ (rel.) |
| 3 | Mamba-2 SSD block algorithm (fwd **and** grads) | Per-step scalar-decay recurrence | 8.6 × 10⁻⁶ / 6.8 × 10⁻⁶ |
| 4 | Tiled sliding-window attention | Dense O(L²) masked attention, same weights/RoPE | 0.0 |
| 5 | Sparse MoE dispatch (sort/gather/GEMM/scatter) | Dense per-token loop over selected experts | 1.2 × 10⁻⁷ |
| 6 | Switch load-balancing loss | Direct recomputation from definitions | 1 × 10⁻⁶ |
| 7 | VICReg | Analytic fixed points (Inv(z,z)=0; Var hinge → 2γ at collapse) | exact |
| 8 | JEPA end-to-end | Invariants: finite loss; **zero** grads on EMA target; nonzero grads on router/predictor; EMA fixed points at m ∈ {0,1} | holds |
| 9 | Recurrent O(1)/token inference (both SSM versions + SWA cache + MoE) | Parallel forward logits over the same sequence | 1.9 × 10⁻⁷ |

Tests 2 and 3 compare *gradients*, not just outputs; test 9 guarantees that the
deployed generation path is exactly the trained model. Edge cases are exercised
deliberately: sequence lengths that are not multiples of chunk/window sizes, the
`|Δa| → 0` guarded branch, and the empty-expert case in MoE dispatch.

## 4. Empirical observations

All measurements: 16-core consumer CPU (8 PyTorch threads), fp32, PyTorch 2.12-cpu,
no GPU. Single runs (no seed averaging); we therefore report only effects far larger
than run-to-run noise.

### 4.1 Mamba-2/SSD vs. Mamba-1/S6 on CPU

Identical training setup (character LM below; 1.59M vs 1.61M parameters, same depth,
batch 8, sequence 192), differing only in the SSM block:

| Mixer | Throughput (train) | ce @ step 15 |
|---|---|---|
| Mamba-1 (S6, chunked scan + adjoint bwd) | 484 chars/s | 3.479 |
| **Mamba-2 (SSD, default)** | **3,550 chars/s (7.3×)** | 3.478 |

The loss trajectories coincide step-for-step within noise, as expected from two
formulations of closely related operators at equal parameter count. The speedup on
*CPU* is noteworthy: SSD is usually motivated by GPU tensor cores, but the same
reorganization — scalar-per-head decay shrinking the scan state from `[B,L,D,N]`
elementwise tensors to `[B,L,H]`, and intra-chunk work becoming batched GEMMs —
equally relieves a BLAS-bound CPU. The S6 path retains the exact-ZOH diagonal-`A`
parameterization for users who want it (`--ssm-version 1`).

### 4.2 Character-level LM demo (visible behavior)

A 1.59M-parameter instance (D = 64, depth 10 ⇒ 9 Mamba-2 + 1 SWA, 8 experts top-2,
~0.9M parameters active per token) was trained for 2,500 steps (≈ 18 minutes, ≈ 3.8M
characters ≈ 12 epochs) on *Memórias Póstumas de Brás Cubas* (Machado de Assis, 1881;
public domain, Project Gutenberg #54829; 368k characters, 99-symbol vocabulary,
original 1881 orthography), with a 10% held-out validation split.

- Cross-entropy: 4.60 (= ln 99, uniform) → **1.66** validation (perplexity
  **5.28**/char); train ce ≈ 1.1 (ppl ≈ 3.1). The train/validation gap is expected
  memorization on a single-book corpus and we report both honestly.
- Generation runs through the recurrent path (SSM state + K−1 attention KV cache +
  per-token MoE routing): **O(1) compute and constant memory per token** — measured
  flat at ≈52 chars/s on this CPU regardless of generated length, in contrast to a
  Transformer's growing KV-cache cost.
- Sample at temperature 0.8 (prompt in bold; orthography is the model's own — it
  learned the 1881 spelling conventions of the corpus, e.g. "ella", "idéa"):

> **Ao verme que** barbante de ver cá. E depois donravel, como uma mulher, creia se
> ir, com ella casada, a noiva de dama; vinha constituiu-lhe por esse rapido,
> inclinei-a muito, — repouso religiosa... capravam aos jornadas rifectos; era elle
> iam para a aventura da minha feição; essa nossa casa no cerebro de cima [...]

The output is not literary Portuguese — at 1.59M parameters trained on one novel, it
memorizes orthographic statistics, morphology, and short-range phrase structure. The
demo's purpose is to make the full pipeline (hybrid backbone, sparse routing,
recurrent inference) *observable* on minimal hardware.

### 4.3 JEPA training sanity

The full H-JEPA configuration (104.0M trainable parameters + 91.9M frozen EMA copy)
instantiates and trains; the tiny CPU preset shows the expected signatures over short
runs: decomposed VICReg terms reported per iteration, MoE balance loss pinned near the
balanced value, zero gradient flow into the target encoder, and EMA fixed-point
behavior at m ∈ {0, 1}. We make **no claims** about downstream representation quality;
evaluating JEPA pretraining properly requires scale and probes beyond this report.

## 5. Limitations

- **Scale.** All experiments are toy-scale by design. Nothing here predicts behavior
  at billions of parameters.
- **No controlled quality baseline.** We did not train a size-matched Transformer or
  evaluate long-context recall benchmarks; the Samba-style recall claim is inherited
  from [3], not re-established here.
- **Throughput is implementation-bound.** Our Mamba-1 path is a fair PyTorch
  implementation but not cache-optimal; official CUDA kernels would change absolute
  numbers (though not the architectural points).
- **Single seed, single corpus, single machine** for the reported runs.
- The Triton kernel is validated only when CUDA hardware is present; on the release
  machine it is exercised structurally but not executed.

## 6. Reproducibility

```bash
python3 -m venv .venv && .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python tests/self_test.py                 # verification suite (Sec. 3)
.venv/bin/python train_text.py                      # Sec. 4.2 demo (downloads corpus)
.venv/bin/python train_text.py sample --prompt "Eu era " --tokens 400
.venv/bin/python train_synthetic.py --preset tiny   # Sec. 4.3 JEPA loop
```

The released checkpoint (`lm_machado.pt`, ~6.5 MB) and exact training log are included
in the Hugging Face repository.

## Acknowledgments

The implementation, test suite, and this manuscript were developed in collaboration
with **Claude (Anthropic)** operating as an engineering assistant; all reported
numbers were produced by the released code on the author's machine. The corpus is in
the public domain (Machado de Assis, 1839–1908; digitized by Project Gutenberg).

## References

[1] A. Gu, T. Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* arXiv:2312.00752, 2023.
[2] T. Dao, A. Gu. *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality.* arXiv:2405.21060, 2024.
[3] L. Ren et al. *Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling.* arXiv:2406.07522, 2024.
[4] T. Gale, D. Narayanan, C. Young, M. Zaharia. *MegaBlocks: Efficient Sparse Training with Mixture-of-Experts.* arXiv:2211.15841, 2022.
[5] W. Fedus, B. Zoph, N. Shazeer. *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.* arXiv:2101.03961, 2021.
[6] N. Shazeer et al. *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.* arXiv:1701.06538, 2017.
[7] A. Bardes, J. Ponce, Y. LeCun. *VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning.* arXiv:2105.04906, 2021.
[8] Y. LeCun. *A Path Towards Autonomous Machine Intelligence.* OpenReview, 2022.
[9] M. Assran et al. *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA).* arXiv:2301.08243, 2023.
[10] T. Dao, D. Fu, S. Ermon, A. Rudra, C. Ré. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* arXiv:2205.14135, 2022.
[11] I. Beltagy, M. Peters, A. Cohan. *Longformer: The Long-Document Transformer.* arXiv:2004.05150, 2020.
[12] A. Gu, A. Gupta, K. Goel, C. Ré. *On the Parameterization and Initialization of Diagonal State Space Models (S4D).* arXiv:2206.11893, 2022.
[13] J. Su et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding.* arXiv:2104.09864, 2021.
