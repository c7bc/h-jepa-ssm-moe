# -*- coding: utf-8 -*-
"""
Dados: dataset sintético (JEPA) e corpus de texto público (demo do mini-LM).

O corpus padrão é "Memórias Póstumas de Brás Cubas" (Machado de Assis, 1881 —
domínio público, Project Gutenberg). Fallbacks: Dom Casmurro e Tiny Shakespeare.
"""

from __future__ import annotations

import math
import os
import urllib.request

import torch
from torch.utils.data import Dataset

CORPUS_URLS = [
    # (apelido, url)
    ("bras_cubas", "https://www.gutenberg.org/cache/epub/54829/pg54829.txt"),
    ("dom_casmurro", "https://www.gutenberg.org/cache/epub/55752/pg55752.txt"),
    ("tiny_shakespeare",
     "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"),
]


# ═════════════════════════════════════════════════════════════════════════════════════
# Dataset sintético (treino JEPA auto-supervisionado)
# ═════════════════════════════════════════════════════════════════════════════════════


class SyntheticSequenceDataset(Dataset):
    """
    Sequências [L, D_in] com dinâmica latente determinística por índice:
      • M osciladores (freq/fase/amplitude por item) → projeção fixa em D_in;
      • pulsos esparsos localizados ("agulhas" para a camada SWA recuperar);
      • ruído de sensor gaussiano.
    A variável de AÇÃO é [freqs/4 ‖ amps/2] — a informação de que o preditor
    precisa para extrapolar no espaço latente. Determinístico por (seed, idx).
    """

    def __init__(self, num_samples: int, seq_len: int, input_dim: int,
                 action_dim: int, seed: int = 1234):
        super().__init__()
        self.num_samples = num_samples
        self.seq_len, self.input_dim = seq_len, input_dim
        self.n_modes = action_dim // 2                              # M
        self.seed = seed
        g = torch.Generator().manual_seed(seed)
        self.proj = torch.randn(self.n_modes, input_dim, generator=g) / math.sqrt(self.n_modes)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + idx)
        L, M = self.seq_len, self.n_modes

        freq = torch.empty(M).uniform_(0.5, 4.0, generator=g)       # ciclos por sequência
        phase = torch.empty(M).uniform_(0.0, 2 * math.pi, generator=g)
        amp = torch.empty(M).uniform_(0.5, 2.0, generator=g)

        t = torch.arange(L, dtype=torch.float32).unsqueeze(-1) / L  # [L, 1]
        modes = amp * torch.sin(2 * math.pi * freq * t + phase)     # [L, M]
        x = modes @ self.proj                                       # [L, D_in]

        n_events = int(torch.randint(1, 4, (1,), generator=g))
        for _ in range(n_events):
            width = int(torch.randint(2, max(3, L // 16) + 1, (1,), generator=g))
            pos = int(torch.randint(0, L - width + 1, (1,), generator=g))
            pattern = torch.randn(self.input_dim, generator=g) * 0.5
            x[pos:pos + width] += pattern                           # pulso [w, D_in]

        x = x + torch.randn(L, self.input_dim, generator=g) * 0.05
        action = torch.cat([freq / 4.0, amp / 2.0])                 # [A] normalizado
        return x.contiguous(), action.contiguous()


# ═════════════════════════════════════════════════════════════════════════════════════
# Corpus de texto público (demo do mini-LM em nível de caractere)
# ═════════════════════════════════════════════════════════════════════════════════════


def _strip_gutenberg(text: str) -> str:
    """Remove cabeçalho/rodapé legal do Project Gutenberg (*** START/END ***)."""
    start = text.find("*** START")
    if start != -1:
        nl = text.find("\n", start)
        if nl != -1:
            text = text[nl + 1:]
    end = text.find("*** END")
    if end != -1:
        text = text[:end]
    return text.strip()


def load_char_corpus(path_or_url: str | None = None, cache_dir: str = "data") -> str:
    """
    Carrega o corpus: arquivo local, URL explícita, cache em disco ou download dos
    corpora públicos padrão (na ordem de CORPUS_URLS).
    """
    os.makedirs(cache_dir, exist_ok=True)

    if path_or_url and os.path.exists(path_or_url):
        with open(path_or_url, encoding="utf-8") as fh:
            return _strip_gutenberg(fh.read())

    candidates = ([("custom", path_or_url)] if path_or_url else []) + CORPUS_URLS
    for name, url in candidates:
        cache = os.path.join(cache_dir, f"{name}_raw.txt")
        try:
            if not os.path.exists(cache):
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=60) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
                with open(cache, "w", encoding="utf-8") as fh:
                    fh.write(raw)
            with open(cache, encoding="utf-8") as fh:
                text = _strip_gutenberg(fh.read())
            if len(text) > 10_000:
                print(f"[dados] corpus '{name}': {len(text):,} caracteres")
                return text
        except Exception as exc:                                    # tenta o próximo
            print(f"[dados] falha em '{name}' ({exc}); tentando o próximo…")
    raise RuntimeError("nenhum corpus disponível (sem rede e sem cache em data/)")


class CharVocab:
    """Vocabulário em nível de caractere (encode/decode reversíveis)."""

    def __init__(self, text: str):
        self.itos = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, s: str) -> torch.Tensor:
        return torch.tensor([self.stoi[c] for c in s if c in self.stoi], dtype=torch.long)

    def decode(self, t: torch.Tensor) -> str:
        return "".join(self.itos[int(i)] for i in t)


def get_batch(data: torch.Tensor, seq_len: int, batch_size: int,
              device: torch.device):
    """Janelas aleatórias contíguas: x = data[i:i+L], y = data[i+1:i+L+1]. [B, L]."""
    ix = torch.randint(0, data.shape[0] - seq_len - 1, (batch_size,))
    x = torch.stack([data[i:i + seq_len] for i in ix])
    y = torch.stack([data[i + 1:i + seq_len + 1] for i in ix])
    return x.to(device), y.to(device)
