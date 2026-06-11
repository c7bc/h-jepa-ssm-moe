# -*- coding: utf-8 -*-
"""
Chat local no terminal com um modelo da MESMA FAMÍLIA ARQUITETURAL deste repositório:
híbrido Mamba-2 (SSD) + atenção — como o backbone de hjepa/ — já instruído a conversar.

Padrão: Falcon-H1-0.5B-Instruct (TII, Apache-2.0): camadas híbridas Mamba-2+atenção,
rodando 100% em CPU, sem API, sem nuvem.

    .venv/bin/python chat.py                                        # 1º uso baixa ~1 GB
    .venv/bin/python chat.py --model tiiuae/Falcon-H1-1.5B-Deep-Instruct   # mais esperto, mais lento
    .venv/bin/python chat.py --model ibm-granite/granite-4.0-h-tiny # híbrido Mamba-2 + MoE (como o nosso!)

Por que o nosso checkpoint (lm_machado.pt) não conversa? Conversar exige bilhões de
parâmetros + instruction tuning (PAPER.md §5). O Falcon-H1 é o "primo crescido" da
arquitetura: mesma engenharia de backbone, escala e treino de verdade.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, StoppingCriteria,
                          StoppingCriteriaList, TextIteratorStreamer)
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()          # sem warnings no meio da conversa

SYSTEM_PROMPT = (
    "Você é o Brás, um assistente batizado em homenagem a Brás Cubas, rodando 100% "
    "offline no computador do usuário, sem nenhuma conexão com a nuvem. Você é "
    "reflexivo, bem-humorado e direto. Responde sempre em português do Brasil, em "
    "tom de conversa, e admite sem rodeios quando não sabe algo."
)

GREY, RESET = "\033[90m", "\033[0m"
CONTEXT_TOKEN_BUDGET = 3000          # janela deslizante de histórico


class _StopOnEvent(StoppingCriteria):
    """Permite Ctrl+C cortar a geração: o evento derruba a thread do generate()."""

    def __init__(self, event: threading.Event):
        self.event = event

    def __call__(self, *args, **kwargs) -> bool:
        return self.event.is_set()


def _flush_stdin() -> None:
    """Descarta o que foi digitado fora de hora (durante a resposta do modelo) —
    evita que o texto "vaze" para o turno seguinte."""
    try:
        import termios
        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    except Exception:
        pass                                       # pipe/Windows: sem tty, sem problema


def build_inputs(tokenizer, history: list[dict]):
    """Aplica o chat template; descarta turnos antigos se estourar o orçamento."""
    while True:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + history
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        ids = tokenizer(prompt, return_tensors="pt")
        if ids.input_ids.shape[1] <= CONTEXT_TOKEN_BUDGET or len(history) <= 2:
            return ids
        history[:] = history[2:]         # descarta o par (user, assistant) mais antigo


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chat local (CPU) com modelo híbrido Mamba-2+atenção")
    parser.add_argument("--model", default="tiiuae/Falcon-H1-0.5B-Instruct")
    parser.add_argument("--max-new", type=int, default=400)
    parser.add_argument("--temp", type=float, default=0.7)
    args = parser.parse_args()

    print(f"[setup] carregando {args.model} (a primeira vez baixa o modelo)…")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.eval()
    n_par = sum(p.numel() for p in model.parameters()) / 1e9

    history: list[dict] = []

    print("┌" + "─" * 74)
    print(f"│ Brás · {args.model} ({n_par:.1f}B params) · 100% local, CPU")
    print("│ Arquitetura: híbrido Mamba-2 + atenção — a mesma família do hjepa/ deste")
    print("│ repositório, em escala de verdade e com instruction tuning.")
    print("│ Comandos: /limpar (zera a memória) · /sair · Ctrl+C corta a resposta.")
    print("└" + "─" * 74)

    while True:
        _flush_stdin()                             # ignora o que foi digitado fora de hora
        try:
            user = input("\nvocê>  ")
        except (EOFError, KeyboardInterrupt):
            print("\naté mais!")
            break

        cmd = user.strip()
        if cmd in ("/sair", "/quit", "/exit"):
            print("até mais!")
            break
        if cmd == "/limpar":
            history = []
            print("[ok] memória da conversa zerada")
            continue
        if not cmd:
            continue

        history.append({"role": "user", "content": cmd})
        ids = build_inputs(tokenizer, history)

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True,
                                        skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False)
        stop_event = threading.Event()
        gen_kwargs = dict(
            **ids, streamer=streamer, max_new_tokens=args.max_new,
            do_sample=True, temperature=args.temp, top_p=0.9, top_k=20,
            repetition_penalty=1.05,
            stopping_criteria=StoppingCriteriaList([_StopOnEvent(stop_event)]),
            pad_token_id=tokenizer.eos_token_id,
        )
        worker = threading.Thread(target=model.generate, kwargs=gen_kwargs)
        worker.start()

        # indicador imediato: a 1ª palavra demora (prefill do histórico em CPU) —
        # NÃO digite nada aqui; espere a resposta aparecer.
        sys.stdout.write(f"\nbrás>  {GREY}…formulando (a primeira palavra demora){RESET}")
        sys.stdout.flush()
        pieces: list[str] = []
        first_piece = True
        t0 = time.perf_counter()
        try:
            for piece in streamer:                     # um pedaço por token decodificado
                if first_piece:
                    sys.stdout.write("\r\033[Kbrás>  ")   # apaga o "…formulando"
                    first_piece = False
                pieces.append(piece)
                sys.stdout.write(piece)
                sys.stdout.flush()
        except KeyboardInterrupt:
            stop_event.set()
            sys.stdout.write(" [interrompido]")
        worker.join()
        dt = time.perf_counter() - t0

        history.append({"role": "assistant", "content": "".join(pieces)})
        n_tok = len(pieces)
        print(f"\n{GREY}[{n_tok} tokens em {dt:.1f}s ≈ {n_tok / max(dt, 1e-9):.1f} tok/s]{RESET}")


if __name__ == "__main__":
    main()
