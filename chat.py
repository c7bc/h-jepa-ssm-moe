# -*- coding: utf-8 -*-
"""
Chat de VERDADE no terminal — 100% local, sem API, sem nuvem.

Roda um modelo aberto instruction-tuned (padrão: Qwen3-0.6B, Apache-2.0) em CPU,
com streaming e "modo pensar": digite /pensar e você passa a VER a introspecção
do modelo (em cinza) antes de cada resposta.

    .venv/bin/python chat.py                          # 1º uso baixa ~1,5 GB
    .venv/bin/python chat.py --model Qwen/Qwen3-1.7B  # mais esperto (se tiver RAM)

Por que não é o H-JEPA-SSM-MoE deste repo conversando? Conversar exige bilhões de
parâmetros + instruction tuning (ver PAPER.md §5). Este chat.py existe para a
EXPERIÊNCIA de conversar com um modelo local; o hjepa/ existe para a ENGENHARIA
da arquitetura. São peças complementares do mesmo estudo.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, StoppingCriteria,
                          StoppingCriteriaList, TextIteratorStreamer)

SYSTEM_PROMPT = (
    "Você é o Brás, um assistente batizado em homenagem a Brás Cubas, rodando 100% "
    "offline no computador do usuário, sem nenhuma conexão com a nuvem. Você é "
    "reflexivo, bem-humorado e direto. Responde sempre em português do Brasil, em "
    "tom de conversa; admite sem rodeios quando não sabe algo; e quando o assunto "
    "pede, gosta de pensar em voz alta antes de concluir."
)

GREY, RESET = "\033[90m", "\033[0m"
CONTEXT_TOKEN_BUDGET = 3000          # janela deslizante de histórico


class _StopOnEvent(StoppingCriteria):
    """Permite Ctrl+C cortar a geração: o evento derruba a thread do generate()."""

    def __init__(self, event: threading.Event):
        self.event = event

    def __call__(self, *args, **kwargs) -> bool:
        return self.event.is_set()


def build_inputs(tokenizer, history: list[dict], thinking: bool):
    """Aplica o chat template; descarta turnos antigos se estourar o orçamento."""
    while True:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + history
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=thinking,
        )
        ids = tokenizer(prompt, return_tensors="pt")
        if ids.input_ids.shape[1] <= CONTEXT_TOKEN_BUDGET or len(history) <= 2:
            return ids
        history[:] = history[2:]         # descarta o par (user, assistant) mais antigo


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat local (CPU) com modelo aberto")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max-new", type=int, default=600)
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--thinking", action="store_true",
                        help="começa com o modo pensar ligado")
    args = parser.parse_args()

    print(f"[setup] carregando {args.model} (a primeira vez baixa o modelo)…")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.eval()
    n_par = sum(p.numel() for p in model.parameters()) / 1e9

    thinking = args.thinking
    history: list[dict] = []

    print("┌" + "─" * 74)
    print(f"│ Brás · {args.model} ({n_par:.1f}B params) · 100% local, CPU")
    print("│ Converse normalmente. Comandos: /pensar (vê a introspecção em cinza),")
    print("│ /limpar (zera a memória), /sair. Ctrl+C corta uma resposta no meio.")
    print(f"│ modo pensar: {'LIGADO' if thinking else 'desligado'}")
    print("└" + "─" * 74)

    while True:
        try:
            user = input("\nvocê>  ")
        except (EOFError, KeyboardInterrupt):
            print("\naté mais!")
            break

        cmd = user.strip()
        if cmd in ("/sair", "/quit", "/exit"):
            print("até mais!")
            break
        if cmd == "/pensar":
            thinking = not thinking
            print(f"[ok] modo pensar {'LIGADO — a introspecção aparece em cinza' if thinking else 'desligado'}")
            continue
        if cmd == "/limpar":
            history = []
            print("[ok] memória da conversa zerada")
            continue
        if not cmd:
            continue

        history.append({"role": "user", "content": cmd})
        ids = build_inputs(tokenizer, history, thinking)

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True,
                                        skip_special_tokens=True)
        stop_event = threading.Event()
        # amostragem recomendada pelo Qwen3: pensar → 0.6/0.95; direto → temp/0.8
        gen_kwargs = dict(
            **ids, streamer=streamer, max_new_tokens=args.max_new,
            do_sample=True,
            temperature=0.6 if thinking else args.temp,
            top_p=0.95 if thinking else 0.8, top_k=20,
            stopping_criteria=StoppingCriteriaList([_StopOnEvent(stop_event)]),
            pad_token_id=tokenizer.eos_token_id,
        )
        worker = threading.Thread(target=model.generate, kwargs=gen_kwargs)
        worker.start()

        sys.stdout.write("\nbrás>  ")
        sys.stdout.flush()
        pieces: list[str] = []
        t0 = time.perf_counter()
        try:
            for piece in streamer:                     # um pedaço por token decodificado
                pieces.append(piece)
                shown = piece.replace("<think>", GREY + "┆ pensando… ")
                shown = shown.replace("</think>", RESET + "\n")
                sys.stdout.write(shown)
                sys.stdout.flush()
        except KeyboardInterrupt:
            stop_event.set()
            sys.stdout.write(RESET + " [interrompido]")
        worker.join()
        dt = time.perf_counter() - t0

        reply = "".join(pieces)
        if "</think>" in reply:                        # só a conclusão vai pra memória
            reply = reply.split("</think>", 1)[1].lstrip()
        history.append({"role": "assistant", "content": reply})

        n_tok = len(pieces)
        print(f"\n{GREY}[{n_tok} tokens em {dt:.1f}s ≈ {n_tok / max(dt, 1e-9):.1f} tok/s]{RESET}")


if __name__ == "__main__":
    main()
