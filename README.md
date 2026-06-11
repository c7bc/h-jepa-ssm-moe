# H-JEPA-SSM-MoE

Uma arquitetura de IA pós-Transformer, escrita do zero em PyTorch, que roda **num
laptop sem placa de vídeo**. Junta quatro ideias de pesquisa:

| Peça | Paper | Analogia de dev web |
|---|---|---|
| **Mamba-2 (SSM/SSD)** | Gu & Dao, 2023/2024 | Um *stream* com estado: processa a sequência como um `reduce()` — guarda um "resumo" de tamanho fixo e atualiza a cada item. Custo linear, não quadrático como a atenção do Transformer. A versão 2 (SSD) reorganiza a conta em multiplicações de matrizes densas: ~7× mais rápida até em CPU. O Mamba-1 clássico segue disponível via `--ssm-version 1`. |
| **Atenção local (SWA)** | Samba, 2024 | Um `Ctrl+F` numa janela dos últimos K tokens. O Mamba comprime (e perde detalhe); a atenção recupera o detalhe exato. 1 camada de atenção a cada 9 de Mamba. |
| **MoE esparso** | MegaBlocks, 2022 | Um *load balancer* na frente de 8 microsserviços (especialistas). Cada token é roteado para os 2 mais adequados — o modelo tem muitos parâmetros, mas cada token só paga por 2/8 deles. |
| **JEPA** | LeCun, 2022 | Aprender *embeddings* sem rótulo: esconde pedaços da sequência e treina o modelo para prever a **representação** do que falta (não o pixel/caractere em si). Tipo treinar um modelo de recomendação só observando, sem ninguém anotar nada. |

## Rodar (3 comandos)

```bash
python3 -m venv .venv && .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python tests/self_test.py        # verifica a matemática (tudo deve dar PASS)
.venv/bin/python train_text.py             # treina o mini-LM no Machado de Assis e GERA texto
```

O `train_text.py` baixa "Memórias Póstumas de Brás Cubas" (1881, domínio público,
Project Gutenberg), treina um modelo de ~1,5M de parâmetros caractere por caractere
(~40 min de CPU) e imprime amostras de texto durante e ao final do treino. Para gerar
de novo depois, sem retreinar:

```bash
.venv/bin/python train_text.py sample --prompt "Eu era " --tokens 400
```

### O que esperar

- A loss começa em ≈ 4,6 (`ln(99)`: chute uniforme entre 99 caracteres) e cai para ≈ 1,6–1,9.
- Amostras no começo são sopa de letras; no fim viram português arcaico inventado,
  com pontuação e nomes do livro. É um modelo *minúsculo* treinado num livro só —
  o objetivo é ver a mecânica funcionando, não competir com ChatGPT 😄.
- Na geração, repare na taxa de chars/s **constante**: a inferência é recorrente
  (custo O(1) por token). Num Transformer, gerar fica mais lento conforme o texto cresce.

## Os dois treinos

| Script | O que faz | Para que serve |
|---|---|---|
| `train_text.py` | Mini modelo de **linguagem** (gera texto) | Demo visível do backbone híbrido |
| `train_synthetic.py` | Treino **JEPA** auto-supervisionado (sem rótulos, sem gerar nada) | A proposta de pesquisa em si: aprender representações prevendo no espaço latente |

Os dois usam exatamente o mesmo backbone (`hjepa/backbone.py`).

## Estrutura

```
hjepa/
  config.py    dataclasses de configuração e presets
  scan.py      Mamba-1: recorrência S6 (forward paralelo em blocos + backward
               analítico derivado à mão + kernel Triton p/ GPU)
  mamba2.py    Mamba-2: algoritmo SSD por blocos (padrão — ~7× mais rápido em CPU)
  layers.py    bloco Mamba-1, atenção local de janela, RMSNorm, RoPE
  moe.py       roteador top-2 com ruído + despacho esparso dos especialistas
  backbone.py  empilha tudo (9 SSM : 1 atenção, MoE em toda camada)
  jepa.py      encoders contexto/alvo (EMA), preditor latente, perda VICReg
  lm.py        cabeça de linguagem p/ o demo + geração recorrente O(1)/token
  data.py      dataset sintético + download do corpus público
  train.py     otimizador, schedules, loop de treino JEPA
tests/self_test.py   compara cada peça com uma implementação-oráculo independente
```

## FAQ

**Por que roda sem GPU?** Tudo tem caminho PyTorch puro; o kernel Triton (GPU) é só
uma otimização opcional, ativada automaticamente quando existe CUDA.

**Isso é "o" Samba/Mamba oficial?** É uma implementação independente e didática da
mesma matemática, verificada contra oráculos numéricos nos testes. Para produção de
verdade em GPU, as libs oficiais (`mamba-ssm`, `megablocks`) têm kernels CUDA maduros.

**O texto gerado é "plágio" do livro?** O modelo aprende estatísticas de caracteres e
inventa continuações; o corpus é domínio público (Machado morreu em 1908).
