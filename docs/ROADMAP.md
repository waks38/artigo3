# ROADMAP — estado vivo do projeto

> Documento de memória do projeto. Atualize ao fim de cada sessão. O `CLAUDE.md`
> é o resumo sempre-carregado; este arquivo é o detalhe (decisões, estado, o que
> falta definir). Em caso de divergência, este arquivo é a fonte de verdade do
> *planejamento*; o código é a fonte de verdade da *implementação*.

## Objetivo do artigo

Estudo **comparativo de agregadores de embeddings de bag em MIL**: comparar
**Hopfield (Ramsauer 2020)**, **attention (Ilse 2018)**, **max-pooling** e
**average-pooling**, mantendo encoder + head **fixos** por dataset, de modo que
o agregador seja a única variável. Foco em comparação justa e reprodutível
(performance + interpretabilidade), não em eleger um vencedor a priori.

## Decisões fechadas

| Tema | Decisão |
|---|---|
| Gerenciador de pacotes | **uv** (`.venv`), Python **3.11** (torch não tem wheel p/ 3.14) |
| Camada Hopfield | **hopfield-layers (ml-jku)**, import lazy, extra opcional `hopfield` |
| Tracking | **Weights & Biases**, projeto `hopmil` (detalhes de logging: a definir) |
| Loop de treino / configs | **Lightning** + **Hydra** |
| Testes | **pytest** (marca `slow` p/ testes que importam hflayers) |
| Compute | **Kaggle Notebooks** (GPU grátis); roda local e no Kaggle |
| Fluxo Kaggle | push GitHub → notebook clona + `pip install -e .` → entrypoint; W&B via Kaggle Secret |
| Encoder | fixo por modalidade: `MLPEncoder` (tabular), `CNNEncoder` (imagem) |
| Datasets | MNIST-bags, CIFAR-bags, e elephant/fox/tiger (cada um separado) |

## Implementado e verificado

- **Agregadores** (`models/aggregators.py`): mean, max, attention, hopfield + `build_aggregator`. Contrato testado.
- **Encoders** (`models/encoders.py`): `MLPEncoder`, `CNNEncoder` (RGB/tamanho dinâmico).
- **Modelo** (`models/mil_model.py`): `MILClassifier` (lista de bags → logits + weights).
- **Lightning** (`training/lightning_module.py`): BCE + AUROC/accuracy.
- **Dados**:
  - `MNISTBags` (`data/mnist_bags.py`) — verificado no notebook 01.
  - `ClassicMIL` (`data/classic.py`) — elephant/fox/tiger via Figshare; verificado (tiger: 200 bags, 230 feats).
  - `CIFARBags` (`data/cifar_bags.py`) — com fallback de mirror + cache; verificado no notebook 03.
- **Notebooks**: 01 (MNIST), 02 (tiger/elephant), 03 (CIFAR). Todos executados.
- **Testes**: passam (`pytest -m "not slow"`).

## Pendente (próximos passos)

1. **`training/train.py`** — wiring Hydra→Lightning (hoje stub). *Bloqueador do treino.*
2. **DataModule** — dataset + split treino/val/test (CV nos clássicos) + `mil_collate`.
3. **W&B**: o que logar, nomenclatura de runs, artefatos, recuperação dos resultados.
4. **Kaggle runner** (`kaggle/run.ipynb`).
5. **Convenção de relatório por experimento** (`experiments/`).
6. Resolver justiça de parâmetros (reportar nº de params por agregador).
7. Ablações do Hopfield (β/temperatura, nº de cabeças).

## Decisões EM ABERTO (precisamos definir juntos)

### A. Ambiente de execução dos experimentos
- Os clássicos (200 bags) treinam em segundos — rodar **local** ou também no Kaggle?
- GPU do Kaggle só para CIFAR/MNIST-bags maiores e (futuro) WSI?
- Como anexar dados no Kaggle (CIFAR/MNIST baixam; clássicos = Figshare ou Kaggle Dataset)?

### B. Logging e recuperação dos resultados
- Modo W&B no Kaggle: **offline + sync** depois, ou online via Secret?
- O que logar por run: métricas (AUC/acc), curva, pesos de atenção, contagem de params, config completa, seed.
- Como puxar os números de volta para as tabelas do artigo: export via API do W&B? CSV versionado em `results/`?
- Checkpoints: salvar quais, onde (gitignored), como recuperar o melhor.

### C. Relatórios de experimento
- Formato: um markdown por experimento em `experiments/<nome>/`? Gerado a partir do W&B?
- O que cada relatório contém: hipótese, setup, tabela de resultados, figuras, conclusão.
- Reprodutibilidade: comando Hydra exato + commit hash + seeds.

## Notebooks (o que cada um mostra)
- **01_explore_and_forward** — MNIST-bags: anatomia da bag, forward passo a passo, interpretabilidade.
- **02_explore_tiger_elephant** — clássicos tabulares: por que não são imagens, heatmap + PCA das features.
- **03_explore_cifar_bags** — CIFAR-bags: crops visíveis, testemunha conhecida, forward + interpretabilidade.

## Gotchas (Windows)
- `PYTHONUTF8=1 uv sync --extra hopfield` (cp1252 quebra o build do hflayers).
- Kernel Jupyter registrado como **"Python (hopmil)"**.
- CIFAR-10: host de Toronto pode estar bloqueado → fallback fast.ai + cache automático.
