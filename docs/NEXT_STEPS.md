# NEXT STEPS / HANDOFF — leia isto primeiro ao continuar

> Documento de passagem para uma nova conversa. Objetivo: dar contexto máximo
> para retomar o trabalho sem reconstruir o raciocínio. Leia também, na ordem:
> `CLAUDE.md` (resumo sempre-carregado) e `docs/ROADMAP.md` (decisões + metodologia
> completas). Este arquivo foca em **o que fazer a seguir e como**.

## 1. O que é o projeto (1 parágrafo)

Artigo: **estudo comparativo de agregadores de embeddings de bag em MIL**.
Comparamos 4 agregadores — **Hopfield, attention, max-pooling, average-pooling** —
mantendo encoder + head **fixos** por dataset (só o agregador varia). **Tese
empírica:** verificar que *atenção pode ser substituída por Hopfield* (eles
empatam nas métricas) e que *ambos superam agregação simples* (mean/max). A
equivalência Hopfield↔atenção já é provada matematicamente no paper base
(Ramsauer et al. 2020), então **NÃO** fazemos análise mecanística do β — só
evidência empírica em métricas. Compute alvo: **Kaggle** (GPU grátis) para
imagem; **local** para os clássicos.

## 2. Estado atual (implementado e verificado)

- **Agregadores** `src/hopmil/models/aggregators.py`: `mean`, `max`, `attention`
  (Ilse), `hopfield` (wrap `hflayers.HopfieldPooling`) + `build_aggregator`.
  Contrato: `(n_instances, dim) -> (bag_embedding (dim,), weights (n,)|None)`.
- **Encoders** `src/hopmil/models/encoders.py`: `MLPEncoder` (tabular),
  `CNNEncoder` (imagem; 1×28 MNIST e 3×32 CIFAR via flatten dinâmico).
- **Modelo** `src/hopmil/models/mil_model.py`: `MILClassifier` (lista de bags →
  `(logits, list_of_weights)`).
- **Lightning** `src/hopmil/training/lightning_module.py`: `MILLitModule`,
  BCE + AUROC/accuracy.
- **Dados** (todos carregam dados reais e foram verificados):
  - `MNISTBags` `src/hopmil/data/mnist_bags.py` (notebook 01).
  - `CIFARBags` `src/hopmil/data/cifar_bags.py` — fallback de mirror + cache (notebook 03).
  - `ClassicMIL` `src/hopmil/data/classic.py` — elephant/fox/tiger via Figshare (tiger: 200 bags, 230 feats).
- **Notebooks** executados: `01` (MNIST-bags), `02` (tiger/elephant), `03` (CIFAR-bags).
- **Testes**: `pytest -m "not slow"` passa.
- **Stub que falta**: `src/hopmil/training/train.py` levanta `NotImplementedError`.

## 3. Decisões que já estão fechadas (NÃO reabrir sem motivo)

- Pacotes: **uv**, Python **3.11**. Tracking: **W&B** (projeto `hopmil`).
- Loop/configs: **Lightning + Hydra**. Testes: **pytest**.
- Datasets: MNIST-bags, CIFAR-bags, e elephant/fox/tiger (cada um separado).
- Encoder fixo por modalidade; só o agregador varia.
- **Estatística**: k-fold CV (descritivo, AUC média±desvio) + **t-test Bayesiano
  correlacionado + ROPE** (inferência). **Sem frequentista / sem p-valor.**
  Ferramenta: `baycomp`. Refs no ROADMAP (Dietterich 1998; Nadeau&Bengio 2003;
  Corani&Benavoli 2015; Benavoli et al. 2017).
- **Comparação justa ("maçã com maçã")**: unidade = **condição fixa**
  `(tamanho da bag, nº testemunhas, nº de bags)`. Dentro dela rodam todos os
  k-folds e comparam-se os 4 agregadores **só ali**. HP compartilhados idênticos;
  HP específicos (β/cabeças do Hopfield; hidden da atenção) em **default fixo, sem
  tuning por agregador**.
- **Varredura = OFAT (um fator por vez) a partir de um baseline** (proposto:
  `bag=10, testemunhas=1, bags=250`; tamanho ∈ [5,10,50]; testemunhas ∈ [1,2,4];
  bags ∈ [50,100,250]). Fundamentação: ablação/sensibilidade (Ilse 2018);
  ressalva: OFAT não mede interações.
- **Mudança declarada**: tamanho de bag **fixo** por condição (`var_bag_size=0`),
  divergindo do MNIST-bags original (variável). Declarar no artigo.
- Experimentos: catálogo **E1–E5** (uma pergunta cada), parametrizável por lista
  (grupo `experiment/` no Hydra). Ver tabela no ROADMAP.
- W&B no Kaggle: **online via Secret** (`WANDB_API_KEY`); local offline/disabled.
- Recuperação: **W&B API → CSV versionado em `results/`**.
- Relatórios: `experiments/<nome>/report.md` por experimento.
- Ambiente: clássicos **local/CPU**; imagem **Kaggle/GPU**.

## 4. O QUE FAZER A SEGUIR — Marco 1 (comece aqui)

Objetivo do Marco 1: **pipeline de treino mínimo rodando** (1 condição, 1
dataset, local), provando a ponta-a-ponta antes de escalar.

### Passo 1 — `DataModule` (novo arquivo, ex. `src/hopmil/data/datamodule.py`)
- `LightningDataModule` que recebe um dataset MIL (`MILDataset`) e produz
  DataLoaders com `mil_collate` (lembrar: batch é **lista** de tensores, bags não
  são padded).
- Implementar **k-fold**: dividir os índices das bags em k folds (estratificar
  pelo label da bag é desejável). Expor `fold: int` para selecionar o fold atual.
  Os scores por fold serão a entrada do teste Bayesiano — guardar isso em mente.
- Split treino/val dentro do treino do fold (val para early-stopping/seleção).
- Para sintéticos (MNIST/CIFAR): tamanho de bag **fixo** (`var_bag_size=0`),
  `mean_bag_size`, `num_bags`, `target` etc. parametrizados pela condição.
- Seeds: as **mesmas** entre os 4 agregadores (comparação pareada).

### Passo 2 — `training/train.py` (hoje stub)
Wiring Hydra→Lightning. Pseudocódigo do `main(cfg)`:
1. `seed_everything(cfg.seed)` (já existe em `utils/seed.py`).
2. Instanciar dataset conforme `cfg.data.loader` (mnist_bags / cifar_bags / classic).
3. Encoder conforme modalidade (`MLPEncoder(in_dim=cfg.data.in_dim, dim=cfg.model.dim)`
   para tabular; `CNNEncoder(dim, in_channels, image_size)` para imagem).
4. `aggregator = build_aggregator(cfg.model.aggregator, dim=cfg.model.dim, **specifics)`.
5. `model = MILClassifier(encoder, aggregator, num_classes=cfg.data.num_classes, dim=cfg.model.dim)`.
6. `lit = MILLitModule(model, lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)`.
7. `WandbLogger` se `cfg.wandb.mode != disabled`; senão sem logger.
8. `Trainer(**cfg.trainer)` + `ModelCheckpoint(monitor="val/auc", mode="max")`.
9. `trainer.fit(lit, datamodule)`; `trainer.test(...)`.
- **Atenção** ao default do Hydra: `configs/config.yaml` aponta `data: mnist_bags`,
  `model: attention`. Há `configs/data/{mnist_bags,cifar_bags,elephant,fox,tiger}.yaml`
  e `configs/model/{attention,hopfield}.yaml`. Faltam configs de `mean` e `max`
  (criar `configs/model/mean.yaml` e `max.yaml`).

### Passo 3 — validar
Rodar local, sem W&B, 1 agregador, dataset pequeno:
```bash
uv run hopmil-train data=elephant model=attention wandb.mode=disabled trainer.max_epochs=20
```
Confirmar que treina, loga val/auc, e termina sem erro. Esse é o "verde" do Marco 1.

## 5. Como rodar / ambiente

- Ambiente já instalado: `.venv` (via `uv sync --extra dev`). Sempre via `uv run`.
- Para o agregador Hopfield: `PYTHONUTF8=1 uv sync --extra hopfield` (Windows/cp1252).
- Kernel Jupyter registrado: **"Python (hopmil)"**.
- Testes: `uv run pytest -m "not slow"`.
- Dados já em cache local: MNIST, CIFAR (`data/raw/cifar10_train.pt`), elephant/fox/tiger.
  Tudo em `data/` é **gitignored**.

## 6. Gotchas (já resolvidos, mas saiba)

- `hopfield-layers` quebra build no Windows sem `PYTHONUTF8=1` (cp1252).
- CIFAR-10: host de Toronto pode estar bloqueado → `CIFARBags` cai num mirror
  fast.ai e cacheia (`data/raw/cifar10_*.pt`) automaticamente.
- `MILClassifier` recebe **lista** de bags; não tente empilhar num tensor denso.

## 7. Micro-decisões ainda em aberto (perguntar ao usuário)

> Preferência do usuário: **perguntas abertas, uma de cada vez** (não baterias de
> múltipla escolha). Está salvo na memória do projeto.

- Valor de **k** no k-fold (10?) e nº de **repetições** (ex. 10×10-fold) para alimentar o Bayesiano.
- Valor final da **ROPE** `r` (proposto 0.01 de AUC) e a regra de decisão (proposto `P(rope)≥0.95`).
- Valores finais do **baseline** e das varreduras OFAT (os atuais são propostas).
- GitHub: ainda **não existe repo remoto**; criar quando for rodar no Kaggle.

## 8. Git

Branch `master`. Commits até aqui:
- `42db8ae` scaffold; `45703b9` convenções de experimento; `ed02ddc` desenho experimental + metodologia.
Sem remoto ainda. Dados/resultados/checkpoints gitignored; `uv.lock` commitado.
