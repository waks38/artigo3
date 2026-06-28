# NEXT STEPS / HANDOFF — leia isto primeiro ao continuar

> Documento de passagem para uma nova conversa. Objetivo: retomar o trabalho sem
> reconstruir o raciocínio. Leia também: `CLAUDE.md` (resumo sempre-carregado),
> `docs/ROADMAP.md` (plano) e `docs/METHODOLOGY.md` (metodologia para o texto).

## 1. O que é o projeto (1 parágrafo)

Artigo: **estudo comparativo de agregadores de bag em MIL**. Comparamos 4
agregadores — **Hopfield, attention, max, mean** — com encoder + head **fixos**
por dataset (só o agregador varia). **Tese empírica:** *atenção pode ser
substituída por Hopfield* (empatam) e *ambos superam agregação simples*
(mean/max). Sem análise mecanística do β — só evidência empírica em métricas.

## 2. Estado atual (o que JÁ está pronto e verificado)

**Pipeline e comparação (Marcos 1 e 2 — concluídos):**
- `MILDataModule` (`data/datamodule.py`): k-fold estratificado + train/val/test +
  `mil_collate`, determinístico no seed.
- `training/train.py`: 1 treino (Hydra→Lightning). `training/factory.py`: builders
  de dataset/encoder compartilhados.
- `eval/compare.py` (entrypoint **`hopmil-compare`**): compara os 4 agregadores
  nos **mesmos folds** (pareado), k-fold repetido, early-stopping em val/auc,
  **`n_jobs`** (folds em paralelo via joblib), 1 run W&B por dataset, CSVs em
  `results/`. Logs do terminal são ASCII (Windows cp1252 não quebra mais).
- `eval/stats.py`: t-test Bayesiano correlacionado + ROPE via `baycomp`
  (`BayesResult` com `p_a_better/p_rope/p_b_better`; orientação travada em teste).
- **Suíte completa de métricas por fold** (`eval/metrics.py`): auroc, auprc,
  accuracy, balanced_acc, f1, precision, recall, mcc, brier, log_loss, e
  **localization_auc** (nos datasets com `instance_labels`). Tudo vai pros CSVs e
  W&B; o Bayesiano roda em toda métrica bem-definida.

**Datasets (E1 + E2):**
- Tabular: `elephant/fox/tiger` (`ClassicMIL`).
- Imagem sintético: `mnist_bags`, `fashion_mnist_bags`, `cifar_bags` — **condição
  fixa** (bag=15, **1 testemunha** nas positivas, balanceado, 250 bags) via
  `data/synthetic.py::make_witness_bags`. Têm `instance_labels` → localização.
- Imagem real (histopatologia): `colon_cancer` (patch 27×27 por núcleo, positiva
  se ≥1 epitelial; tem `instance_labels`) e `ucsb_breast` (patch 32×32 em grade,
  rótulo benign/malignant pelo nome; sem `instance_labels`). **Não baixam
  sozinhos** (hosts gated/down): baixar via Kaggle e apontar `data.root`. Parsers
  testados com fixtures sintéticas.

**E1 (clássicos) — JÁ RODADO** (10×10-fold, local). Resultados em
`results/{elephant,fox,tiger}_{folds,summary,bayesian}.csv` e no W&B `artigo-3`.
Resumo: os 4 empatam (~0.87–0.90 AUC), vereditos "inconclusivo" — esperado para
os clássicos (datasets fáceis; a tese se decide na imagem/E2).

**Kaggle runner — PRONTO** (`kaggle/run.ipynb` + `kaggle/README.md`): clona o
repo, instala `[hopfield]`, lê `WANDB_API_KEY` do Secret, roda `compare` em GPU
nos datasets de imagem, salva CSVs em `/kaggle/working`. `QUICK=True` faz um smoke
em todos no projeto `artigo-3-smoke`.

**Testes**: `pytest` = 27 passam; `ruff` limpo.

## 3. Decisões fechadas (NÃO reabrir sem motivo)

- uv + Python 3.11; Lightning + Hydra; W&B (projeto `artigo-3`); pytest.
- Estatística: k-fold CV (descritivo) + **t-test Bayesiano correlacionado + ROPE**
  (`baycomp`); **sem frequentista**. ROPE `r=0.01` AUC; regra `P≥0.95`.
- Comparação pareada: 4 agregadores nos mesmos folds/seeds; HP compartilhados
  idênticos; HP específicos (β/heads Hopfield; hidden atenção) em default fixo,
  **sem tuning por agregador**.
- **OFAT (E3–E5) DESCARTADO.** Sintéticos = **uma condição fixa** (bag=15, 1
  testemunha, balanceado, 250 bags). Catálogo final: **E1 (tabular) + E2 (imagem)**.
- CV: clássicos 10×10 (local/CPU); **imagem 10-fold simples** (`cv.n_repeats=1`,
  GPU/Kaggle, pelo custo) — o teste Bayesiano correlacionado é válido com 1 k-fold.
- Ambiente: clássicos local/CPU (`n_jobs=-1/-2`); imagem Kaggle/GPU (`n_jobs=1`).

## 4. O QUE FAZER A SEGUIR (em ordem)

1. **Criar o repo no GitHub** (não existe remoto). O Kaggle clona de lá; pôr a URL
   em `kaggle/run.ipynb` (`REPO_URL`). Sugиro `gh repo create` + push do `master`.
2. **Subir os dados reais como Kaggle Datasets** e ajustar `COLON_ROOT`/`UCSB_ROOT`
   no notebook. UCSB tem mirror pronto (`andrewmvd/breast-cancer-cell-segmentation`);
   colon precisa de um mirror seu (Warwick é gated). Sem isso, o runner **pula**
   colon/ucsb (não quebra) — dá pra rodar só os 3 sintéticos primeiro.
3. **Rodar o Kaggle**: `QUICK=True` (valida logging em `artigo-3-smoke`) → depois
   `QUICK=False` (E2 completo em `artigo-3`).
4. **Análise/figuras**: tabelas AUC média±desvio, gráfico simplex do baycomp por
   par, e `experiments/<nome>/report.md` por experimento. (`compare.py` já grava os
   CSVs; um `collect.py` via W&B API é opcional.)

## 5. Como rodar / ambiente

```bash
# clássicos (local, já rodado — refazer se quiser):
uv run hopmil-compare -m data=elephant,fox,tiger n_jobs=-2

# teste rápido de um dataset (sem W&B):
uv run hopmil-compare data=elephant wandb.mode=disabled cv.n_repeats=2 cv.n_folds=3 trainer.max_epochs=5

# imagem: ver kaggle/run.ipynb
uv run pytest -q                      # 27 testes
uv run ruff check . && uv run ruff format .
```

## 6. Gotchas (já resolvidos, mas saiba)

- **`hopfield` extra no Windows**: `PYTHONUTF8=1 uv sync --extra hopfield` (cp1252
  quebra o build do hflayers).
- **`uv run` trava se um `hopmil-compare` estiver rodando** (não consegue
  substituir o `.exe`). Contorno: `./.venv/Scripts/python.exe -m hopmil.eval.compare ...`.
- **Logs no terminal**: glifos `→/≈/±` viram ASCII (`->`, `~=`, `+/-`) — não quebra
  mais no cp1252. Os CSVs/W&B mantêm o conteúdo certo.
- `MILClassifier` recebe **lista** de bags (não empilhar). GPU: o `_step` já move
  as instâncias para o device.
- `results/` é **gitignored** (contradiz o texto "CSV commitado"); decidir se
  versiona (force-add) ou não — está aberto.

## 7. Micro-decisões ainda em aberto

> Preferência do usuário: **perguntas abertas, uma de cada vez**.

- Sensibilidade da ROPE (reportar 0.005/0.02 além de 0.01?).
- Versionar ou não os CSVs de `results/` (hoje gitignored).
- Como hospedar o mirror do colon cancer no Kaggle.

## 8. Git

Branch `master`. Último commit: `4d04ad1` (handoff antigo). **Há ~38 arquivos
não-commitados** com todo o trabalho dos Marcos 1–2 + métricas + datasets +
Kaggle. **Recomendado commitar antes de migrar** (ainda não foi feito; o usuário
não pediu commit explicitamente). Sem remoto ainda.
