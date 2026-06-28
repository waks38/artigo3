# Runner Kaggle — experimentos de imagem (E2) na GPU

`run.ipynb` roda a comparação dos 4 agregadores nos datasets de imagem na GPU do
Kaggle, loga no W&B (`artigo-3`) e salva os CSVs em `/kaggle/working` para download.

## Pré-requisitos (uma vez)

1. **GitHub**: este repo precisa estar publicado (o notebook clona via `REPO_URL`).
   Ajuste `REPO_URL`/`BRANCH` na 1ª célula de código.
2. **Kaggle Notebook settings**:
   - *Accelerator* = **GPU**
   - *Internet* = **On** (clonar repo + baixar MNIST/Fashion/CIFAR)
3. **Secret** `WANDB_API_KEY` (Add-ons → Secrets) com sua chave do W&B.
4. **Datasets reais de histopatologia** (para `colon_cancer` e `ucsb_breast`):
   adicione-os como *Input* do notebook e ajuste `COLON_ROOT`/`UCSB_ROOT`.
   - **UCSB Breast**: Kaggle `andrewmvd/breast-cancer-cell-segmentation`.
   - **Colon (CRCHistoPhenotypes)**: host oficial (Warwick) é gated; suba um mirror
     como Kaggle Dataset privado e aponte o `root` para a pasta extraída
     (`.../CRCHistoPhenotypes_2016_04_28`).
   - Se um `root` não existir, o runner **pula** aquele dataset (não quebra).

## Fluxo

1. **Teste rápido** — `QUICK = True`: roda um mínimo (2 folds, 2 épocas) em todos
   os datasets, no projeto **`artigo-3-smoke`**, só para confirmar que loga certo
   no W&B e que os dados reais foram montados. Não polui o projeto final.
2. **Run completo** — `QUICK = False`: 10-fold em sequência, projeto **`artigo-3`**,
   um run por dataset (nome = dataset).

## Resultados

- **W&B**: um run por dataset com métricas, params e probabilidades Bayesianas.
- **CSVs**: `/kaggle/working/results/<dataset>_{folds,summary,bayesian}.csv`
  (aba *Output*; use *Save Version* para persistir). Suíte completa de métricas
  por fold, incluindo localization AUC nos datasets com `instance_labels`
  (sintéticos + colon).

## Notas

- GPU única → `n_jobs=1` (fits sequenciais; paralelizar processos numa só GPU
  causaria contenção). Cada fit usa a GPU diretamente.
- 10-fold simples (`cv.n_repeats=1`) para imagem, pelo custo; o teste Bayesiano
  correlacionado é válido com um único k-fold.
- Os clássicos (elephant/fox/tiger) rodam **local/CPU** (`hopmil-compare data=...`),
  não aqui.
