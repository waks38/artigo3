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

## Desenho experimental (tese + catálogo)

**Tese central (empírica, só métricas):** verificar a afirmação do paper base de
que **atenção pode ser substituída por Hopfield** — mostrando empiricamente que
Hopfield **empata/supera a atenção** — e que ambos **superam agregação simples**
(mean/max). A equivalência Hopfield↔atenção **já está provada matematicamente**
no paper base, então a análise mecanística do β está **fora do escopo**; nossa
contribuição é a evidência empírica em métricas.

**Papéis dos 4 agregadores:** atenção = referência que o Hopfield substitui;
mean/max = piso ingênuo a ser batido. Em todo experimento variam-se os mesmos 4.

**Regra:** cada experimento responde UMA pergunta do artigo (= um parágrafo/figura).

| Experimento | Pergunta | Cenário (o que muda) | Mede |
|---|---|---|---|
| **E1 — Benchmark tabular** | Hopfield empata a atenção e ambos batem mean/max nos benchmarks? | elephant / fox / tiger | AUC (10-fold) |
| **E2 — Benchmark imagem** | Mesma pergunta, em MIL de imagem | MNIST-bags, CIFAR-bags | AUC + localization |
| **E3 — Estresse: tamanho da bag** | A vantagem sobre pooling simples cresce com a bag? | bag = [5,10,50,100] | AUC vs tamanho |
| **E4 — Estresse: testemunhas** | E com poucas/muitas instâncias-alvo? | nº testemunhas | AUC vs testemunhas |
| **E5 — Eficiência de dados** | Quem aprende com menos bags de treino? | nº bags = [50..500] | AUC vs nº treino |

E1–E2 = evidência da tese; E3–E5 = suporte ("quando/quanto").

**Parametrização:** cada experimento é um config (grupo `experiment/` no Hydra);
uma **lista** seleciona quais rodar; dentro de cada um, o eixo varrido também é
uma lista editável. Nada roda que não esteja listado.

### Protocolo de comparação justa (maçã com maçã)
- **Unidade de comparação = uma condição fixa** `(tamanho da bag, nº testemunhas,
  nº de bags)`. Dentro dela rodam-se **todos os k-folds** e comparam-se os 4
  agregadores **só ali**. Não se misturam condições diferentes numa comparação
  intra-dataset; o veredito cross-condição (Bayesiano multi-dataset) é uma etapa
  separada.
- **Hiperparâmetros**: os **compartilhados** (encoder, head, lr, épocas, batch,
  `D`, folds, seeds) são **idênticos** para os 4. Os **específicos** de cada um
  (β/cabeças do Hopfield; hidden da atenção) ficam num **default fixo, sem tuning
  por agregador**.
- **Eixos varridos = um fator por vez (OFAT) a partir de um baseline**, para não
  explodir a grade (3×3×3=27 → ~7 condições):
  - baseline: `bag=10, testemunhas=1, bags=250`
  - tamanho da bag: `[5, 10, 50]`
  - nº testemunhas: `[1, 2, 4]`
  - nº de bags: `[50, 100, 250]`
- **Fundamentação**: é análise de sensibilidade / ablação, prática padrão em ML
  (ex. **Ilse et al., 2018**, variam o nº de bags de treino no MNIST-bags).
  **Ressalva**: OFAT não captura interações entre fatores (um fatorial completo
  capturaria, mas explode); aceitamos a troca por clareza e custo.
- **MUDANÇA vs. MNIST-bags original**: Ilse et al. usam **tamanho de bag
  variável** (amostrado de uma Gaussiana). Nós usamos **tamanho FIXO por
  condição** (`var_bag_size=0`), porque o OFAT exige o fator controlado, não
  aleatório. Divergência intencional — deve ser declarada no artigo.

## Metodologia estatística (como comparamos — e o que podemos afirmar)

> Escrita para ser explicável no artigo. Resumo: **NÃO** usamos testes de
> significância clássicos (p-valor). Usamos **análise Bayesiana** com ROPE, que
> (a) corrige a correlação da validação cruzada e (b) permite *afirmar
> equivalência* — exatamente o que a tese "atenção pode ser substituída por
> Hopfield" exige.

### Por que não o t-test clássico
1. **Correlação da CV.** Em k-fold, os treinos se sobrepõem → as métricas dos
   folds são correlacionadas, não independentes. O t-test pareado assume
   independência; sem ela ele **subestima a variância** e produz significância
   falsa (erro tipo I inflado). Demonstrado em **Dietterich (1998)**.
2. **Mais folds não ajuda.** Aumentar k (ex. 30-fold) aumenta a sobreposição
   dos treinos (mais correlação) e reduz a variância de forma *artificial* — não
   gera informação independente. Nº de folds é para *estimar performance*, não
   para validar o teste.
3. **Equivalência.** Um resultado não-significativo **não prova** que dois
   métodos são iguais (ausência de evidência ≠ evidência de ausência). Para
   *afirmar* "pode substituir", o frequentista é a ferramenta errada.

### Duas camadas de análise
1. **Descritiva (referência):** **10-fold CV**, reportando **AUC média ± desvio**
   por agregador e dataset. É a tabela que o leitor espera ver.
2. **Inferencial (as afirmações):** **t-test Bayesiano correlacionado + ROPE**.

### O teste Bayesiano correlacionado + ROPE, passo a passo
1. Rodar os dois agregadores **nos mesmos folds** → diferença por fold
   `d_i = AUC_A(i) − AUC_B(i)`.
2. Estimar a **distribuição a posteriori da diferença média** `P(μ | dados)` com
   o **t-test Bayesiano correlacionado** (**Corani & Benavoli, 2015**), que usa a
   **correção de variância de Nadeau & Bengio (2003)** para a sobreposição dos
   treinos (fator `ρ = n_teste / (n_treino + n_teste)`). Isso alarga
   honestamente a incerteza, em vez de fingir independência.
3. Definir a **ROPE** (Region of Practical Equivalence): faixa `[−r, +r]` em AUC
   na qual a diferença é praticamente irrelevante. **Proposta: r = 0.01** (1
   ponto de AUC); justificar no texto e reportar sensibilidade (ex. 0.005, 0.02).
4. Integrar a posterior nas três regiões → **três probabilidades que somam 1**:
   - `P(μ < −r)` = B praticamente melhor,
   - `P(−r ≤ μ ≤ r)` = **praticamente equivalentes**,
   - `P(μ > +r)` = A praticamente melhor.

### Regra de decisão (fixada a priori, reportada)
- **"Pode substituir" (Hopfield ≈ atenção):** afirmamos equivalência se
  `P(rope) ≥ 0.95`.
- **"Melhor que agregação simples" (Hopfield/atenção > mean/max):** afirmamos
  superioridade se `P(direita) ≥ 0.95`.
- Casos intermediários: reportar as três probabilidades sem conclusão forte.

### Panorama entre datasets/condições (opcional, de suporte)
Para um resumo entre os vários problemas (E1–E5 e os pontos de varredura), o
**teste Bayesiano de postos sinalizados** (signed-rank) ou o **hierárquico
correlacionado** (**Benavoli et al., 2017**). *Sem frequentista (Friedman/
Nemenyi/CD) — optamos por só Bayesiano.*

### Ferramenta
**`baycomp`** (janezd/baycomp): `two_on_single(scores_a, scores_b, rope=r)` →
`(p_left, p_rope, p_right)` + gráfico de simplex/triângulo pronto para o paper.

### Exemplo de frase para o artigo
> "Com ROPE de 1 ponto de AUC, obtivemos P(Hopfield ≈ atenção) = 0.96,
> sustentando empiricamente que a atenção pode ser substituída pelo Hopfield;
> e P(Hopfield > mean-pooling) = 0.99, confirmando a vantagem sobre agregação
> simples."

### Referências
- **Dietterich, T. G. (1998).** *Approximate Statistical Tests for Comparing
  Supervised Classification Learning Algorithms.* Neural Computation 10(7):
  1895–1923. https://direct.mit.edu/neco/article-abstract/10/7/1895/6224/
- **Nadeau, C. & Bengio, Y. (2003).** *Inference for the Generalization Error.*
  Machine Learning 52:239–281. https://link.springer.com/article/10.1023/A:1024068626366
- **Demšar, J. (2006).** *Statistical Comparisons of Classifiers over Multiple
  Data Sets.* JMLR 7:1–30. https://jmlr.org/papers/v7/demsar06a.html
- **Corani, G. & Benavoli, A. (2015).** *A Bayesian approach for comparing
  cross-validated algorithms on multiple data sets.* Machine Learning
  100(2–3):285–304. https://link.springer.com/article/10.1007/s10994-015-5486-z
- **Benavoli, A., Corani, G., Demšar, J. & Zaffalon, M. (2017).** *Time for a
  Change: a Tutorial for Comparing Multiple Classifiers Through Bayesian
  Analysis.* JMLR 18(77):1–36. https://jmlr.org/papers/v18/16-305.html
- **baycomp** (implementação): https://github.com/janezd/baycomp

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
| W&B no Kaggle | **online via Kaggle Secret** (`WANDB_API_KEY`); local = offline/disabled |
| Recuperação dos resultados | **W&B API → CSV versionado em `results/`** (artigo reprodutível sem W&B) |
| Relatórios | **markdown por experimento** em `experiments/<nome>/report.md` |
| Ambiente | **local p/ clássicos** (200 bags, CPU), **Kaggle/GPU p/ imagem** e pesados |
| Avaliação/estatística | **10-fold CV** (referência descritiva) + **t-test Bayesiano correlacionado + ROPE** (inferência); **sem NHST/p-valor** |

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

## Próximos passos (plano de ação ordenado para COMEÇAR)

**Marco 1 — pipeline de treino mínimo rodando (1 condição, 1 dataset, local).**
1. **`DataModule`** com **k-fold** + `mil_collate`; expõe os folds para o teste estatístico.
   Para sintéticos: tamanho de bag **fixo** (`var_bag_size=0`), parametrizado por condição.
2. **`training/train.py`** — wiring Hydra→Lightning (hoje stub). *Bloqueador do treino.*
   Validar local no **elephant** com `wandb.mode=disabled`, 1 agregador, ponta-a-ponta.

**Marco 2 — comparação justa dos 4 agregadores numa condição.**
3. Rodar os 4 agregadores nos **mesmos folds/seeds**, HP compartilhados idênticos.
4. **`eval/stats.py`** + dep **`baycomp`**: recebe scores por fold → `(p_left, p_rope, p_right)`.
5. Reportar **nº de params** por agregador (justiça).

**Marco 3 — varreduras e escala.**
6. Configs de experimento (grupo `experiment/`) para E1–E5 com OFAT a partir do baseline.
7. **W&B online** (Kaggle Secret) + `experiments/collect.py` (W&B API → `results/*.csv`).
8. **Kaggle runner** (`kaggle/run.ipynb`) para os experimentos de imagem (GPU).
9. **Relatório** `experiments/<nome>/report.md` por experimento.

> Comece pelo **Marco 1**: é o menor pedaço que prova a pipeline antes de escalar.

## Convenções de experimento (definidas — falta implementar)

### Ambiente
- **Clássicos (elephant/fox/tiger)**: rodam local (CPU), iteração rápida.
- **Imagem (CIFAR/MNIST-bags) e futuros pesados (WSI)**: Kaggle com GPU.
- Dados no Kaggle: MNIST/CIFAR baixam (CIFAR com fallback); clássicos via Figshare ou Kaggle Dataset.

### Logging e recuperação
- W&B **online via Kaggle Secret** (`WANDB_API_KEY`); local roda `wandb.mode=offline/disabled`.
- **Logar por run**: métricas (AUC/acc + curvas), config Hydra completa, seed, commit hash,
  nº de params do agregador, e (quando houver `instance_labels`) a localization AUC.
- **Recuperação**: script `experiments/collect.py` puxa as métricas finais via API do W&B e
  grava `results/<tabela>.csv` (commitado) — fonte reprodutível das tabelas do artigo.
- Checkpoints: salvar o melhor (val/auc) em `checkpoints/` (gitignored).

### Relatórios
- Um `experiments/<nome>/report.md` por experimento, contendo: hipótese, setup,
  **comando Hydra exato + commit hash + seeds**, tabela de resultados, figuras, conclusão.

## Notebooks (o que cada um mostra)
- **01_explore_and_forward** — MNIST-bags: anatomia da bag, forward passo a passo, interpretabilidade.
- **02_explore_tiger_elephant** — clássicos tabulares: por que não são imagens, heatmap + PCA das features.
- **03_explore_cifar_bags** — CIFAR-bags: crops visíveis, testemunha conhecida, forward + interpretabilidade.

## Gotchas (Windows)
- `PYTHONUTF8=1 uv sync --extra hopfield` (cp1252 quebra o build do hflayers).
- Kernel Jupyter registrado como **"Python (hopmil)"**.
- CIFAR-10: host de Toronto pode estar bloqueado → fallback fast.ai + cache automático.
