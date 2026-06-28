# Metodologia — comparação de agregadores MIL

> Documento de referência da metodologia do artigo, pronto para virar texto.
> Implementação: `src/hopmil/eval/compare.py` (orquestração) + `eval/stats.py`
> (camada inferencial). Decisões de planejamento em `docs/ROADMAP.md`.

## 1. Objetivo e tese

Comparar quatro agregadores de embeddings de bag em MIL — **mean-pooling**,
**max-pooling**, **attention** (Ilse et al., 2018) e **Hopfield** (Ramsauer et
al., 2020) — mantendo **encoder + head fixos** por dataset, de modo que o
agregador seja a **única** variável.

**Tese (empírica, só métricas):**
1. **Equivalência** — *atenção pode ser substituída por Hopfield*: os dois
   empatam nas métricas. (A equivalência matemática já está provada no paper
   base; aqui só damos a evidência empírica — sem análise mecanística do β.)
2. **Superioridade** — atenção e Hopfield **superam** agregação simples
   (mean/max).

Papéis: atenção = referência que o Hopfield substitui; mean/max = piso a bater.

## 2. Protocolo de comparação justa ("maçã com maçã")

- **Unidade de comparação = uma condição fixa.** Para os clássicos
  (elephant/fox/tiger) a condição é o próprio dataset; para sintéticos é a tupla
  `(tamanho da bag, nº testemunhas, nº de bags)`. Os 4 agregadores são comparados
  **só dentro da mesma condição**.
- **Mesmos dados, mesmos folds.** Os 4 agregadores percorrem **exatamente as
  mesmas partições** (mesmo split em cada `(repetição, fold)`) → comparação
  **pareada**. É isso que valida o teste estatístico pareado — e não depende de
  controlar o tamanho da bag: a variabilidade das bags é idêntica para os 4.
- **Hiperparâmetros compartilhados idênticos** (encoder, head, `dim`, lr, épocas,
  batch, folds, seeds). Os **específicos** de cada agregador (β/cabeças do
  Hopfield; hidden da atenção) ficam em **default fixo, sem tuning por
  agregador** — mean/max não têm nenhum.
- **Nº de parâmetros reportado** por agregador (do agregador isolado e do modelo
  total), como parte da justiça da comparação.

## 3. Avaliação — validação cruzada repetida

Duas camadas:

1. **Descritiva (referência):** **k-fold estratificado repetido** (R×k; padrão
   **10×10 = 100 medições/agregador**), reportando **AUC média ± desvio** por
   agregador. É a tabela que o leitor espera.
2. **Inferencial (as afirmações):** **t-test Bayesiano correlacionado + ROPE**
   sobre os scores por fold (§5).

**Esquema repetido:** o conjunto de bags é construído **uma vez**; cada
repetição `r` reembaralha a partição em folds (`split seed = seed + r`).
Estratificação pelo label da bag em todos os folds. O score por fold é a **AUC
no fold de teste** (o fold retido), com seleção do melhor modelo por
early-stopping em `val/auc` (split treino/val interno ao conjunto de treino do
fold).

## 4. Por que NÃO usar teste frequentista (p-valor)

1. **Correlação da CV.** Em k-fold os treinos se sobrepõem → os scores dos folds
   são **correlacionados**, não independentes. O t-test pareado assume
   independência; sem ela **subestima a variância** e infla o erro tipo I
   (Dietterich, 1998).
2. **Mais folds não resolve.** Aumentar `k` aumenta a sobreposição dos treinos
   (mais correlação) e reduz a variância **artificialmente** — não gera
   informação independente.
3. **Equivalência.** Um resultado não-significativo **não prova** igualdade
   (ausência de evidência ≠ evidência de ausência). Para *afirmar* "pode
   substituir", o frequentista é a ferramenta errada.

## 5. Teste Bayesiano correlacionado + ROPE (passo a passo)

1. Rodar A e B **nos mesmos folds** → diferença por fold `d_i = AUC_A(i) − AUC_B(i)`.
2. Estimar a **posterior da diferença média** `P(μ | dados)` com o **t-test
   Bayesiano correlacionado** (Corani & Benavoli, 2015), que aplica a **correção
   de variância de Nadeau & Bengio (2003)** para a sobreposição dos treinos
   (fator `ρ = n_teste/(n_treino+n_teste)`). Isso **alarga honestamente** a
   incerteza em vez de fingir independência. O nº de repetições (`runs`) informa
   a estrutura da CV ao estimador.
3. **ROPE** (Region of Practical Equivalence): faixa `[−r, +r]` em AUC em que a
   diferença é praticamente irrelevante. **Padrão `r = 0.01`** (1 ponto de AUC);
   reportar sensibilidade (ex. 0.005, 0.02).
4. Integrar a posterior nas três regiões → **três probabilidades que somam 1**:
   - `P(A melhor)` = `P(μ > +r)`,
   - `P(rope)` = `P(−r ≤ μ ≤ r)` = **praticamente equivalentes**,
   - `P(B melhor)` = `P(μ < −r)`.

> **Convenção de implementação** (importante): `baycomp.two_on_single(x, y)`
> devolve `(P(x melhor), P(rope), P(y melhor))`. O wrapper `eval/stats.py` mapeia
> isso para `p_a_better / p_rope / p_b_better` com nomes explícitos para evitar
> inversão de veredito.

### Regra de decisão (fixada a priori, reportada)
- **"Pode substituir" (Hopfield ≈ atenção):** afirmamos equivalência se
  `P(rope) ≥ 0.95`.
- **"Melhor que agregação simples":** afirmamos superioridade se
  `P(A melhor) ≥ 0.95` (resp. B).
- Caso contrário: **inconclusivo** — reportar as três probabilidades sem
  conclusão forte.

### Pares testados
`hopfield vs attention` (equivalência), e `hopfield/attention vs mean/max`
(superioridade). Configurável em `compare.yaml: pairs`.

## 6. O que é logado e como recuperar

- **W&B**: **um run por dataset** (projeto `artigo-3`, run = nome do dataset).
  Logamos: tabela de scores por fold, AUC média±desvio por agregador, nº de
  parâmetros, e as probabilidades Bayesianas por par.
- **CSVs versionados** em `results/<dataset>_{folds,summary,bayesian}.csv` —
  fonte reprodutível das tabelas do artigo (independe do dashboard).
- **Terminal**: progresso por fold com média corrente e ETA; ao final, o resumo
  descritivo e os vereditos Bayesianos.

## 7. Como rodar

```bash
# clássicos: local/CPU, um dataset por vez (run W&B com o nome do dataset)
hopmil-compare data=elephant
hopmil-compare data=fox
hopmil-compare data=tiger

# checagem rápida (poucos folds/épocas, sem W&B):
hopmil-compare data=elephant wandb.mode=disabled cv.n_repeats=2 cv.n_folds=3 trainer.max_epochs=10

# sensibilidade da ROPE / esquema de CV:
hopmil-compare data=tiger rope=0.005
hopmil-compare data=tiger cv.n_repeats=5
```

Pré-requisitos: `WANDB_API_KEY` configurado (`wandb login`) para `mode=online`;
extra Hopfield instalado (`PYTHONUTF8=1 uv sync --extra hopfield`).

## 8. Condição sintética fixa (sem varreduras) e divergências declaradas

Decidimos **não fazer varreduras OFAT** (os antigos E3/E4/E5): o ganho esperado
de variar tamanho de bag / nº de bags era pequeno e custoso. Os sintéticos rodam
numa **única condição fixa e controlada**:

- **tamanho de bag fixo = 15** (sem variância);
- **exatamente 1 testemunha** numa bag positiva (testemunha esparsa, 1/15 — o
  regime em que atenção/Hopfield devem superar mean/max), zero nas negativas;
- **classes balanceadas** (50% positivas);
- **nº de bags fixo** (250).

Divergências vs. MNIST-bags original (Ilse et al.): eles usam tamanho de bag
variável e nº de testemunhas aleatório; nós fixamos ambos para um contraste
limpo e reprodutível. Declarar no artigo. Nos clássicos e na histopatologia real
as bags são naturais (não se mexe nelas).

O catálogo de experimentos fica: **E1 (clássicos tabulares)** e **E2 (imagem:
MNIST/Fashion/CIFAR-bags + colon/UCSB reais)** — cada um uma condição fixa,
comparando os 4 agregadores.

## Referências
- Dietterich, T. G. (1998). *Approximate Statistical Tests for Comparing
  Supervised Classification Learning Algorithms.* Neural Computation 10(7).
- Nadeau, C. & Bengio, Y. (2003). *Inference for the Generalization Error.*
  Machine Learning 52.
- Corani, G. & Benavoli, A. (2015). *A Bayesian approach for comparing
  cross-validated algorithms on multiple data sets.* Machine Learning 100.
- Benavoli, A., Corani, G., Demšar, J. & Zaffalon, M. (2017). *Time for a Change:
  a Tutorial for Comparing Multiple Classifiers Through Bayesian Analysis.* JMLR
  18(77).
- baycomp: https://github.com/janezd/baycomp
