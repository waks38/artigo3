# hopmil — Hopfield layers for Multiple Instance Learning

Research code for the article investigating **Modern Hopfield layers as an
attention-replacement for instance aggregation in Multiple Instance Learning (MIL)**.

In MIL a sample is a *bag* of instances with a single bag-level label. The
aggregation step — turning a variable set of instance embeddings into one bag
embedding — is where the modeling choice matters. We replace the gated
attention pooling of Ilse et al. (2018) with a Modern Hopfield layer
(Ramsauer et al., 2020, *"Hopfield Networks is All You Need"*), which
generalizes attention via energy-based associative retrieval, and compare
predictive performance and interpretability.

## Setup

```bash
uv sync --extra dev                              # creates .venv (no Hopfield aggregator)
PYTHONUTF8=1 uv sync --extra dev --extra hopfield  # add the Hopfield aggregator (builds hflayers from git)
```

## Run

```bash
# compare all 4 aggregators on one dataset (repeated k-fold + Bayesian test):
uv run hopmil-compare data=elephant                          # classics: local/CPU
uv run hopmil-compare -m data=elephant,fox,tiger n_jobs=-2   # all classics, parallel folds
# single training run:
uv run hopmil-train data=mnist_bags model=hopfield
uv run pytest                                                 # tests
```

Image datasets (mnist/fashion/cifar bags + real histopathology colon/UCSB) run on
**Kaggle GPU** — see `kaggle/run.ipynb`. Read `docs/NEXT_STEPS.md` to continue.

## Layout

| Path | Purpose |
|------|---------|
| `src/hopmil/data/` | `Bag`/`MILDataset` + loaders (tabular, synthetic bags, real histo) |
| `src/hopmil/models/aggregators.py` | the swappable core: mean/max/attention/Hopfield |
| `src/hopmil/models/mil_model.py` | encoder → aggregator → head |
| `src/hopmil/training/` | Lightning module + Hydra entrypoints (`train`, `factory`) |
| `src/hopmil/eval/` | `compare.py` (4-way comparison), `stats.py` (baycomp), `metrics.py` |
| `configs/` | Hydra config groups (`data/`, `model/`) + `compare.yaml` |
| `kaggle/` | GPU runner notebook for the image experiments |
| `docs/` | `NEXT_STEPS.md` (handoff), `ROADMAP.md`, `METHODOLOGY.md` |

## References

- Ilse, Tomczak, Welling (2018). *Attention-based Deep Multiple Instance Learning.*
- Ramsauer et al. (2020). *Hopfield Networks is All You Need.* — `ml-jku/hopfield-layers`
