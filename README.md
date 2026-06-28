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
uv sync --extra dev          # creates .venv and installs everything (incl. hopfield-layers from git)
```

## Run

```bash
uv run hopmil-train model=hopfield data=musk1            # single run
uv run hopmil-train -m model=attention,hopfield seed=0,1,2   # sweep over Hydra
uv run pytest                                            # tests (skips slow/hopfield by default config)
```

## Layout

| Path | Purpose |
|------|---------|
| `src/hopmil/data/` | `Bag`/`MILDataset` abstractions + dataset loaders |
| `src/hopmil/models/aggregators.py` | the swappable core: mean/max/attention/Hopfield |
| `src/hopmil/models/mil_model.py` | encoder → aggregator → head |
| `src/hopmil/training/` | Lightning module + Hydra entrypoint |
| `configs/` | Hydra config groups (`data/`, `model/`) |
| `experiments/` | reproducible experiment scripts & paper figures |

## References

- Ilse, Tomczak, Welling (2018). *Attention-based Deep Multiple Instance Learning.*
- Ramsauer et al. (2020). *Hopfield Networks is All You Need.* — `ml-jku/hopfield-layers`
