# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code for an **article** (not a product): a **comparative study of
bag-embedding aggregators for Multiple Instance Learning (MIL)**. We compare
four aggregators — **Modern Hopfield** (Ramsauer et al., 2020), **attention**
(Ilse et al., 2018), **max-pooling**, and **average-pooling** — under one fixed
encoder + head per dataset, so the aggregator is the only thing that varies.
The goal is a fair, reproducible comparison (performance + interpretability),
not declaring a single winner a priori. Decisions should optimize for
reproducibility, clean ablations, and figure-generating experiments.

**Compute target: Kaggle Notebooks** (free GPU). Code must run both locally and
on Kaggle (workflow: push to GitHub → Kaggle notebook clones + `pip install -e .`
→ calls the entrypoint). See `docs/ROADMAP.md` for the full plan and open decisions.

## Commands

```bash
uv sync --extra dev                # install env (.venv); for the Hopfield aggregator add --extra hopfield
uv run hopmil-train model=attention data=mnist_bags    # one run (Hydra entrypoint) — train.py still a stub
uv run hopmil-train -m model=attention,hopfield seed=0,1,2   # multirun sweep
uv run pytest                      # all tests
uv run pytest tests/test_aggregators.py::test_unknown_aggregator_raises   # single test
uv run pytest -m "not slow"        # skip the hopfield/import-heavy tests
uv run ruff check . && uv run ruff format .
# execute a notebook headless (uses the registered venv kernel):
uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=hopmil notebooks/01_explore_and_forward.ipynb
```

Always invoke tools through `uv run` — the project lives in `.venv` managed by uv,
not a globally-activated env. Python is pinned to 3.11 (torch lacks 3.14 wheels).

**Windows gotchas:**
- The `hopfield` extra builds `hopfield-layers` from a git sdist whose `setup.py`
  is misread under cp1252. Always sync with UTF-8 forced: `PYTHONUTF8=1 uv sync --extra hopfield`.
  Plain `uv sync --extra dev` (no hopfield) is unaffected.
- The `.venv` is registered as the Jupyter kernel **"Python (hopmil)"** — select it in the IDE.
- CIFAR-10's canonical host (cs.toronto.edu) may be unreachable; `CIFARBags` falls
  back to a fast.ai mirror and caches to `data/raw/cifar10_*.pt` automatically.

## Architecture — the one thing to understand

The entire experimental design hinges on a **single swappable component**: the
*aggregator*. Everything else (encoder, classifier head, training loop, data)
is held fixed so that comparing the four aggregators is a clean ablation.

Data flow per bag: `encoder` → `aggregator` → `head`.

- `src/hopmil/models/aggregators.py` — `Aggregator` base class defines the
  contract every aggregator must honor: `(n_instances, dim) -> (bag_embedding (dim,), weights (n,)|None)`.
  Implementations: `MeanPool`, `MaxPool`, `AttentionMIL` (Ilse), `HopfieldMIL`
  (wraps `hflayers.HopfieldPooling`). Add new aggregators here and register them
  in `build_aggregator`; `test_aggregators.py` enforces the contract automatically.
  `mean`/`max` return `weights=None` (no per-instance weight to interpret).
- `src/hopmil/models/encoders.py` — instance encoders φ, held FIXED across the four
  aggregators. `MLPEncoder` for tabular bags; `CNNEncoder` for image bags
  (defaults to 1×28×28 MNIST; pass `in_channels=3, image_size=32` for CIFAR — the
  flatten size is computed from a dummy pass, so one class serves both).
- `src/hopmil/models/mil_model.py` — `MILClassifier` composes encoder + aggregator
  + head. Takes a **list** of bags, returns `(logits, list_of_weights)`. Keep the
  three components orthogonal; never bake aggregation logic into the model.
- `src/hopmil/data/mil_dataset.py` — `Bag`/`MILDataset`/`mil_collate`. Bags are
  variable-length and **not padded**; a batch is a *list* of instance tensors.
  `Bag.instance_labels` is populated only for synthetic data (MNIST/CIFAR bags),
  and drives the interpretability eval (`eval/metrics.py::instance_localization_auc`).

### Datasets (3 families, same pipeline)

| Loader | Instance | Visible? | `instance_labels`? | Role |
|---|---|---|---|---|
| `data/mnist_bags.py` `MNISTBags` | 28×28 digit | yes | yes (target digit) | sanity check + interpretability |
| `data/cifar_bags.py` `CIFARBags` | 32×32 RGB image | yes | yes (target class) | realistic image MIL + interpretability |
| `data/classic.py` `ClassicMIL` | 230-d feature vector | no | **None** | standard tabular benchmark (elephant/fox/tiger) |

Synthetic bags (MNIST/CIFAR) sample bag sizes from a seeded Gaussian and assemble
instances from the base dataset; the bag is positive iff it contains ≥1 target
instance. `ClassicMIL` parses PRTools `.mat` files (Andrews 2002, via Figshare),
grouping rows by `ident.milbag`; there are no true instance labels.

## Conventions

- **Configs are Hydra**, composed from `configs/` groups (`data/`, `model/`). Don't
  hardcode hyperparameters in Python — add a config and override via CLI. The
  `model=` group selects the aggregator; `data=` selects the dataset.
- **Experiment tracking is W&B** (project `hopmil`): **online via Kaggle Secret**
  (`WANDB_API_KEY`) on Kaggle; `wandb.mode=offline` locally/CI. Final metrics are
  pulled back via the W&B API into a **committed CSV in `results/`** (the
  reproducible source for the paper tables) — don't rely on the dashboard alone.
- **Experiments**: classics (elephant/fox/tiger) run **locally** (tiny, CPU);
  image bags run on **Kaggle/GPU**. Each experiment gets an
  `experiments/<name>/report.md` (hypothesis, exact Hydra command + commit hash +
  seeds, results table, figures, conclusion).
- **Lightning** owns the training loop, seeding, and checkpointing — put metrics in
  the `LightningModule`, not in ad-hoc scripts.
- `data/`, `results/`, `wandb/`, checkpoints are **gitignored**; `uv.lock` is committed.
- Heavy git dependency `hflayers` is imported **lazily** inside `HopfieldMIL` so that
  fast tests and non-Hopfield runs don't pay for it (it's an optional extra).

## Status

**Implemented & verified (forward pass runs end-to-end):**
- Aggregator contract + all 4 aggregators; `MILClassifier`; `MLPEncoder`/`CNNEncoder`.
- `LightningModule` (`training/lightning_module.py`) with BCE loss + AUROC/accuracy.
- All 3 dataset loaders load real data (MNIST/CIFAR/elephant·fox·tiger).
- Notebooks: `01` (MNIST-bags), `02` (tiger/elephant tabular), `03` (CIFAR-bags) —
  each walks data → encoder → aggregator → head with shapes and figures.
- Tests pass (`pytest -m "not slow"`).

**Pending (next steps, in rough order):**
1. `training/train.py` — the Hydra→Lightning wiring is still a stub raising
   `NotImplementedError`. This is the blocker for actually training/comparing.
2. A `DataModule` (dataset + train/val/test split + `mil_collate`).
3. W&B logging conventions + how to recover results for the paper (offline sync from Kaggle).
4. Kaggle runner notebook (`kaggle/`).
5. Per-experiment report convention (`experiments/`).
