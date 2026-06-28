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
→ calls the entrypoint).

**Continuing the work?** Read `docs/NEXT_STEPS.md` first (handoff with the next
task — Marco 1 — spelled out), then `docs/ROADMAP.md` for the full plan,
experimental design, and Bayesian methodology.

## Commands

```bash
uv sync --extra dev                # install env (.venv); for the Hopfield aggregator add --extra hopfield
uv run hopmil-train model=attention data=mnist_bags    # one training run (Hydra entrypoint)
uv run hopmil-train -m model=attention,hopfield seed=0,1,2   # multirun sweep
uv run hopmil-compare data=elephant    # Marco 2: compare all 4 aggregators (repeated k-fold + Bayesian)
uv run hopmil-compare data=fox wandb.mode=disabled cv.n_repeats=2 cv.n_folds=3   # quick check
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

### Datasets (same pipeline; tabular + synthetic-image + real-histopathology)

| Loader | Instance | `instance_labels`? | Role |
|---|---|---|---|
| `data/mnist_bags.py` `MNISTBags` | 28×28 digit | yes (target digit) | sanity check + interpretability |
| `data/fashion_mnist_bags.py` `FashionMNISTBags` | 28×28 grayscale | yes (target class) | harder-texture synthetic image MIL |
| `data/cifar_bags.py` `CIFARBags` | 32×32 RGB image | yes (target class) | realistic image MIL + interpretability |
| `data/colon_cancer.py` `ColonCancerBags` | 27×27 RGB patch (nucleus) | yes (nucleus class) | **real** histopathology + interpretability |
| `data/ucsb_breast.py` `UCSBBreastBags` | 32×32 RGB grid patch | **None** | **real** histopathology, bag-label only |
| `data/classic.py` `ClassicMIL` | 230-d feature vector | **None** | standard tabular benchmark (elephant/fox/tiger) |

Real-histo data is **not auto-downloadable** (Warwick/UCSB hosts gated/down):
download once (Kaggle mirrors) under each loader's `root`; the loaders parse the
documented layout and raise a clear hint if files are missing. Colon Cancer:
bag=image, instance=27×27 patch per nucleus, positive iff ≥1 epithelial nucleus
(witnesses known → localization applies). UCSB Breast: bag=image, instance=32×32
grid patch (near-white tiles dropped), label malignant/benign from the filename.

Synthetic bags (MNIST/Fashion/CIFAR) use one **fixed controlled condition** (no
OFAT sweeps): shared assembly in `data/synthetic.py::make_witness_bags` builds
**fixed-size bags (15)**, **exactly one witness** in a positive bag, **balanced**
(50%), fixed `num_bags` (250) — seeded/reproducible. `ClassicMIL` parses PRTools
`.mat` files (Andrews 2002, via Figshare), grouping rows by `ident.milbag`; no
true instance labels.

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
- **Marco 1 done**: `data/datamodule.py` (`MILDataModule`, stratified k-fold +
  train/val/test + `mil_collate`) and `training/train.py` (full Hydra→Lightning
  wiring, runs `fit`+`test`). Configs `model/{mean,max}.yaml` added; `cv`/`loader`
  groups in `config.yaml`; synthetic `var_bag_size=0` (fixed bag size). Verified:
  `hopmil-train data=elephant model=attention wandb.mode=disabled` trains end-to-end.
- Tests pass (`pytest -m "not slow"`), incl. `test_datamodule.py` (fold invariants).
- **Marco 2 done**: `eval/compare.py` (entrypoint `hopmil-compare`) runs the 4
  aggregators through the *same* repeated stratified k-fold (paired), default
  10×10; `eval/stats.py` wraps `baycomp.two_on_single` → `BayesResult`
  (`p_a_better/p_rope/p_b_better` — orientation locked by `test_stats.py`). One
  W&B run per dataset (project `artigo-3`, run=dataset); CSVs to
  `results/<dataset>_{folds,summary,bayesian}.csv`; clean terminal progress.
  Methodology written in `docs/METHODOLOGY.md`. Builders shared via
  `training/factory.py`. Deps `baycomp`, `joblib` added.
- **Full metric suite**: every fold logs AUROC, AUPRC, accuracy, balanced_acc, F1,
  precision, recall, MCC, brier, log_loss, and (synthetic/colon) localization AUC
  (`eval/metrics.py`); all go to the CSVs/W&B and the Bayesian test runs on every
  well-defined metric. `n_jobs` parallelizes folds (joblib).
- **6 image datasets + extras**: added `fashion_mnist_bags`, and real
  histopathology `colon_cancer` (witnesses → localization) and `ucsb_breast`
  (bag-label only). Parsers tested against synthetic fixtures (`test_histopath.py`).

- **Synthetic design simplified**: dropped the OFAT sweeps (old E3–E5); synthetic
  bags now a single fixed condition (bag=15, 1 witness, balanced) via
  `data/synthetic.py`. Study = **E1 (classics, local/CPU) + E2 (image, Kaggle/GPU)**.
- **Kaggle runner**: `kaggle/run.ipynb` (+ `kaggle/README.md`) clones the repo,
  installs `[hopfield]`, reads `WANDB_API_KEY` from a Secret, runs `compare` on GPU
  for the image datasets (skips real-histo if its `root` isn't mounted), saves CSVs
  to `/kaggle/working`. `QUICK=True` → tiny smoke into project `artigo-3-smoke`.

**Pending (next steps, in rough order):**
1. Create the **GitHub repo** (Kaggle clones from it) and upload colon/UCSB as
   **Kaggle Datasets**; then run `kaggle/run.ipynb` (QUICK first, then full).
2. **Run E1** locally if not done: `hopmil-compare -m data=elephant,fox,tiger n_jobs=-2`.
3. Analysis/figures: AUC mean±std tables, baycomp simplex plots, per-experiment
   `experiments/<name>/report.md`.
5. `results/` is gitignored but CLAUDE/methodology call the CSVs the committed
   paper source — reconcile (force-add results CSVs or adjust `.gitignore`).
