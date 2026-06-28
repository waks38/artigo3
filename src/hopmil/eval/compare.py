"""Fair comparison of the four aggregators on ONE dataset (Marco 2).

Runs ``mean / max / attention / hopfield`` through the *same* repeated stratified
k-fold partitions (paired), with identical shared hyperparameters and the fixed
encoder/head — so the aggregator is the only thing that varies. Per fold we log
the **full metric suite** (AUROC, AUPRC, accuracy, F1, ..., and localization AUC
on synthetic data); the thesis verdict uses AUROC but everything is stored so the
paper can pick. Per-fold scores feed the correlated Bayesian t-test + ROPE.

Entrypoint::

    hopmil-compare data=elephant                       # one W&B run "elephant"
    hopmil-compare data=mnist_bags cv.n_repeats=1 n_jobs=-1   # plain 10-fold, all cores
    hopmil-compare data=fox wandb.mode=offline n_jobs=4

Design (see docs/METHODOLOGY.md):
- **Repeated CV**: the dataset is built ONCE; each repeat reshuffles the fold
  partition (split seed = seed + repeat). All four aggregators see the exact same
  (repeat, fold) split → valid pairing.
- **No per-aggregator tuning**: aggregator-specific HPs are fixed defaults.
- **Parallelism**: fits are independent → ``n_jobs`` runs them across CPU cores
  (joblib); each worker is seeded by its split seed, so results are deterministic
  regardless of completion order.
- W&B: one run per dataset (name = dataset); fold table, per-aggregator mean±std
  for every metric, param counts, and Bayesian verdicts all logged there.
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from hopmil.data.datamodule import MILDataModule
from hopmil.eval.stats import compare_bayesian
from hopmil.models.aggregators import build_aggregator
from hopmil.models.mil_model import MILClassifier
from hopmil.training.factory import build_dataset, build_encoder
from hopmil.training.lightning_module import MILLitModule
from hopmil.utils.seed import seed_everything

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_DIR = str(_PROJECT_ROOT / "configs")
_PRIMARY = "auroc"  # metric used for the printed verdict and progress line

log = logging.getLogger("hopmil.compare")


def _quiet_lightning() -> None:
    for name in (
        "lightning.pytorch",
        "lightning.pytorch.utilities.rank_zero",
        "lightning.pytorch.accelerators.cuda",
        "pytorch_lightning",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)


def _aggregator_kwargs(cfg: DictConfig, agg: str) -> dict:
    specifics = OmegaConf.to_container(cfg.model_specifics, resolve=True) or {}
    return dict(specifics.get(agg, {}) or {})


def _run_one_fold(
    dataset, cfg: DictConfig, agg: str, split_seed: int, fold: int, ckpt_dir: Path
) -> dict:
    """Train one aggregator on one fold; return the full test-metric dict."""
    import lightning as L
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    seed_everything(split_seed, deterministic=cfg.trainer.deterministic)

    encoder = build_encoder(cfg.data, cfg.dim)
    aggregator = build_aggregator(agg, dim=cfg.dim, **_aggregator_kwargs(cfg, agg))
    model = MILClassifier(encoder, aggregator, num_classes=cfg.data.num_classes, dim=cfg.dim)
    lit = MILLitModule(model, lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)

    dm = MILDataModule(
        dataset,
        n_folds=cfg.cv.n_folds,
        fold=fold,
        val_frac=cfg.cv.val_frac,
        batch_size=cfg.loader.batch_size,
        num_workers=cfg.loader.num_workers,
        seed=split_seed,
    )

    tag = f"{agg}_s{split_seed}_f{fold}"
    ckpt = ModelCheckpoint(
        dirpath=str(ckpt_dir), filename=tag, monitor="val/auc", mode="max", save_top_k=1
    )
    es = EarlyStopping(
        monitor=cfg.early_stopping.monitor,
        mode=cfg.early_stopping.mode,
        patience=cfg.early_stopping.patience,
    )
    trainer = L.Trainer(
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        callbacks=[ckpt, es],
        **cfg.trainer,
    )
    trainer.fit(lit, datamodule=dm)
    result = trainer.test(lit, datamodule=dm, ckpt_path="best", verbose=False)

    best = ckpt.best_model_path
    if best and Path(best).exists():  # don't accumulate thousands of checkpoints
        Path(best).unlink()

    return {k[len("test/") :]: float(v) for k, v in result[0].items() if k.startswith("test/")}


def _fit_task(dataset, cfg, agg, split_seed, r, f, ckpt_dir) -> dict:
    """Picklable worker for joblib: one fold -> one labelled metric row."""
    import torch

    torch.set_num_threads(1)  # avoid oversubscription when n_jobs > 1
    metrics = _run_one_fold(dataset, cfg, agg, split_seed, f, ckpt_dir)
    return {"aggregator": agg, "repeat": r, "fold": f, **metrics}


def _count_params(cfg: DictConfig, agg: str) -> tuple[int, int]:
    aggregator = build_aggregator(agg, dim=cfg.dim, **_aggregator_kwargs(cfg, agg))
    n_agg = sum(p.numel() for p in aggregator.parameters())
    model = MILClassifier(
        build_encoder(cfg.data, cfg.dim),
        aggregator,
        num_classes=cfg.data.num_classes,
        dim=cfg.dim,
    )
    return n_agg, sum(p.numel() for p in model.parameters())


def _force_utf8_streams() -> None:
    """Windows consoles default to cp1252, which can't encode the glyphs we log
    (→, ≈, ±). Reconfigure the live stdout/stderr objects to UTF-8 (Hydra's log
    handlers hold references to these same objects, so this fixes them too);
    errors='replace' guarantees no crash even on a pure-cp1252 console."""
    import sys

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


@hydra.main(version_base=None, config_path=_CONFIG_DIR, config_name="compare")
def main(cfg: DictConfig) -> None:
    _force_utf8_streams()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    _quiet_lightning()

    import wandb

    dataset_name = cfg.data.name
    aggregators = list(cfg.aggregators)
    n_repeats, n_folds, n_jobs = cfg.cv.n_repeats, cfg.cv.n_folds, cfg.n_jobs
    total = n_repeats * n_folds * len(aggregators)

    log.info("=" * 72)
    log.info("COMPARAÇÃO DE AGREGADORES — dataset=%s", dataset_name)
    log.info(
        "agregadores=%s | %dx%d-fold = %d fits/agg | total=%d | n_jobs=%s",
        aggregators,
        n_repeats,
        n_folds,
        n_repeats * n_folds,
        total,
        n_jobs,
    )
    log.info(
        "ROPE=+/-%.3f AUC | regra: P>=%.2f | metrica-tese=%s | W&B=%s/%s",
        cfg.rope,
        cfg.decision_threshold,
        _PRIMARY,
        cfg.wandb.project,
        cfg.wandb.mode,
    )
    log.info("=" * 72)

    run = wandb.init(
        project=cfg.wandb.project,
        name=dataset_name,
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    dataset = build_dataset(cfg.data, cfg.seed)
    ckpt_dir = _PROJECT_ROOT / "checkpoints" / f"compare_{dataset_name}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # task list in (repeat, fold, aggregator) order; split seed = seed + repeat
    tasks = [
        (agg, r, f, cfg.seed + r)
        for r in range(n_repeats)
        for f in range(n_folds)
        for agg in aggregators
    ]

    rows: list[dict] = []
    running: dict[str, list[float]] = {a: [] for a in aggregators}
    done = 0
    t0 = time.time()

    def _on_result(row: dict) -> None:
        nonlocal done
        done += 1
        rows.append(row)
        running[row["aggregator"]].append(row.get(_PRIMARY, float("nan")))
        arr = running[row["aggregator"]]
        elapsed = time.time() - t0
        eta = elapsed / done * (total - done)
        log.info(
            "[%s] %-9s rep %02d/%d fold %02d/%d  %s=%.4f  (media %.4f+/-%.4f n=%d)  "
            "[%d/%d, ETA %dm%02ds]",
            dataset_name,
            row["aggregator"],
            row["repeat"] + 1,
            n_repeats,
            row["fold"] + 1,
            n_folds,
            _PRIMARY,
            row.get(_PRIMARY, float("nan")),
            float(np.nanmean(arr)),
            float(np.nanstd(arr)),
            len(arr),
            done,
            total,
            int(eta // 60),
            int(eta % 60),
        )

    if n_jobs in (1, None):
        for agg, r, f, ss in tasks:
            _on_result(_fit_task(dataset, cfg, agg, ss, r, f, ckpt_dir))
    else:
        from joblib import Parallel, delayed

        gen = Parallel(n_jobs=n_jobs, return_as="generator")(
            delayed(_fit_task)(dataset, cfg, agg, ss, r, f, ckpt_dir) for agg, r, f, ss in tasks
        )
        for row in gen:
            _on_result(row)

    # metric columns = everything that isn't an identifier
    metric_keys = [k for k in rows[0] if k not in ("aggregator", "repeat", "fold")]
    scores = {
        a: {m: np.array([r[m] for r in rows if r["aggregator"] == a]) for m in metric_keys}
        for a in aggregators
    }

    # ---- descriptive summary: mean±std for EVERY metric ----------------------
    log.info("-" * 72)
    log.info(
        "RESUMO DESCRITIVO (%d medicoes/agg) - metrica-tese em destaque: %s",
        n_repeats * n_folds,
        _PRIMARY,
    )
    summary_rows = []
    for agg in aggregators:
        n_par_agg, n_par_total = _count_params(cfg, agg)
        prim = scores[agg][_PRIMARY]
        log.info(
            "  %-9s  %s=%.4f+/-%.4f   params(agg)=%d  params(total)=%d",
            agg,
            _PRIMARY,
            np.nanmean(prim),
            np.nanstd(prim),
            n_par_agg,
            n_par_total,
        )
        row = {"aggregator": agg, "n_params_aggregator": n_par_agg, "n_params_total": n_par_total}
        for m in metric_keys:
            row[f"{m}_mean"] = float(np.nanmean(scores[agg][m]))
            row[f"{m}_std"] = float(np.nanstd(scores[agg][m]))
            wandb.summary[f"{agg}/{m}_mean"] = row[f"{m}_mean"]
            wandb.summary[f"{agg}/{m}_std"] = row[f"{m}_std"]
        wandb.summary[f"{agg}/n_params_aggregator"] = n_par_agg
        wandb.summary[f"{agg}/n_params_total"] = n_par_total
        summary_rows.append(row)

    # ---- inferential layer: Bayesian test on every well-defined metric -------
    log.info("-" * 72)
    log.info("VEREDITO BAYESIANO (correlated t-test + ROPE=+/-%.3f, runs=%d)", cfg.rope, n_repeats)
    bayes_rows = []
    for a, b in cfg.pairs:
        for m in metric_keys:
            sa, sb = scores[a][m], scores[b][m]
            if np.isnan(sa).any() or np.isnan(sb).any():
                continue  # metric undefined on some fold (e.g. localization on tabular)
            res = compare_bayesian(sa, sb, name_a=a, name_b=b, rope=cfg.rope, runs=n_repeats)
            row = {"metric": m, **res.as_row()}
            bayes_rows.append(row)
            if m == _PRIMARY:
                log.info(
                    "  P(%s>)=%.3f  P(rope)=%.3f  P(%s>)=%.3f  ->  %s",
                    a,
                    res.p_a_better,
                    res.p_rope,
                    b,
                    res.p_b_better,
                    res.verdict(cfg.decision_threshold),
                )
                wandb.summary[f"bayes/{a}_vs_{b}/p_a_better"] = res.p_a_better
                wandb.summary[f"bayes/{a}_vs_{b}/p_rope"] = res.p_rope
                wandb.summary[f"bayes/{a}_vs_{b}/p_b_better"] = res.p_b_better

    # ---- W&B tables + committed CSVs -----------------------------------------
    wandb.log(
        {
            "fold_scores": wandb.Table(
                columns=list(rows[0]), data=[list(r.values()) for r in rows]
            ),
            "summary": wandb.Table(
                columns=list(summary_rows[0]), data=[list(r.values()) for r in summary_rows]
            ),
            "bayesian": wandb.Table(
                columns=list(bayes_rows[0]), data=[list(r.values()) for r in bayes_rows]
            ),
        }
    )

    results_dir = _PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    def _write_csv(path: Path, data: list[dict]) -> None:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)

    _write_csv(results_dir / f"{dataset_name}_folds.csv", rows)
    _write_csv(results_dir / f"{dataset_name}_summary.csv", summary_rows)
    _write_csv(results_dir / f"{dataset_name}_bayesian.csv", bayes_rows)

    log.info("-" * 72)
    log.info("Métricas/fold: %s", ", ".join(metric_keys))
    log.info(
        "CSVs em results/%s_{folds,summary,bayesian}.csv | concluído em %.1f min",
        dataset_name,
        (time.time() - t0) / 60,
    )
    run.finish()


if __name__ == "__main__":
    main()
