"""Hydra entrypoint: ``hopmil-train`` / ``python -m hopmil.training.train``.

Composes data + encoder + aggregator + head from configs/, then runs Lightning
with an optional W&B logger. Only the aggregator (``model=``) varies between
runs; the encoder/head/training are fixed so the comparison is a clean ablation.

Override anything from the CLI, e.g.::

    hopmil-train data=elephant model=attention wandb.mode=disabled
    hopmil-train -m model=attention,hopfield,mean,max cv.fold=0,1,2   # sweep
"""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from hopmil.data.datamodule import MILDataModule
from hopmil.models.aggregators import build_aggregator
from hopmil.models.mil_model import MILClassifier
from hopmil.training.factory import build_dataset, build_encoder
from hopmil.training.lightning_module import MILLitModule
from hopmil.utils.seed import seed_everything

# Absolute path so the entrypoint resolves configs regardless of CWD or whether
# it's launched as a console script (relative config_path breaks there).
_CONFIG_DIR = str(Path(__file__).resolve().parents[3] / "configs")


@hydra.main(version_base=None, config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed, deterministic=cfg.trainer.deterministic)

    dataset = build_dataset(cfg.data, cfg.seed)
    encoder = build_encoder(cfg.data, cfg.model.dim)

    # Aggregator-specific kwargs (e.g. attention.hidden, hopfield.beta) come
    # straight from the model config; shared keys are dropped.
    agg_kwargs = {k: v for k, v in cfg.model.items() if k not in ("aggregator", "dim")}
    aggregator = build_aggregator(cfg.model.aggregator, dim=cfg.model.dim, **agg_kwargs)

    model = MILClassifier(encoder, aggregator, num_classes=cfg.data.num_classes, dim=cfg.model.dim)
    lit = MILLitModule(model, lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)

    datamodule = MILDataModule(
        dataset,
        n_folds=cfg.cv.n_folds,
        fold=cfg.cv.fold,
        val_frac=cfg.cv.val_frac,
        batch_size=cfg.loader.batch_size,
        num_workers=cfg.loader.num_workers,
        seed=cfg.seed,
    )

    logger = False
    if cfg.wandb.mode != "disabled":
        from lightning.pytorch.loggers import WandbLogger

        logger = WandbLogger(
            project=cfg.wandb.project,
            offline=(cfg.wandb.mode == "offline"),
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    import lightning as L
    from lightning.pytorch.callbacks import ModelCheckpoint

    ckpt = ModelCheckpoint(monitor="val/auc", mode="max")
    trainer = L.Trainer(logger=logger, callbacks=[ckpt], **cfg.trainer)
    trainer.fit(lit, datamodule=datamodule)
    trainer.test(lit, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
