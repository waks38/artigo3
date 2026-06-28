"""Hydra entrypoint: ``hopmil-train`` / ``python -m hopmil.training.train``.

Composes data + model + aggregator from configs/, then runs Lightning with a
W&B logger. Override anything from the CLI, e.g.::

    hopmil-train model.aggregator=hopfield data=musk1 seed=1
    hopmil-train -m model.aggregator=attention,hopfield seed=0,1,2   # sweep
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    # TODO: instantiate dataset, encoder, aggregator (build_aggregator),
    #       MILClassifier, MILLitModule; wire WandbLogger + seed_everything;
    #       run trainer.fit / trainer.test.
    raise NotImplementedError("training loop wiring pending")


if __name__ == "__main__":
    main()
