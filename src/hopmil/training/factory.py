"""Builders shared by the single-run trainer (``train.py``) and the multi-run
aggregator comparison (``eval/compare.py``).

Keeping dataset/encoder construction here guarantees both entrypoints assemble
the *exact same* fixed pipeline (encoder + head per modality), so the only thing
that ever differs between runs is the aggregator.
"""

from __future__ import annotations

from omegaconf import DictConfig

from hopmil.data.mil_dataset import MILDataset
from hopmil.models.encoders import CNNEncoder, MLPEncoder


def build_dataset(data: DictConfig, seed: int) -> MILDataset:
    """Instantiate the dataset selected by ``data.loader``."""
    loader = data.loader
    if loader == "classic":
        from hopmil.data.classic import ClassicMIL

        return ClassicMIL(name=data.name, root=data.root, normalize=data.normalize)
    if loader == "mnist_bags":
        from hopmil.data.mnist_bags import MNISTBags

        return MNISTBags(
            root=data.root,
            target_digit=data.target_digit,
            bag_size=data.bag_size,
            num_witnesses=data.num_witnesses,
            positive_ratio=data.positive_ratio,
            num_bags=data.num_bags,
            seed=seed,
        )
    if loader == "fashion_mnist_bags":
        from hopmil.data.fashion_mnist_bags import FashionMNISTBags

        return FashionMNISTBags(
            root=data.root,
            target_class=data.target_class,
            bag_size=data.bag_size,
            num_witnesses=data.num_witnesses,
            positive_ratio=data.positive_ratio,
            num_bags=data.num_bags,
            seed=seed,
        )
    if loader == "cifar_bags":
        from hopmil.data.cifar_bags import CIFARBags

        return CIFARBags(
            root=data.root,
            target_class=data.target_class,
            bag_size=data.bag_size,
            num_witnesses=data.num_witnesses,
            positive_ratio=data.positive_ratio,
            num_bags=data.num_bags,
            seed=seed,
        )
    if loader == "colon_cancer":
        from hopmil.data.colon_cancer import ColonCancerBags

        return ColonCancerBags(
            root=data.root,
            target_class=data.target_class,
            patch_size=data.patch_size,
            normalize=data.normalize,
        )
    if loader == "ucsb_breast":
        from hopmil.data.ucsb_breast import UCSBBreastBags

        return UCSBBreastBags(
            root=data.root,
            patch_size=data.patch_size,
            white_frac=data.white_frac,
            normalize=data.normalize,
        )
    raise ValueError(f"unknown data.loader {loader!r}")


def build_encoder(data: DictConfig, dim: int):
    """One encoder per modality, held FIXED across the four aggregators."""
    loader = data.loader
    if loader == "classic":
        return MLPEncoder(in_dim=data.in_dim, dim=dim)
    if loader in ("mnist_bags", "fashion_mnist_bags"):
        return CNNEncoder(dim=dim, in_channels=1, image_size=28)
    if loader == "cifar_bags":
        return CNNEncoder(dim=dim, in_channels=3, image_size=32)
    if loader == "colon_cancer":
        return CNNEncoder(dim=dim, in_channels=3, image_size=data.patch_size)
    if loader == "ucsb_breast":
        return CNNEncoder(dim=dim, in_channels=3, image_size=data.patch_size)
    raise ValueError(f"unknown data.loader {loader!r}")
