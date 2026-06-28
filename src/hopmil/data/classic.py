"""Classic tabular MIL benchmarks: Elephant / Fox / Tiger (Andrews et al., 2002)
and Musk1 / Musk2 (Dietterich et al., 1997).

Source: the MIProblems repository (Cheplygina), now hosted on Figshare. Each
file is a PRTools ``prdataset`` MATLAB object with:

- ``data``: ``(n_instances, n_features)`` feature matrix (230 for elephant/fox/
  tiger image-region features; 166 for the musk molecular descriptors).
- ``ident.milbag``: 1-indexed bag id per instance (groups rows into bags).
- ``nlab`` in {1, 2} with ``lablist = [negative, positive]`` — the BAG label,
  broadcast to every instance (there are NO true instance-level labels here,
  unlike the synthetic MNIST-bags). Hence ``Bag.instance_labels`` is ``None``.

Download once with :func:`download_fte` (fte.zip) / :func:`download_musk`
(musk.zip); both extract into the same ``root``.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

from hopmil.data.mil_dataset import Bag, MILDataset

# Figshare archives from the MIProblems repository.
_FTE_URL = "https://ndownloader.figshare.com/files/12144461"  # elephant/fox/tiger
_MUSK_URL = "https://ndownloader.figshare.com/files/12144473"  # musk1/musk2


def _download_zip(url: str, members: tuple[str, ...], root: str) -> None:
    """Download + extract a MIProblems .zip into ``root`` if any member is missing."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if all((root / f"{n}.mat").exists() for n in members):
        return
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    blob = urllib.request.urlopen(req, timeout=120).read()
    zipfile.ZipFile(io.BytesIO(blob)).extractall(root)


def download_fte(root: str = "data/raw/mil") -> None:
    """Download + extract elephant/fox/tiger .mat files if missing."""
    _download_zip(_FTE_URL, ("elephant", "fox", "tiger"), root)


def download_musk(root: str = "data/raw/mil") -> None:
    """Download + extract musk1/musk2 .mat files if missing."""
    _download_zip(_MUSK_URL, ("musk1", "musk2"), root)


class ClassicMIL(MILDataset):
    SUPPORTED = {"elephant", "fox", "tiger", "musk1", "musk2"}

    def __init__(
        self,
        name: str,
        root: str = "data/raw/mil",
        normalize: bool = True,
        download: bool = True,
    ) -> None:
        if name not in self.SUPPORTED:
            raise ValueError(f"{name!r} not in {self.SUPPORTED}")
        if download:
            (download_musk if name.startswith("musk") else download_fte)(root)

        x = sio.loadmat(Path(root) / f"{name}.mat")["x"][0, 0]
        feats = x["data"].astype(np.float32)  # (N, 230)
        nlab = x["nlab"].ravel()  # 1=neg, 2=pos
        milbag = x["ident"][0, 0]["milbag"].ravel()  # 1-indexed bag id

        if normalize:  # z-score per feature across all instances
            feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-8)
        feats = torch.from_numpy(feats)

        self.name = name
        self.bags = []
        for bag_id in np.unique(milbag):
            mask = milbag == bag_id
            label = int((nlab[mask] == 2).any())  # positive iff any pos
            self.bags.append(
                Bag(
                    instances=feats[mask],  # (n_i, 230)
                    label=torch.tensor(label),
                    instance_labels=None,  # no per-instance truth
                )
            )
