"""Classic tabular MIL benchmarks: Elephant / Fox / Tiger (Andrews et al., 2002).

Source: the MIProblems repository (Cheplygina), now hosted on Figshare. Each
file is a PRTools ``prdataset`` MATLAB object with:

- ``data``: ``(n_instances, 230)`` feature matrix.
- ``ident.milbag``: 1-indexed bag id per instance (groups rows into bags).
- ``nlab`` in {1, 2} with ``lablist = [negative, positive]`` — the BAG label,
  broadcast to every instance (there are NO true instance-level labels here,
  unlike the synthetic MNIST-bags). Hence ``Bag.instance_labels`` is ``None``.

Download once with :func:`download_fte`.
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

# Figshare "fte.zip" (Fox/Tiger/Elephant) from the MIProblems repository.
_FTE_URL = "https://ndownloader.figshare.com/files/12144461"


def download_fte(root: str = "data/raw/mil") -> None:
    """Download + extract elephant/fox/tiger .mat files if missing."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if all((root / f"{n}.mat").exists() for n in ("elephant", "fox", "tiger")):
        return
    req = urllib.request.Request(_FTE_URL, headers={"User-Agent": "Mozilla/5.0"})
    blob = urllib.request.urlopen(req, timeout=120).read()
    zipfile.ZipFile(io.BytesIO(blob)).extractall(root)


class ClassicMIL(MILDataset):
    SUPPORTED = {"elephant", "fox", "tiger"}

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
            download_fte(root)

        x = sio.loadmat(Path(root) / f"{name}.mat")["x"][0, 0]
        feats = x["data"].astype(np.float32)               # (N, 230)
        nlab = x["nlab"].ravel()                            # 1=neg, 2=pos
        milbag = x["ident"][0, 0]["milbag"].ravel()         # 1-indexed bag id

        if normalize:  # z-score per feature across all instances
            feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-8)
        feats = torch.from_numpy(feats)

        self.name = name
        self.bags = []
        for bag_id in np.unique(milbag):
            mask = milbag == bag_id
            label = int((nlab[mask] == 2).any())            # positive iff any pos
            self.bags.append(
                Bag(
                    instances=feats[mask],                  # (n_i, 230)
                    label=torch.tensor(label),
                    instance_labels=None,                   # no per-instance truth
                )
            )
