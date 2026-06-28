"""UCSB Breast Cancer (Bio-Segmentation) — real histopathology MIL, no witnesses.

Gelasca et al. (2008); used for MIL by Kandemir & Hamprecht (2014) and Ilse et
al. (2018). 58 H&E TMA images (896x768), each benign or malignant. MIL setup:

- **Bag** = one image.
- **Instance** = a 32x32 RGB patch from a regular non-overlapping grid; patches
  that are mostly background (>``white_frac`` nearly-white pixels) are dropped, so
  bag sizes vary.
- **Bag label** = malignant (positive) vs benign (negative), read from the
  filename (the original ``ytma..._benign/malignant.tif`` convention).
- **No instance labels** (``instance_labels=None``) — like the tabular classics,
  this set has no per-patch ground truth, so it gives bag-level performance only.

Not auto-downloadable (UCSB host is down). Download once (Kaggle mirror
``andrewmvd/breast-cancer-cell-segmentation``) and place the ``.tif`` images
anywhere under ``root``; labels are parsed from filenames.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from hopmil.data.mil_dataset import Bag, MILDataset

_MEAN = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
_STD = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)

_DOWNLOAD_HINT = (
    "UCSB Breast data not found under {root!r}.\n"
    "It is not auto-downloadable (UCSB host is down). Download once and place the "
    "H&E .tif images under {root!r} (filenames must contain 'benign'/'malignant').\n"
    "Source: Kaggle 'andrewmvd/breast-cancer-cell-segmentation'."
)


def _label_from_name(name: str) -> int | None:
    low = name.lower()
    if "malignant" in low:
        return 1
    if "benign" in low:
        return 0
    return None


class UCSBBreastBags(MILDataset):
    def __init__(
        self,
        root: str = "data/raw/ucsb_breast",
        patch_size: int = 32,
        white_frac: float = 0.75,
        white_thresh: int = 200,
        normalize: bool = True,
    ) -> None:
        self.patch_size = patch_size
        root = Path(root)
        # Prefer an "Images" subfolder if present, else any tif under root.
        paths = sorted(p for p in root.rglob("*.tif") if "mask" not in p.parent.name.lower())
        paths = [p for p in paths if _label_from_name(p.name) is not None]
        if not paths:
            raise FileNotFoundError(_DOWNLOAD_HINT.format(root=str(root)))

        from PIL import Image

        ps = patch_size
        self.bags = []
        for img_path in paths:
            label = _label_from_name(img_path.name)
            img = np.asarray(Image.open(img_path).convert("RGB"))  # (H, W, 3)
            h, w = img.shape[:2]

            patches = []
            for i in range(0, h - ps + 1, ps):
                for j in range(0, w - ps + 1, ps):
                    tile = img[i : i + ps, j : j + ps, :]
                    # drop near-white background tiles
                    white = (tile >= white_thresh).all(axis=-1).mean()
                    if white <= white_frac:
                        patches.append(torch.from_numpy(tile.copy()).permute(2, 0, 1))
            if not patches:  # degenerate image -> keep one tile so the bag exists
                patches.append(torch.from_numpy(img[:ps, :ps, :].copy()).permute(2, 0, 1))

            instances = torch.stack(patches).float().div(255.0)  # (n,3,ps,ps)
            if normalize:
                instances = (instances - _MEAN) / _STD
            self.bags.append(
                Bag(instances=instances, label=torch.tensor(label), instance_labels=None)
            )
