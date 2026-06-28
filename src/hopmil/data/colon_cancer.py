"""Colon Cancer (CRCHistoPhenotypes) — real histopathology MIL with witnesses.

Sirinukunwattana et al. (2016), as used for MIL by Ilse et al. (2018). 100 H&E
images (500x500); each annotated nucleus has a class (epithelial / inflammatory
/ fibroblast / others). MIL setup:

- **Bag** = one image.
- **Instance** = a 27x27 RGB patch centered on an annotated nucleus.
- **Bag label** = positive iff the bag contains >=1 nucleus of ``target_class``
  (default ``epithelial``, the cancer-relevant class).
- **Instance labels** exist (the nucleus class) and drive the interpretability
  eval — used for evaluation only, never for training.

Data is NOT auto-downloadable: the canonical Warwick host is behind an SSO form.
Download once (Warwick ``CRCHistoPhenotypes_2016_04_28.zip`` or a Kaggle mirror)
and extract under ``root`` so that image folders ``img1/ .. img100/`` (each with
``imgK.bmp`` and ``imgK_<class>.mat``) are reachable below it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

from hopmil.data.mil_dataset import Bag, MILDataset

NUCLEUS_CLASSES = ["epithelial", "fibroblast", "inflammatory", "others"]

# Simple symmetric normalization to ~[-1, 1]; one fixed encoder per modality.
_MEAN = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
_STD = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)

_DOWNLOAD_HINT = (
    "Colon Cancer data not found under {root!r}.\n"
    "It is not auto-downloadable (Warwick host is SSO-gated). Download once and "
    "extract so that per-image folders (img1/, img2/, ... each with imgK.bmp and "
    "imgK_epithelial.mat etc.) live under {root!r}.\n"
    "Sources: Warwick CRCHistoPhenotypes_2016_04_28.zip, or a Kaggle/Drive mirror."
)


def _load_detection(mat_path: Path) -> np.ndarray:
    """Return an (N, 2) array of [x, y] nucleus centroids (0-indexed)."""
    if not mat_path.exists():
        return np.empty((0, 2), dtype=np.int64)
    det = sio.loadmat(mat_path).get("detection")
    if det is None or det.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    # MATLAB coords are 1-indexed; columns are [x, y] = [col, row].
    return np.asarray(det, dtype=np.float64).reshape(-1, 2) - 1.0


class ColonCancerBags(MILDataset):
    SUPPORTED_TARGETS = set(NUCLEUS_CLASSES)

    def __init__(
        self,
        root: str = "data/raw/colon_cancer",
        target_class: str = "epithelial",
        patch_size: int = 27,
        normalize: bool = True,
    ) -> None:
        if target_class not in self.SUPPORTED_TARGETS:
            raise ValueError(f"{target_class!r} not in {self.SUPPORTED_TARGETS}")
        self.target_class = target_class
        self.patch_size = patch_size

        root = Path(root)
        image_paths = sorted(root.rglob("*.bmp"))
        if not image_paths:
            raise FileNotFoundError(_DOWNLOAD_HINT.format(root=str(root)))

        from PIL import Image

        r = patch_size // 2  # patch is (2r+1); default 27 -> r=13
        self.bags = []
        for img_path in image_paths:
            stem = img_path.stem  # e.g. "img1"
            img = np.asarray(Image.open(img_path).convert("RGB"))  # (H, W, 3) uint8
            padded = np.pad(img, ((r, r), (r, r), (0, 0)), mode="reflect")

            coords, inst_labels = [], []
            for cls in NUCLEUS_CLASSES:
                det = _load_detection(img_path.with_name(f"{stem}_{cls}.mat"))
                for x, y in det:
                    coords.append((int(round(x)), int(round(y))))
                    inst_labels.append(1 if cls == target_class else 0)
            if not coords:
                continue  # no annotated nuclei -> skip (cannot form a bag)

            patches = []
            for cx, cy in coords:
                # center (cy, cx) maps to (cy+r, cx+r) in padded; take (2r+1) window
                patch = padded[cy : cy + 2 * r + 1, cx : cx + 2 * r + 1, :]
                patches.append(torch.from_numpy(patch.copy()).permute(2, 0, 1))

            instances = torch.stack(patches).float().div(255.0)  # (n,3,ps,ps)
            if normalize:
                instances = (instances - _MEAN) / _STD
            instance_labels = torch.tensor(inst_labels, dtype=torch.long)
            label = instance_labels.any().long()  # positive iff a target nucleus
            self.bags.append(Bag(instances=instances, label=label, instance_labels=instance_labels))
