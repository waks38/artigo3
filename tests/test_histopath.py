"""Parser tests for the real-histopathology loaders, using synthetic fixtures
that mimic the documented on-disk layout (no downloads needed)."""

import numpy as np
import pytest
import scipy.io as sio

from hopmil.data.colon_cancer import ColonCancerBags
from hopmil.data.ucsb_breast import UCSBBreastBags

PIL = pytest.importorskip("PIL.Image")


def _save_img(path, arr):
    PIL.fromarray(arr.astype(np.uint8)).save(path)


def _make_colon_image(folder, stem, nuclei: dict, size=100):
    """nuclei: {class_name: [(x,y), ...]} in 1-indexed MATLAB coords."""
    folder.mkdir(parents=True, exist_ok=True)
    _save_img(folder / f"{stem}.bmp", np.full((size, size, 3), 120, np.uint8))
    for cls in ("epithelial", "fibroblast", "inflammatory", "others"):
        det = np.array(nuclei.get(cls, []), dtype=float).reshape(-1, 2)
        sio.savemat(folder / f"{stem}_{cls}.mat", {"detection": det})


def test_colon_positive_bag_has_correct_instances_and_labels(tmp_path):
    root = tmp_path / "colon"
    _make_colon_image(
        root / "img1",
        "img1",
        {"epithelial": [(50, 40), (20, 20)], "fibroblast": [(70, 70)]},
    )
    ds = ColonCancerBags(root=str(root), target_class="epithelial")
    assert len(ds.bags) == 1
    bag = ds.bags[0]
    assert bag.instances.shape == (3, 3, 27, 27)  # 3 nuclei, 27x27 RGB
    assert bag.instance_labels.tolist() == [1, 1, 0]  # 2 epithelial, 1 other
    assert int(bag.label) == 1


def test_colon_negative_bag_when_target_absent(tmp_path):
    root = tmp_path / "colon"
    _make_colon_image(root / "img1", "img1", {"fibroblast": [(30, 30), (60, 60)]})
    ds = ColonCancerBags(root=str(root), target_class="epithelial")
    bag = ds.bags[0]
    assert bag.instance_labels.tolist() == [0, 0]
    assert int(bag.label) == 0


def test_colon_missing_data_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ColonCancerBags(root=str(tmp_path / "empty"))


def test_ucsb_labels_from_filename_and_patch_grid(tmp_path):
    root = tmp_path / "ucsb" / "Images"
    root.mkdir(parents=True)
    # 96x64 -> 3x2 grid of 32x32 = 6 patches; mid-gray so none are dropped
    _save_img(root / "ytma1_benign.tif", np.full((64, 96, 3), 100, np.uint8))
    _save_img(root / "ytma2_malignant.tif", np.full((64, 96, 3), 100, np.uint8))
    ds = UCSBBreastBags(root=str(tmp_path / "ucsb"))
    labels = sorted(int(b.label) for b in ds.bags)
    assert labels == [0, 1]
    for b in ds.bags:
        assert b.instances.shape == (6, 3, 32, 32)
        assert b.instance_labels is None


def test_ucsb_missing_data_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        UCSBBreastBags(root=str(tmp_path / "empty"))
