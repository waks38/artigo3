import torch

from hopmil.data import Bag, MILDataset
from hopmil.data.datamodule import MILDataModule


class _ToyMIL(MILDataset):
    def __init__(self, n=100, pos_frac=0.4):
        n_pos = int(n * pos_frac)
        labels = [1] * n_pos + [0] * (n - n_pos)
        self.bags = [Bag(instances=torch.randn(3, 8), label=torch.tensor(y)) for y in labels]


def _indices(dm):
    dm.setup()
    return set(dm.train_idx), set(dm.val_idx), set(dm.test_idx)


def test_splits_are_disjoint_and_cover_everything():
    ds = _ToyMIL(n=100)
    train, val, test = _indices(MILDataModule(ds, n_folds=10, fold=0, seed=0))
    assert train & val == set()
    assert train & test == set()
    assert val & test == set()
    assert train | val | test == set(range(100))


def test_every_fold_is_eventually_the_test_set():
    ds = _ToyMIL(n=100)
    seen = set()
    for f in range(10):
        _, _, test = _indices(MILDataModule(ds, n_folds=10, fold=f, seed=0))
        assert not (seen & test)  # folds are disjoint across f
        seen |= test
    assert seen == set(range(100))  # the 10 test folds tile the dataset


def test_split_is_deterministic_in_seed():
    ds = _ToyMIL(n=100)
    a = _indices(MILDataModule(ds, n_folds=5, fold=2, seed=7))
    b = _indices(MILDataModule(ds, n_folds=5, fold=2, seed=7))
    assert a == b


def test_test_fold_is_stratified():
    ds = _ToyMIL(n=100, pos_frac=0.4)
    labels = [int(b.label) for b in ds.bags]
    dm = MILDataModule(ds, n_folds=10, fold=0, seed=0)
    dm.setup()
    pos = sum(labels[i] for i in dm.test_idx)
    # ~4 positives expected in a 10-bag fold of a 40%-positive set.
    assert 2 <= pos <= 6
