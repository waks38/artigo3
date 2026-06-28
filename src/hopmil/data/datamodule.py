"""LightningDataModule with stratified k-fold for MIL bags.

The whole comparison is paired: the four aggregators must see the *same* folds
(same train/val/test bag indices) so their per-fold scores can be fed to the
correlated Bayesian t-test (baycomp). This module owns that split.

Design:

- Wraps any :class:`~hopmil.data.mil_dataset.MILDataset` already built by the
  caller (the dataset itself is dataset-specific; splitting is not).
- **Stratified k-fold** over the bag-level labels. ``fold`` selects which fold
  is held out as the *test* set for this run; the rest is the train+val pool.
- A stratified ``val_frac`` slice of that pool is the *validation* set
  (early-stopping / checkpoint selection); the remainder is *train*.
- Bags are variable-length and unpadded, so all loaders use ``mil_collate``
  (a batch is a list of instance tensors, not a dense tensor).

The fold partition is deterministic in ``seed`` and independent of the
aggregator, which is exactly what the paired statistical test requires.
"""

from __future__ import annotations

import lightning as L
import numpy as np
from torch.utils.data import DataLoader, Subset

from hopmil.data.mil_dataset import MILDataset, mil_collate


class MILDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset: MILDataset,
        n_folds: int = 10,
        fold: int = 0,
        val_frac: float = 0.2,
        batch_size: int = 8,
        num_workers: int = 0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if not 0 <= fold < n_folds:
            raise ValueError(f"fold {fold} out of range [0, {n_folds})")
        self.dataset = dataset
        self.n_folds = n_folds
        self.fold = fold
        self.val_frac = val_frac
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.train_idx: np.ndarray | None = None
        self.val_idx: np.ndarray | None = None
        self.test_idx: np.ndarray | None = None

    def _bag_labels(self) -> np.ndarray:
        return np.array([int(b.label) for b in self.dataset.bags])

    def setup(self, stage: str | None = None) -> None:
        if self.train_idx is not None:  # idempotent across fit/test calls
            return
        labels = self._bag_labels()
        n = len(labels)
        rng = np.random.default_rng(self.seed)

        # Stratified k-fold: within each class, shuffle then round-robin into
        # folds so every fold has near-identical class balance.
        fold_of = np.empty(n, dtype=int)
        for c in np.unique(labels):
            idx = np.where(labels == c)[0]
            rng.shuffle(idx)
            fold_of[idx] = np.arange(len(idx)) % self.n_folds

        test_idx = np.where(fold_of == self.fold)[0]
        trainval_idx = np.where(fold_of != self.fold)[0]

        # Stratified validation slice carved out of the train+val pool.
        tv_labels = labels[trainval_idx]
        val_parts = []
        for c in np.unique(tv_labels):
            idx = trainval_idx[tv_labels == c]
            rng.shuffle(idx)
            k = max(1, int(round(len(idx) * self.val_frac)))
            val_parts.append(idx[:k])
        val_idx = np.sort(np.concatenate(val_parts)) if val_parts else np.array([], int)
        train_idx = np.sort(np.setdiff1d(trainval_idx, val_idx))

        self.train_idx, self.val_idx, self.test_idx = train_idx, val_idx, test_idx

    def _loader(self, idx: np.ndarray, shuffle: bool) -> DataLoader:
        return DataLoader(
            Subset(self.dataset, idx.tolist()),
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=mil_collate,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_idx, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_idx, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_idx, shuffle=False)
