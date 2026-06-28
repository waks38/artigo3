"""Evaluation metrics beyond standard classification.

Bag-level: accuracy, AUROC (in the Lightning module).
Instance-level (synthetic data only): how well do the aggregator's attention
weights localize the witness instances? This is the interpretability claim the
Hopfield vs. attention comparison hinges on.
"""

from __future__ import annotations

import torch


def instance_localization_auc(weights: torch.Tensor, instance_labels: torch.Tensor) -> float:
    """AUROC of per-instance attention weights against true instance labels."""
    from sklearn.metrics import roc_auc_score

    y = instance_labels.detach().cpu().numpy()
    if len(set(y.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y, weights.detach().cpu().numpy()))
