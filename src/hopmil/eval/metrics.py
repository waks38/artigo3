"""Evaluation metrics.

Two families:

- **Bag-level classification** (``bag_classification_metrics``): the full suite
  computed per fold on the held-out test set. We log *all* of them so the paper
  can pick whichever it needs — the cost is negligible. The thesis verdict uses
  AUROC, but accuracy/AUPRC/F1/etc. are stored alongside.
- **Instance-level localization** (``instance_localization_auc``): synthetic data
  only (where ``instance_labels`` exist). Measures whether the aggregator's
  attention weights land on the witness instances — the interpretability claim
  the Hopfield-vs-attention comparison hinges on.
"""

from __future__ import annotations

import math

import numpy as np
import torch


def bag_classification_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Full binary-classification metric suite for one fold.

    Args:
        y_true: bag labels in {0, 1}.
        y_prob: predicted positive-class probabilities.
        threshold: decision threshold for the count-based metrics.

    Returns a flat dict; metrics undefined on a single-class fold are ``nan``.
    """
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        f1_score,
        log_loss,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    both_classes = len(np.unique(y_true)) == 2

    def _ranking(fn):  # metrics needing both classes present
        return float(fn(y_true, y_prob)) if both_classes else float("nan")

    out = {
        "auroc": _ranking(roc_auc_score),
        "auprc": _ranking(average_precision_score),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if both_classes else float("nan"),
        "brier": float(brier_score_loss(y_true, y_prob)) if both_classes else float("nan"),
    }
    try:
        out["log_loss"] = float(log_loss(y_true, y_prob, labels=[0, 1]))
    except ValueError:
        out["log_loss"] = float("nan")
    return out


def instance_localization_auc(weights: torch.Tensor, instance_labels: torch.Tensor) -> float:
    """AUROC of per-instance attention weights against true instance labels."""
    from sklearn.metrics import roc_auc_score

    y = instance_labels.detach().cpu().numpy()
    if len(set(y.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y, weights.detach().cpu().numpy()))


def mean_localization_auc(pairs: list[tuple[torch.Tensor, torch.Tensor]]) -> float:
    """Average per-bag localization AUC over a fold (skips undefined bags)."""
    aucs = [instance_localization_auc(w, il) for w, il in pairs]
    aucs = [a for a in aucs if not math.isnan(a)]
    return float(np.mean(aucs)) if aucs else float("nan")
