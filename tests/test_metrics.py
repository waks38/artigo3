import math

import torch

from hopmil.eval.metrics import (
    bag_classification_metrics,
    instance_localization_auc,
    mean_localization_auc,
)


def test_perfect_classifier_metrics():
    y_true = [0, 0, 1, 1]
    y_prob = [0.1, 0.2, 0.8, 0.9]
    m = bag_classification_metrics(y_true, y_prob)
    assert m["auroc"] == 1.0
    assert m["accuracy"] == 1.0
    assert m["f1"] == 1.0
    # every expected metric is present
    for k in (
        "auroc",
        "auprc",
        "accuracy",
        "balanced_accuracy",
        "f1",
        "precision",
        "recall",
        "mcc",
        "brier",
        "log_loss",
    ):
        assert k in m


def test_single_class_fold_yields_nan_ranking():
    m = bag_classification_metrics([1, 1, 1], [0.6, 0.7, 0.8])
    assert math.isnan(m["auroc"])  # AUROC undefined with one class
    assert not math.isnan(m["accuracy"])  # count metrics still defined


def test_localization_auc_perfect_and_undefined():
    weights = torch.tensor([0.05, 0.9, 0.05])
    labels = torch.tensor([0, 1, 0])
    assert instance_localization_auc(weights, labels) == 1.0
    # all-negative bag -> undefined
    assert math.isnan(instance_localization_auc(weights, torch.tensor([0, 0, 0])))


def test_mean_localization_skips_undefined_bags():
    good = (torch.tensor([0.1, 0.9]), torch.tensor([0, 1]))
    bad = (torch.tensor([0.1, 0.9]), torch.tensor([0, 0]))  # undefined
    assert mean_localization_auc([good, bad]) == 1.0
    assert math.isnan(mean_localization_auc([bad]))
