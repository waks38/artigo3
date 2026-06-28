"""LightningModule wrapping a MILClassifier for binary bag tasks.

Validation keeps lightweight torchmetrics (``val/auc`` drives early-stopping and
checkpoint selection). The **test** path instead accumulates raw predictions and
computes the *full* metric suite (``eval/metrics``) once per fold — bag-level
classification metrics plus, when ``instance_labels`` are present, the
instance-localization AUC. Everything is logged under ``test/*`` so the
comparison runner can collect all of it (we log all metrics; the paper picks).
"""

from __future__ import annotations

import lightning as L
import torch
import torch.nn.functional as F
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC

from hopmil.eval.metrics import bag_classification_metrics, mean_localization_auc
from hopmil.models.mil_model import MILClassifier


class MILLitModule(L.LightningModule):
    def __init__(self, model: MILClassifier, lr: float = 1e-3, weight_decay: float = 1e-4):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.val_auc = BinaryAUROC()
        self.val_acc = BinaryAccuracy()
        # test-time buffers (filled per fold, consumed in on_test_epoch_end)
        self._test_probs: list[torch.Tensor] = []
        self._test_targets: list[torch.Tensor] = []
        self._test_loc: list[tuple[torch.Tensor, torch.Tensor]] = []

    def _step(self, batch):
        # Bags are a list of variable-length tensors; move each to the model's
        # device explicitly so GPU runs don't hit a CPU/GPU mismatch.
        instances = [x.to(self.device) for x in batch["instances"]]
        logits, attns = self.model(instances)
        target = batch["labels"].float().view(-1, 1).to(self.device)
        loss = F.binary_cross_entropy_with_logits(logits, target)
        return loss, logits.sigmoid().view(-1), batch["labels"].int(), attns

    def training_step(self, batch, _):
        loss, *_ = self._step(batch)
        self.log("train/loss", loss, batch_size=len(batch["labels"]))
        return loss

    def validation_step(self, batch, _):
        loss, probs, target, _ = self._step(batch)
        self.val_auc.update(probs, target)
        self.val_acc.update(probs, target)
        self.log("val/loss", loss, prog_bar=True, batch_size=len(target))

    def on_validation_epoch_end(self):
        self.log("val/auc", self.val_auc.compute(), prog_bar=True)
        self.log("val/acc", self.val_acc.compute(), prog_bar=True)
        self.val_auc.reset()
        self.val_acc.reset()

    def test_step(self, batch, _):
        loss, probs, target, attns = self._step(batch)
        self._test_probs.append(probs.detach().cpu())
        self._test_targets.append(target.detach().cpu())
        # localization: keep (weights, instance_labels) for bags that have both
        for w, il in zip(attns, batch["instance_labels"], strict=False):
            if w is not None and il is not None:
                self._test_loc.append((w.detach().cpu(), il.detach().cpu()))
        self.log("test/loss", loss, batch_size=len(target))

    def on_test_epoch_end(self):
        probs = torch.cat(self._test_probs).numpy()
        targets = torch.cat(self._test_targets).numpy()
        metrics = bag_classification_metrics(targets, probs)
        metrics["localization_auc"] = mean_localization_auc(self._test_loc)
        for name, value in metrics.items():
            self.log(f"test/{name}", value)
        self._test_probs.clear()
        self._test_targets.clear()
        self._test_loc.clear()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
