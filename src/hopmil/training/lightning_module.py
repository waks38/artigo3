"""LightningModule wrapping a MILClassifier for binary/multiclass bag tasks."""

from __future__ import annotations

import lightning as L
import torch
import torch.nn.functional as F
from torchmetrics.classification import BinaryAUROC, BinaryAccuracy

from hopmil.models.mil_model import MILClassifier


class MILLitModule(L.LightningModule):
    def __init__(self, model: MILClassifier, lr: float = 1e-3, weight_decay: float = 1e-4):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.val_auc = BinaryAUROC()
        self.val_acc = BinaryAccuracy()

    def _step(self, batch):
        logits, _ = self.model(batch["instances"])
        target = batch["labels"].float().view(-1, 1)
        loss = F.binary_cross_entropy_with_logits(logits, target)
        return loss, logits.sigmoid().view(-1), batch["labels"].int()

    def training_step(self, batch, _):
        loss, *_ = self._step(batch)
        self.log("train/loss", loss, batch_size=len(batch["labels"]))
        return loss

    def validation_step(self, batch, _):
        loss, probs, target = self._step(batch)
        self.val_auc.update(probs, target)
        self.val_acc.update(probs, target)
        self.log("val/loss", loss, prog_bar=True, batch_size=len(target))

    def on_validation_epoch_end(self):
        self.log("val/auc", self.val_auc.compute(), prog_bar=True)
        self.log("val/acc", self.val_acc.compute(), prog_bar=True)
        self.val_auc.reset()
        self.val_acc.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
