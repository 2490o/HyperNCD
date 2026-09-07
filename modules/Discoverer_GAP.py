"""
Improved training wrapper for the GAP discoverer.

This module keeps the original model and losses intact, but makes optimization
less abrupt by adding warmup-aware cosine scheduling and optional region-loss
ramping. It is intended to be used by train_GAP_new.py.
"""
import math

from torch import optim
from torch.optim.lr_scheduler import LambdaLR

from modules.Discoverer_GAP import Discoverer as GAPDiscoverer


class Discoverer(GAPDiscoverer):
    def __init__(self, label_mapping, label_mapping_inv, unknown_label, **kwargs):
        super().__init__(label_mapping, label_mapping_inv, unknown_label, **kwargs)
        self._target_alpha = float(getattr(self.hparams, "alpha", 1.0))

    def on_train_epoch_start(self):
        ramp_epochs = int(getattr(self.hparams, "region_warmup_epochs", 0))
        if ramp_epochs > 0:
            progress = min(1.0, float(self.current_epoch + 1) / float(ramp_epochs))
            self.hparams.alpha = self._target_alpha * progress
        else:
            self.hparams.alpha = self._target_alpha
        super().on_train_epoch_start()

    def configure_optimizers(self):
        weight_decay = float(self.hparams.weight_decay_for_optim)
        decay_params, no_decay_params = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or name.endswith(".bias"):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = optim.AdamW(
            [
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=float(self.hparams.train_lr),
        )

        max_epochs = max(int(self.hparams.epochs), 1)
        warmup_epochs = max(int(getattr(self.hparams, "warmup_epochs", 0)), 0)
        min_lr = float(getattr(self.hparams, "min_lr", 1e-5))
        base_lr = max(float(self.hparams.train_lr), 1e-12)
        min_factor = min(max(min_lr / base_lr, 0.0), 1.0)

        def lr_lambda(epoch):
            if warmup_epochs > 0 and epoch < warmup_epochs:
                return max(float(epoch + 1) / float(warmup_epochs), min_factor)
            if max_epochs <= warmup_epochs:
                return 1.0
            progress = float(epoch - warmup_epochs) / float(max_epochs - warmup_epochs)
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_factor + (1.0 - min_factor) * cosine

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
