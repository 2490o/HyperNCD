"""
Improved training entry for GAP + hypergraph discovery.

The original train_GAP.py is left untouched. This entry adds practical training
improvements: full checkpoint resume, warmup-aware optimization, region-loss
ramping, gradient clipping, and optional mixed precision/SWA.
"""
import os
from argparse import ArgumentParser
from datetime import datetime

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from modules.Discoverer_GAP_new import Discoverer
from train_GAP import parser as base_parser
from train_GAP import set_seed
from utils import unkn_labels as unk_labels
from utils.callbacks import mIoUEvaluatorCallback


parser = ArgumentParser(parents=[base_parser], conflict_handler="resolve")
parser.set_defaults(
    comment="GAP_new_" + datetime.now().strftime("%b%d_%H-%M-%S"),
    use_scheduler=True,
    warmup_epochs=2,
    use_prototype_memory=True,
    label_smoothing=0.10,
)
parser.add_argument("--resume_from_checkpoint", default=None, type=str, help="Lightning checkpoint path to resume from")
parser.add_argument("--auto_resume", default=False, action="store_true", help="resume from checkpoint_dir/dataset/comment/last.ckpt if it exists")
parser.add_argument("--no_prototype_memory", default=False, action="store_true", help="disable the new default prototype memory")
parser.add_argument("--region_warmup_epochs", default=3, type=int, help="linearly ramp region loss weight over the first N epochs")
parser.add_argument("--gradient_clip_val", default=1.0, type=float, help="Lightning gradient clipping value; 0 disables")
parser.add_argument("--accumulate_grad_batches", default=1, type=int, help="gradient accumulation")
parser.add_argument("--precision", default="32", type=str, choices=["32", "16-mixed", "bf16-mixed"], help="Lightning precision setting")
parser.add_argument("--enable_swa", default=False, action="store_true", help="enable stochastic weight averaging near the end of training")
parser.add_argument("--swa_lrs", default=None, type=float, help="SWA learning rate; defaults to train_lr * 0.1")
parser.add_argument("--validate_best_after_fit", default=False, action="store_true", help="run validation once with the best checkpoint after training")


def _resolve_dataset_config(args):
    if args.dataset_config is not None:
        return
    if args.dataset == "SemanticKITTI":
        args.dataset_config = "config/semkitti_dataset.yaml"
    elif args.dataset == "SemanticPOSS":
        args.dataset_config = "config/semposs_dataset.yaml"
    else:
        raise NameError(f"Dataset {args.dataset} not implemented")


def _build_trainer(args, loggers, callbacks):
    common_kwargs = dict(
        max_epochs=args.epochs,
        logger=loggers,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        log_every_n_steps=50,
        accumulate_grad_batches=max(int(args.accumulate_grad_batches), 1),
    )
    if float(args.gradient_clip_val) > 0:
        common_kwargs["gradient_clip_val"] = float(args.gradient_clip_val)
        common_kwargs["gradient_clip_algorithm"] = "norm"
    if args.precision != "32":
        common_kwargs["precision"] = args.precision

    try:
        return pl.Trainer(accelerator="gpu", devices=1, enable_progress_bar=True, **common_kwargs)
    except TypeError:
        legacy_kwargs = dict(common_kwargs)
        legacy_kwargs.pop("gradient_clip_algorithm", None)
        if legacy_kwargs.get("precision") == "16-mixed":
            legacy_kwargs["precision"] = 16
        elif legacy_kwargs.get("precision") == "bf16-mixed":
            legacy_kwargs["precision"] = "bf16"
        try:
            return pl.Trainer(gpus=1, progress_bar_refresh_rate=10, **legacy_kwargs)
        except TypeError:
            legacy_kwargs.pop("precision", None)
            return pl.Trainer(gpus=1, **legacy_kwargs)


def _maybe_add_swa(callbacks, args):
    if not args.enable_swa:
        return
    try:
        from pytorch_lightning.callbacks import StochasticWeightAveraging
    except ImportError:
        print("SWA callback is unavailable in this PyTorch Lightning version; continuing without SWA.")
        return
    swa_lrs = args.swa_lrs if args.swa_lrs is not None else float(args.train_lr) * 0.1
    callbacks.append(StochasticWeightAveraging(swa_lrs=swa_lrs))


def main(args):
    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
    if args.no_prototype_memory:
        args.use_prototype_memory = False

    _resolve_dataset_config(args)
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.dataset, args.comment)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    auto_ckpt = os.path.join(args.checkpoint_dir, "last.ckpt")
    ckpt_path = args.resume_from_checkpoint
    if args.auto_resume and ckpt_path is None and os.path.exists(auto_ckpt):
        ckpt_path = auto_ckpt

    print(args)
    if ckpt_path is not None:
        print(f"Resuming training from: {ckpt_path}")

    run_name = "-".join([f"S{args.split}", "GAP-new", args.dataset, args.comment])
    wandb_kwargs = dict(save_dir=args.log_dir, name=run_name, project=args.project, offline=args.offline)
    if args.entity is not None:
        wandb_kwargs["entity"] = args.entity
    wandb_logger = WandbLogger(**wandb_kwargs)

    with open(args.dataset_config, "r") as f:
        dataset_config = yaml.safe_load(f)

    unknown_labels = unk_labels.unknown_labels(split=args.split, dataset_config=dataset_config)
    label_mapping, label_mapping_inv, unknown_label = unk_labels.label_mapping(
        unknown_labels, dataset_config["learning_map_inv"].keys()
    )

    args.num_classes = len(label_mapping)
    args.num_unlabeled_classes = len(unknown_labels)
    args.num_labeled_classes = args.num_classes - args.num_unlabeled_classes

    checkpoint_callback = ModelCheckpoint(
        monitor="valid/mIoU",
        mode="max",
        save_top_k=1,
        save_last=True,
        save_weights_only=False,
        dirpath=args.checkpoint_dir,
        filename=f"{args.dataset}_S{args.split}_GAP_new_best-{{epoch}}-{{step}}",
        verbose=True,
    )
    callbacks = [
        mIoUEvaluatorCallback(),
        checkpoint_callback,
        LearningRateMonitor(logging_interval="epoch"),
    ]
    _maybe_add_swa(callbacks, args)

    loggers = [wandb_logger, CSVLogger(save_dir=args.log_dir)]
    model = Discoverer(label_mapping, label_mapping_inv, unknown_label, **args.__dict__)
    trainer = _build_trainer(args, loggers, callbacks)
    trainer.fit(model, ckpt_path=ckpt_path)
    if args.validate_best_after_fit:
        best_model_path = checkpoint_callback.best_model_path
        if best_model_path:
            print(f"Validating best checkpoint: {best_model_path}")
            trainer.validate(model, ckpt_path=best_model_path)
        else:
            print("No best checkpoint was found; skipping post-training validation.")


if __name__ == "__main__":
    args = parser.parse_args()
    set_seed(args.seed, deterministic=args.set_deterministic)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    main(args)
