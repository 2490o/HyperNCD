"""
train_GAP.py
Training entry for the full Geometry-Aware Prototype + Hypergraph version.

Generated from train.py without modifying the original file. This script imports
modules.Discoverer_GAP.Discoverer and adds GAP/hypergraph ablation arguments.
"""
import os
import random
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from modules.Discoverer_GAP import Discoverer
from utils import unkn_labels as unk_labels
from utils.callbacks import mIoUEvaluatorCallback

SEED = 1234

parser = ArgumentParser()
parser.add_argument("-s", "--split", type=int, help="split", required=True)
parser.add_argument("--dataset", choices=["SemanticKITTI", "SemanticPOSS"], default="SemanticPOSS", type=str, help="dataset")
parser.add_argument("--dataset_config", default=None, type=str, help="dataset config file")
parser.add_argument("--voxel_size", default="0.05", type=float, help="voxel_size")
parser.add_argument("--downsampling", default="60000", type=int, help="number of points per pcd")
parser.add_argument("--batch_size", default=4, type=int, help="batch size")
parser.add_argument("--num_workers", default=8, type=int, help="number of workers")
parser.add_argument("--hungarian_at_each_step", default=True, action="store_true", help="enable hungarian pass at each epoch")
parser.add_argument("--log_dir", default="logs", type=str, help="log directory")
parser.add_argument("--checkpoint_dir", default="checkpoints", type=str, help="checkpoint dir")
parser.add_argument("--train_lr", default=0.001, type=float, help="learning rate")
parser.add_argument("--finetune_lr", default=1.0e-4, type=float, help="finetune learning rate")
parser.add_argument("--use_scheduler", default=False, action="store_true", help="use lr scheduler")
parser.add_argument("--warmup_epochs", default=0, type=int, help="warmup epochs")
parser.add_argument("--detach", default=None, type=int, help="warmup epochs")
parser.add_argument("--min_lr", default=1e-5, type=float, help="min learning rate")
parser.add_argument("--momentum_for_optim", default=0.9, type=float, help="momentum for optimizer")
parser.add_argument("--weight_decay_for_optim", default=1.0e-4, type=float, help="weight decay")
parser.add_argument("--overcluster_factor", default=None, type=int, help="overclustering factor")
parser.add_argument("--num_heads", default=1, type=int, help="number of heads for clustering")
parser.add_argument("--clear_cache_int", default=1, type=int, help="frequency of clear_cache")
parser.add_argument("--num_iters_sk", default=3, type=int, help="number of iters for Sinkhorn")
parser.add_argument("--initial_epsilon_sk", default=0.05, type=float, help="initial epsilon for Sinkhorn")
parser.add_argument("--final_epsilon_sk", default=0.05, type=float, help="final epsilon for Sinkhorn")
parser.add_argument("--adapting_epsilon_sk", default=False, action="store_true", help="use decreasing Sinkhorn epsilon")
parser.add_argument("--queue_start_epoch", default=2, type=int, help="epoch to start using queue; -1 disables")
parser.add_argument("--queue_batches", default=10, type=int, help="number of batches in queue")
parser.add_argument("--queue_percentage", default=0.1, type=float, help="novel points retained in queue")
parser.add_argument("--comment", default=datetime.now().strftime("%b%d_%H-%M-%S"), type=str)
parser.add_argument("--project", default="NCDPC", type=str, help="wandb project")
parser.add_argument("--entity", default=None, type=str, help="wandb entity; leave None to use env/default")
parser.add_argument("--offline", default=False, action="store_true", help="force offline logging")
parser.add_argument("--pretrained", type=str, help="pretrained checkpoint path")
parser.add_argument("--epochs", type=int, default=10, help="training epochs")
parser.add_argument("--set_deterministic", default=False, action="store_true")
parser.add_argument("--alpha", default=1, type=float, help="region loss weight")
parser.add_argument("--mix_pl", default=False, action="store_true", help="parameters for the loss function")
parser.add_argument("--use_reweight", default=False, action="store_true", help="parameters for the loss function")
parser.add_argument("--gamma", type=float, default=10)
parser.add_argument("--num_outer_iters", type=int, default=100)
parser.add_argument("--lr_w", type=float, default=0.1)
parser.add_argument("--gamma_decrease", type=float, default=0.1)
parser.add_argument("--ak_bound", type=float, default=0.005)
parser.add_argument("--smooth_bound", type=int, default=10)
parser.add_argument("--lam", type=float, default=3)
parser.add_argument("--lam_region", type=float, default=4)
parser.add_argument("--use_imbalanced_region", default=False, action="store_true", help="")
parser.add_argument("--use_gt", default=False, action="store_true", help="")
parser.add_argument("--exp_path", default=None, help="")
parser.add_argument("--dbscan", type=float, default=0.5)

parser.add_argument("--hyper_alpha", default=0.25, type=float, help="geometry/semantic similarity fusion weight")
parser.add_argument("--hyper_tau", default=1.0, type=float, help="temperature for geometry RBF similarity")
parser.add_argument("--hyper_topk", default=8, type=int, help="top-k neighbors for hyperedge construction")
parser.add_argument("--hyper_residual", default=0.2, type=float, help="residual strength for hypergraph aggregation")
parser.add_argument("--geo_center_weight", default=0.1, type=float, help="weight for normalized region center coordinates")
parser.add_argument("--disable_hypergraph", default=False, action="store_true", help="disable hypergraph aggregation")
parser.add_argument("--disable_gap", default=False, action="store_true", help="disable geometric cues in GAP")
parser.add_argument("--use_prototype_memory", default=False, action="store_true", help="enable EMA prototype memory")
parser.add_argument("--prototype_momentum", default=0.9, type=float, help="EMA momentum for prototype memory")
parser.add_argument("--label_smoothing", default=0.15, type=float, help="known-class one-hot label smoothing")
parser.add_argument("--seed", default=1234, type=int, help="random seed")


def set_seed(seed, deterministic=False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def build_trainer(args, loggers, callbacks):
    common_kwargs = dict(
        max_epochs=args.epochs,
        logger=loggers,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        log_every_n_steps=50,
    )
    try:
        return pl.Trainer(accelerator="gpu", devices=1, enable_progress_bar=True, **common_kwargs)
    except TypeError:
        try:
            return pl.Trainer(gpus=1, progress_bar_refresh_rate=10, **common_kwargs)
        except TypeError:
            return pl.Trainer(gpus=1, **common_kwargs)


def main(args):
    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.dataset, args.comment)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    print(args)

    run_name = "-".join([f"S{args.split}", "GAP", args.dataset, args.comment])
    wandb_kwargs = dict(save_dir=args.log_dir, name=run_name, project=args.project, offline=args.offline)
    if args.entity is not None:
        wandb_kwargs["entity"] = args.entity
    wandb_logger = WandbLogger(**wandb_kwargs)

    if args.dataset_config is None:
        if args.dataset == "SemanticKITTI":
            args.dataset_config = "config/semkitti_dataset.yaml"
        elif args.dataset == "SemanticPOSS":
            args.dataset_config = "config/semposs_dataset.yaml"
        else:
            raise NameError(f"Dataset {args.dataset} not implemented")

    with open(args.dataset_config, "r") as f:
        dataset_config = yaml.safe_load(f)

    unknown_labels = unk_labels.unknown_labels(split=args.split, dataset_config=dataset_config)
    number_of_unk = len(unknown_labels)
    label_mapping, label_mapping_inv, unknown_label = unk_labels.label_mapping(
        unknown_labels, dataset_config["learning_map_inv"].keys()
    )

    args.num_classes = len(label_mapping)
    args.num_unlabeled_classes = number_of_unk
    args.num_labeled_classes = args.num_classes - args.num_unlabeled_classes

    mIoU_callback = mIoUEvaluatorCallback()
    csv_logger = CSVLogger(save_dir=args.log_dir)
    checkpoint_callback = ModelCheckpoint(
        monitor="valid/mIoU",
        mode="max",
        save_top_k=1,
        save_last=True,
        save_weights_only=True,
        dirpath=args.checkpoint_dir,
        filename=f"{args.dataset}_S{args.split}_GAP_best-{{epoch}}-{{step}}",
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    loggers = [wandb_logger, csv_logger] if wandb_logger is not None else [csv_logger]

    model = Discoverer(label_mapping, label_mapping_inv, unknown_label, **args.__dict__)
    trainer = build_trainer(args, loggers, [mIoU_callback, checkpoint_callback, lr_monitor])
    trainer.fit(model)


if __name__ == "__main__":
    args = parser.parse_args()
    set_seed(args.seed, deterministic=args.set_deterministic)
    main(args)
