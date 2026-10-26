"""Train KSIR-MIL with stratified k-fold cross validation.

Example (CAMELYON-16, 3 folds)::

    python train.py --task CancerDiagnosis --dataset Camelyon16 \
        --data_root data/Camelyon16/resnet50 --in_dim 1024 \
        --k 3 --region_type 1d --region_num 2 --epochs 100
"""

from __future__ import annotations

import argparse
import os
import random

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from ksirmil.data import build_dataset, encode_labels, make_folds, read_labels, resolve_feature_dir
from ksirmil.engine import seed_everything, train_fold
from ksirmil.model import KSIRMIL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KSIR-MIL training")

    # data
    parser.add_argument("--task", default="CancerDiagnosis", type=str,
                        help="[CancerDiagnosis, SubtypeClassification]")
    parser.add_argument("--dataset", default="Camelyon16", type=str,
                        help="dataset name, only used to name the output directory")
    parser.add_argument("--data_root", default="data/Camelyon16/resnet50", type=str,
                        help="directory holding label.csv and the .pt feature folder")
    parser.add_argument("--subtype", default="nsclc", type=str, help="[nsclc, brca]")
    parser.add_argument("--cache", action="store_true", help="keep features in memory")
    parser.add_argument("--num_workers", default=2, type=int, help="dataloader workers")

    # cross validation
    parser.add_argument("--k", default=3, type=int, help="number of folds")
    parser.add_argument("--k_start", default=0, type=int, help="first fold to run")
    parser.add_argument("--val_ratio", default=0.2, type=float, help="validation ratio carved out of train")
    parser.add_argument("--no_val", action="store_true", help="evaluate on the test split instead of a val split")
    parser.add_argument("--split_dir", default="splits", type=str, help="where the fold csv files are written")

    # optimisation
    parser.add_argument("--epochs", default=100, type=int, help="number of epochs")
    parser.add_argument("--batch_size", default=1, type=int, help="bags per batch")
    parser.add_argument("--lr", default=2e-4, type=float, help="learning rate")
    parser.add_argument("--weight_decay", default=2e-5, type=float, help="weight decay")
    parser.add_argument("--opt", default="adam", type=str, help="[adam, adamw]")
    parser.add_argument("--lr_scheduler", default="cosine", type=str, help="[cosine, step, const]")
    parser.add_argument("--patience", default=25, type=int, help="early stopping patience")
    parser.add_argument("--stop_epoch", default=75, type=int, help="earliest epoch allowed to stop")
    parser.add_argument("--seed", default=2024, type=int, help="random seed")

    # model
    parser.add_argument("--in_dim", default=1024, type=int, help="patch feature dim (1024 ResNet50, 512 PLIP)")
    parser.add_argument("--n_classes", default=2, type=int, help="number of classes")
    parser.add_argument("--feat_dim", default=512, type=int, help="projected instance feature dim")
    parser.add_argument("--aggregator", default="abmil", type=str, help="[abmil, abmil_gated]")
    parser.add_argument("--region_type", default="1d", type=str,
                        help="[1d] region_num is the grid side, [2d] region_size is the region side")
    parser.add_argument("--region_num", default=2, type=int, help="grid side length L for --region_type 1d")
    parser.add_argument("--region_size", default=8, type=int, help="region side length M for --region_type 2d")
    parser.add_argument("--topk", default=8, type=int, help="number of Top-K instances for key region selection")
    parser.add_argument("--no_group_mask", action="store_true", help="disable the instance grouping dynamic mask")
    parser.add_argument("--group_type", default="mix", type=str, help="[high, low, mix]")
    parser.add_argument("--high_mask_ratio", default=0.5, type=float, help="masking ratio of the high-PAH group")
    parser.add_argument("--low_mask_ratio", default=0.5, type=float, help="masking ratio of the low-PAH group")
    parser.add_argument("--dropout", default=0.25, type=float, help="dropout in the projection head")

    # output
    parser.add_argument("--output_dir", default="results", type=str, help="root output directory")
    parser.add_argument("--quiet", action="store_true", help="disable progress bars and per-epoch logs")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_dir = os.path.join(args.output_dir, f"{args.task}_{args.dataset}_seed{args.seed}")
    split_dir = os.path.join(args.split_dir, f"{args.task}_{args.dataset}_seed{args.seed}")
    os.makedirs(results_dir, exist_ok=True)

    feature_dir = resolve_feature_dir(args.data_root)
    patients, raw_labels = read_labels(os.path.join(args.data_root, "label.csv"))
    subtype = None if args.task == "CancerDiagnosis" else args.subtype
    labels = encode_labels(raw_labels, subtype)

    # the patient order drives the (unshuffled) stratified folds, so shuffle it per seed
    order = list(range(len(patients)))
    random.shuffle(order)
    patients, labels = patients[order], labels[order]

    folds = make_folds(
        args.k,
        patients,
        labels,
        val_ratio=args.val_ratio,
        use_val=not args.no_val,
        split_dir=split_dir,
    )

    print(
        f"task={args.task} dataset={args.dataset} seed={args.seed} "
        f"slides={len(patients)} folds={args.k} device={device}"
    )

    fold_metrics = []
    for fold in range(args.k_start, args.k):
        loaders = {}
        for split in ("train", "val", "test"):
            index = folds[fold][split] if folds[fold][split].size else folds[fold]["test"]
            dataset = build_dataset(args.task, patients[index], labels[index], feature_dir, cache=args.cache)
            loaders[split] = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.num_workers)

        model = KSIRMIL(
            in_dim=args.in_dim,
            n_classes=args.n_classes,
            feat_dim=args.feat_dim,
            aggregator=args.aggregator,
            region_type=args.region_type,
            region_num=args.region_num,
            region_size=args.region_size,
            topk=args.topk,
            group_mask=not args.no_group_mask,
            group_type=args.group_type,
            high_mask_ratio=args.high_mask_ratio,
            low_mask_ratio=args.low_mask_ratio,
            dropout=args.dropout,
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = (
            torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            if args.opt == "adamw"
            else torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        )
        if args.lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, 0)
        elif args.lr_scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, max(1, args.epochs // 2), 0.2)
        else:
            scheduler = None

        metrics = train_fold(
            model,
            loaders,
            criterion,
            optimizer,
            scheduler,
            device,
            n_classes=args.n_classes,
            epochs=args.epochs,
            results_dir=results_dir,
            fold=fold,
            patience=args.patience,
            stop_epoch=args.stop_epoch,
            verbose=not args.quiet,
        )
        print(
            f"[fold {fold}] ACC {metrics['acc']:.4f} AUC {metrics['auc']:.4f} "
            f"F1 {metrics['f1']:.4f} (threshold {metrics['threshold']:.4f})"
        )
        fold_metrics.append(
            {
                "fold": fold,
                **{key: metrics[key] for key in ("acc", "auc", "f1", "precision", "recall", "threshold")},
            }
        )

    columns = {"acc": "Acc", "auc": "Auc", "f1": "F1Score", "precision": "Precision", "recall": "Recall"}
    summary = pd.DataFrame(fold_metrics).rename(columns={"fold": "Fold", **columns})
    metric_cols = list(columns.values())
    mean, std = summary[metric_cols].mean(), summary[metric_cols].std()
    summary = pd.concat(
        [
            summary,
            pd.DataFrame([{"Fold": "Mean", **mean.to_dict()}, {"Fold": "Std", **std.to_dict()}]),
        ],
        ignore_index=True,
    )
    summary.to_csv(os.path.join(results_dir, "summary.csv"), index=False)

    print(
        f"mean over {len(fold_metrics)} folds: "
        + ", ".join(f"{col} {mean[col]:.4f}+-{std[col]:.4f}" for col in metric_cols)
    )
    print(f"results written to {results_dir}")


if __name__ == "__main__":
    main()
