"""Inference and explanation for a single slide (or a folder of slides).

The model only needs patch features, so this script runs without any WSI. For every
slide it reports the bag prediction plus the interpretable intermediate results of
the framework: per-instance attention, Patch Activation Heat and the key regions
selected by the Top-K attention.

Example::

    python infer.py --checkpoint results/CancerDiagnosis_Camelyon16_seed2024/fold_0_model_best_auc.pt \
        --features data/Camelyon16/resnet50/pt/tumor_001.pt --in_dim 1024 --out_dir predictions
"""

from __future__ import annotations

import argparse
import glob
import os

import pandas as pd
import torch

from ksirmil.model import KSIRMIL, to_region_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KSIR-MIL inference")
    parser.add_argument("--checkpoint", required=True, type=str, help="path to a fold checkpoint")
    parser.add_argument("--features", required=True, type=str, help="a .pt feature file or a folder of them")
    parser.add_argument("--out_dir", default="predictions", type=str, help="output directory")
    parser.add_argument("--class_names", default="normal,tumor", type=str, help="comma separated class names")

    parser.add_argument("--in_dim", default=1024, type=int, help="patch feature dim")
    parser.add_argument("--n_classes", default=2, type=int, help="number of classes")
    parser.add_argument("--feat_dim", default=512, type=int, help="projected instance feature dim")
    parser.add_argument("--aggregator", default="abmil", type=str, help="[abmil, abmil_gated]")
    parser.add_argument("--region_type", default="1d", type=str, help="[1d, 2d]")
    parser.add_argument("--region_num", default=2, type=int, help="grid side length L")
    parser.add_argument("--region_size", default=8, type=int, help="region side length M")
    parser.add_argument("--topk", default=8, type=int, help="Top-K instances for key region selection")
    parser.add_argument("--dropout", default=0.25, type=float, help="dropout in the projection head")
    return parser.parse_args()


def load_model(args, device) -> KSIRMIL:
    model = KSIRMIL(
        in_dim=args.in_dim,
        n_classes=args.n_classes,
        feat_dim=args.feat_dim,
        aggregator=args.aggregator,
        region_type=args.region_type,
        region_num=args.region_num,
        region_size=args.region_size,
        topk=args.topk,
        dropout=args.dropout,
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def collect_features(path: str):
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.pt")))
        if not files:
            raise FileNotFoundError(f"no .pt files found in {path}")
        return files
    return [path]


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    model = load_model(args, device)
    class_names = [name.strip() for name in args.class_names.split(",")]

    predictions = []
    for path in collect_features(args.features):
        features = torch.load(path, weights_only=True).to(device)
        if features.dim() == 2:
            features = features.unsqueeze(0)

        with torch.no_grad():
            logits, probs, _, _ = model(features)
            attention, pah, region_mask = model.attribute(features)

        n_instances = features.shape[1]
        _, num_regions, region_side = to_region_grid(
            features.new_zeros(1, n_instances, 1),
            region_num=args.region_num if args.region_type == "1d" else None,
            region_size=args.region_size if args.region_type == "2d" else None,
        )
        grid = num_regions * region_side

        probs = probs.squeeze(0).cpu().numpy()
        pred = int(probs.argmax())
        predictions.append(
            {
                "slide": os.path.splitext(os.path.basename(path))[0],
                "prediction": class_names[pred] if pred < len(class_names) else pred,
                "probability": float(probs[pred]),
                **{f"prob_{name}": float(p) for name, p in zip(class_names, probs)},
                "num_instances": int(n_instances),
                "num_key_regions": int(region_mask.numel()),
                "num_regions": int(num_regions * num_regions),
            }
        )

        index = torch.arange(n_instances)
        instances = pd.DataFrame(
            {
                "instance": index.numpy(),
                "row": (index // grid).numpy(),
                "col": (index % grid).numpy(),
                "region": ((index // grid) // region_side * num_regions + (index % grid) // region_side).numpy(),
                "attention": attention.cpu().numpy(),
                "pah": pah.cpu().numpy(),
            }
        )
        instances["in_key_region"] = instances["region"].isin(region_mask.cpu().tolist())
        instances.to_csv(
            os.path.join(args.out_dir, f"{predictions[-1]['slide']}_instances.csv"), index=False
        )
        print(
            f"{predictions[-1]['slide']}: {predictions[-1]['prediction']} "
            f"(p={predictions[-1]['probability']:.4f}), "
            f"{predictions[-1]['num_key_regions']}/{predictions[-1]['num_regions']} key regions kept"
        )

    pd.DataFrame(predictions).to_csv(os.path.join(args.out_dir, "predictions.csv"), index=False)
    print(f"predictions written to {os.path.join(args.out_dir, 'predictions.csv')}")


if __name__ == "__main__":
    main()
