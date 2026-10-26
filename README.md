# KSIR-MIL

Reference implementation of **KSIR-MIL: Key Region Selection and Instance Refinement for
Multi-Instance Learning in Whole Slide Image Classification**.

KSIR-MIL turns the pathologist's coarse-to-fine slide review into the model architecture:
patch features are laid out on a square grid, key regions are located with a Top-K
attention, the instances inside those regions are refined with a CAM-driven grouping mask,
and only the refined instances are aggregated into a bag label. Interpretability therefore
comes from the forward pass itself rather than from a post-hoc heatmap.

## Method

| stage | what happens | code |
| --- | --- | --- |
| Instance feature squaring | patch features are zero-padded to a square grid and split into `L x L` regions of `M x M` instances | `to_region_grid` |
| Key region selection | an attention aggregator scores every instance, the Top-K instances are kept, and every region containing at least one of them becomes a key region | `KSIRMIL.select_key_regions` |
| Instance refinement | attention-weighted features are projected through the classifier weights into a class activation map, its class-wise maximum gives the Patch Activation Heat (PAH); key-region instances are split into a high-PAH and a low-PAH group and each group is randomly masked at its own ratio | `KSIRMIL.patch_activation_heat`, `KSIRMIL.refine_instances` |
| Bag prediction | the refined instances are aggregated and classified | `KSIRMIL.forward` |

Refinement only runs in training mode. At inference time the full instance sequence is
aggregated, so the model behaves like a standard attention-based MIL model and can be
dropped into an existing pipeline.

The aggregator is pluggable: `--aggregator abmil` (default) or `--aggregator abmil_gated`.

## Repository layout

```
KSIR-MIL/
├── ksirmil/
│   ├── model.py     # KSIR-MIL, aggregators, region partition, PAH
│   ├── data.py      # label files, k-fold splits, bag datasets
│   └── engine.py    # training / evaluation loop, metrics, early stopping
├── train.py         # k-fold training entry point
├── infer.py         # single-slide inference + explanation export
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Data preparation

WSI features are extracted offline, once. We follow the CLAM preprocessing pipeline
([github.com/mahmoodlab/CLAM](https://github.com/mahmoodlab/CLAM)): `create_patches_fp.py`
for tissue segmentation and patching, then `extract_features_fp.py` with
`resnet50_trunc` to obtain one 1024-dim feature vector per patch (PLIP features are 512-dim,
use `--in_dim 512`).

Expected layout:

```
data/<dataset>/<extractor>/
├── label.csv          # patient,label -- one row per slide
└── pt/                # <slide>.pt, a [N, in_dim] tensor per slide
```

`pt_files/` and `features/` are accepted as alternatives to `pt/`.

`label.csv` holds integers for `--task CancerDiagnosis`:

```csv
patient,label
tumor_020,1
normal_052,0
```

and subtype names for `--task SubtypeClassification` (`LUAD` is mapped to 0, the other
subtype to 1):

```csv
patient,label
TCGA-62-A46R,LUAD
TCGA-05-4417,LUSC
```

## Training

CAMELYON-16, 3-fold cross validation:

```bash
python train.py --task CancerDiagnosis --dataset Camelyon16 \
    --data_root data/Camelyon16/resnet50 --in_dim 1024 \
    --k 3 --val_ratio 0.2 --region_type 1d --region_num 2 --topk 8 \
    --epochs 100 --lr 2e-4 --weight_decay 2e-5 --seed 2024
```

TCGA-NSCLC, 4-fold cross validation with a 65:10:25 train/val/test split:

```bash
python train.py --task SubtypeClassification --dataset TCGA-NSCLC --subtype nsclc \
    --data_root data/TCGA-NSCLC/resnet50 --in_dim 1024 \
    --k 4 --val_ratio 0.1333 --region_type 1d --region_num 8 --topk 8 \
    --epochs 100 --lr 2e-4 --weight_decay 2e-5 --seed 2024
```

Each fold trains with cross-entropy on the bag label, keeps the checkpoint with the best
validation AUC, tunes the decision threshold on the validation split and reports the test
metrics with that threshold. Outputs:

```
results/<task>_<dataset>_seed<seed>/
├── fold_<i>_model_best_auc.pt
└── summary.csv          # per-fold ACC / AUC / F1 / precision / recall + mean, std
```

The fold assignment is written to `splits/<task>_<dataset>_seed<seed>/fold_<i>.csv`.

Useful switches:

| flag | meaning |
| --- | --- |
| `--region_type 1d --region_num L` | grid side length `L` is fixed (paper setting) |
| `--region_type 2d --region_size M` | region side length `M` is fixed |
| `--topk K` | number of Top-K instances driving the region selection (paper: 8) |
| `--group_type {high,low,mix}` | which PAH group is kept, or both |
| `--high_mask_ratio` / `--low_mask_ratio` | masking ratio of each group (paper: 0.5 / 0.5) |
| `--no_group_mask` | switch the instance refinement off (ablation) |
| `--aggregator {abmil,abmil_gated}` | bag aggregator |

## Inference and explanation

```bash
python infer.py --checkpoint results/CancerDiagnosis_Camelyon16_seed2024/fold_0_model_best_auc.pt \
    --features data/Camelyon16/resnet50/pt/tumor_001.pt \
    --in_dim 1024 --region_num 2 --out_dir predictions
```

`predictions/predictions.csv` holds the bag prediction and its probability.
`predictions/<slide>_instances.csv` holds, for every patch, its grid position, its region
index, its attention score, its PAH and whether its region was selected — the raw material
for a heatmap. A single `.pt` file or a folder of them can be passed to `--features`.

## Results reported in the paper

CAMELYON-16 (3-fold) and TCGA-NSCLC (4-fold), ACC / AUC / F1:

| method | CAMELYON-16 | TCGA-NSCLC |
| --- | --- | --- |
| ABMIL | 0.857 / 0.909 / 0.815 | 0.856 / 0.930 / 0.852 |
| MHIM-MIL (ABMIL) | 0.857 / 0.929 / 0.825 | 0.864 / 0.936 / 0.862 |
| KSIR-MIL (ABMIL) | 0.871 / 0.940 / 0.844 | 0.876 / 0.944 / 0.873 |
| KSIR-MIL (ABMIL-Gated) | 0.873 / 0.939 / 0.842 | 0.889 / 0.947 / 0.880 |
| KSIR-MIL (CLAM-MB) | 0.882 / 0.936 / 0.851 | 0.886 / 0.950 / 0.873 |

This release ships the ABMIL and ABMIL-Gated aggregators; the CLAM-MB variant of the
framework is not part of the released code.

Main hyper-parameters: Adam, learning rate `2e-4`, weight decay `2e-5`, 100 epochs,
batch size 1, projected instance dimension 512, dropout 0.25, `K = 8`,
`high_mask_ratio = low_mask_ratio = 0.5`.

## Citation

```bibtex
@article{ksirmil,
  title   = {KSIR-MIL: Key Region Selection and Instance Refinement for Multi-Instance
             Learning in Whole Slide Image Classification},
  author  = {Cui, Shaoguo and Wu, Jiangfeng and Sang, Binbin and Li, Tiansong and
             Zhang, Yi and Cheng, Fumin and Wang, Guofen},
  year    = {2025}
}
```

## Acknowledgement

The feature extraction pipeline and the CAMELYON-16 / TCGA-NSCLC feature files follow
[CLAM](https://github.com/mahmoodlab/CLAM). The instance-group masking idea is related to
[MHIM-MIL](https://github.com/Jiangbo-Shi/MHIM-MIL).
