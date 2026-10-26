# KSIR-MIL

**Key Region Selection and Instance Refinement for Multi-Instance Learning in Whole Slide
Image Classification** 的官方实现。

KSIR-MIL 把病理医生「先粗看定位、再细看边界」的阅片过程写进网络结构本身：patch 特征铺成
方形网格并划分区域，用 Top-K attention 定位关键区域，再由分类器权重反投影得到 CAM、算出每个
实例的 patch 激活热度（PAH），把关键区域实例按 PAH 分成高低两组分别动态掩码，最后聚合细化后
的实例给出切片标签。可解释性来自前向过程，而不是事后画热力图。

## 方法

| 阶段 | 做什么 | 对应代码 |
| --- | --- | --- |
| 特征平方化 | patch 特征零填充成方形网格，切成 `L x L` 个区域，每区 `M x M` 个实例 | `to_region_grid` |
| 关键区域选择 | 注意力聚合器给所有实例打分，保留 Top-K 实例，凡包含其中任一实例的区域即为关键区域 | `KSIRMIL.select_key_regions` |
| 实例细化 | 注意力加权后的特征经分类器权重点乘得到类激活图，按类别维取最大值即 PAH；关键区域实例按 PAH 均值分成高低两组，各组按各自比例随机掩码 | `KSIRMIL.patch_activation_heat`、`KSIRMIL.refine_instances` |
| 包分类 | 细化后的实例聚合、分类 | `KSIRMIL.forward` |

细化只在训练态生效。推理时直接用完整实例序列聚合，因此模型行为与常规注意力 MIL 一致，可直接
接入已有流水线，推理开销也没有增加。

聚合器可替换：`--aggregator abmil`（默认）或 `--aggregator abmil_gated`。

## 仓库结构

```
KSIR-MIL/
├── ksirmil/
│   ├── model.py     # KSIR-MIL、聚合器、区域划分、PAH
│   ├── data.py      # label 文件、k 折划分、bag 数据集
│   └── engine.py    # 训练/评估循环、指标、早停
├── train.py         # k 折训练入口
├── infer.py         # 单张切片推理 + 解释结果导出
├── requirements.txt
└── README.md
```

## 环境

```bash
pip install -r requirements.txt
```

## 数据准备

WSI 特征只需离线抽取一次。流程沿用 CLAM 的预处理管线
（[github.com/mahmoodlab/CLAM](https://github.com/mahmoodlab/CLAM)）：`create_patches_fp.py`
做组织分割与切块，`extract_features_fp.py` 配 `resnet50_trunc` 抽出每个 patch 的 1024 维特征。
用 PLIP 特征是 512 维，需要 `--in_dim 512`。

期望的目录结构：

```
data/<dataset>/<extractor>/
├── label.csv          # patient,label，一行一个切片
└── pt/                # <slide>.pt，每个切片一个 [N, in_dim] 张量
```

特征目录也可以叫 `pt_files/` 或 `features/`，脚本会自动识别。

`--task CancerDiagnosis` 用整数标签：

```csv
patient,label
tumor_020,1
normal_052,0
```

`--task SubtypeClassification` 用亚型名（`LUAD` 映射为 0，另一亚型映射为 1）：

```csv
patient,label
TCGA-62-A46R,LUAD
TCGA-05-4417,LUSC
```

## 训练

CAMELYON-16，3 折交叉验证：

```bash
python train.py --task CancerDiagnosis --dataset Camelyon16 \
    --data_root data/Camelyon16/resnet50 --in_dim 1024 \
    --k 3 --val_ratio 0.2 --region_type 1d --region_num 2 --topk 8 \
    --epochs 100 --lr 2e-4 --weight_decay 2e-5 --seed 2024
```

TCGA-NSCLC，4 折交叉验证，train/val/test 按 65:10:25 划分：

```bash
python train.py --task SubtypeClassification --dataset TCGA-NSCLC --subtype nsclc \
    --data_root data/TCGA-NSCLC/resnet50 --in_dim 1024 \
    --k 4 --val_ratio 0.1333 --region_type 1d --region_num 8 --topk 8 \
    --epochs 100 --lr 2e-4 --weight_decay 2e-5 --seed 2024
```

每一折用包级交叉熵训练，按验证集 AUC 保留最优权重，在验证集上确定判决阈值，再用这个阈值报告
测试集指标。输出：

```
results/<task>_<dataset>_seed<seed>/
├── fold_<i>_model_best_auc.pt
└── summary.csv          # 各折 ACC / AUC / F1 / precision / recall，以及 mean、std
```

折划分同时写到 `splits/<task>_<dataset>_seed<seed>/fold_<i>.csv`。

常用开关：

| 参数 | 含义 |
| --- | --- |
| `--region_type 1d --region_num L` | 固定网格边长 `L`（论文设置） |
| `--region_type 2d --region_size M` | 固定区域边长 `M` |
| `--topk K` | 驱动区域选择的 Top-K 实例数（论文取 8） |
| `--group_type {high,low,mix}` | 保留高 PAH 组、低 PAH 组，或两组都保留 |
| `--high_mask_ratio` / `--low_mask_ratio` | 两组的掩码比例（论文取 0.5 / 0.5） |
| `--no_group_mask` | 关闭实例细化（消融用） |
| `--aggregator {abmil,abmil_gated}` | 包聚合器 |

## 推理与解释

```bash
python infer.py --checkpoint results/CancerDiagnosis_Camelyon16_seed2024/fold_0_model_best_auc.pt \
    --features data/Camelyon16/resnet50/pt/tumor_001.pt \
    --in_dim 1024 --region_num 2 --out_dir predictions
```

`predictions/predictions.csv` 是切片级预测和概率。`predictions/<slide>_instances.csv` 逐 patch
给出网格行列位置、所属区域编号、注意力分数、PAH，以及所在区域是否被选为关键区域——这份表就是
画热力图所需的原始数据。`--features` 既可以传单个 `.pt`，也可以传一个目录。

## 论文结果

CAMELYON-16（3 折）与 TCGA-NSCLC（4 折），ACC / AUC / F1：

| 方法 | CAMELYON-16 | TCGA-NSCLC |
| --- | --- | --- |
| ABMIL | 0.857 / 0.909 / 0.815 | 0.856 / 0.930 / 0.852 |
| MHIM-MIL (ABMIL) | 0.857 / 0.929 / 0.825 | 0.864 / 0.936 / 0.862 |
| KSIR-MIL (ABMIL) | 0.871 / 0.940 / 0.844 | 0.876 / 0.944 / 0.873 |
| KSIR-MIL (ABMIL-Gated) | 0.873 / 0.939 / 0.842 | 0.889 / 0.947 / 0.880 |
| KSIR-MIL (CLAM-MB) | 0.882 / 0.936 / 0.851 | 0.886 / 0.950 / 0.873 |

主要超参：Adam，学习率 `2e-4`，权重衰减 `2e-5`，100 epoch，batch size 1，实例投影维度 512，
dropout 0.25，`K = 8`，`high_mask_ratio = low_mask_ratio = 0.5`。

## 实现说明

- 本仓库发布 ABMIL 与 ABMIL-Gated 两种聚合器；框架在 CLAM-MB 上的结果见论文，对应聚合器不在
  本次发布范围内。
- 两组掩码比例以固定超参给出（默认 0.5，与论文初值一致），不参与梯度更新——按比例做伯努利掩码
  本身不可导，靠反传无法调整比例。
- 两处边界保护：某个分组被掩码成空时回退到关键区域实例；验证集只有单一类别或 sklearn 返回
  无穷阈值时回退 0.5，避免空 bag 和退化指标。

## 引用

```bibtex
@article{ksirmil,
  title   = {KSIR-MIL: Key Region Selection and Instance Refinement for Multi-Instance
             Learning in Whole Slide Image Classification},
  author  = {Cui, Shaoguo and Wu, Jiangfeng and Sang, Binbin and Li, Tiansong and
             Zhang, Yi and Cheng, Fumin and Wang, Guofen},
  year    = {2025}
}
```

## 致谢

特征抽取管线和 CAMELYON-16 / TCGA-NSCLC 的特征文件沿用
[CLAM](https://github.com/mahmoodlab/CLAM)；实例分组掩码的思路与
[MHIM-MIL](https://github.com/Jiangbo-Shi/MHIM-MIL) 相关。
