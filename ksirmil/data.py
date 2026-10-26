"""Label files, k-fold splits and bag datasets.

Feature layout expected on disk::

    <data_root>/
        label.csv          # "patient,label" (one row per slide / patient)
        pt/                # one <slide>.pt tensor per slide, shape [N, in_dim]
        # pt_files/ and features/ are also accepted

``label.csv`` holds integer labels for ``--task CancerDiagnosis`` (e.g. 0/1) and
subtype names for ``--task SubtypeClassification`` (e.g. ``LUAD`` / ``LUSC``).
"""

from __future__ import annotations

import os
import random
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset

FEATURE_DIRS = ("pt", "pt_files", "features")
SUBTYPE_POSITIVE = {"nsclc": "LUAD", "brca": "IDC"}


def resolve_feature_dir(root: str) -> str:
    """Return the directory that stores the ``.pt`` feature files under ``root``."""
    for name in FEATURE_DIRS:
        path = os.path.join(root, name)
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(
        f"no feature directory found under {root!r}; expected one of {FEATURE_DIRS}"
    )


def read_labels(csv_path: str):
    """Read ``patient,label`` pairs from a csv file."""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"label file not found: {csv_path}")
    frame = pd.read_csv(csv_path)
    if frame.shape[1] < 2:
        raise ValueError(f"{csv_path} must have at least two columns (patient, label)")
    patients = frame.iloc[:, 0].astype(str).to_numpy()
    labels = frame.iloc[:, 1].astype(str).to_numpy()
    print(f"[data] {len(patients)} slides, labels: {dict(Counter(labels))}")
    return patients, labels


def encode_labels(labels, subtype: str = None):
    """Turn raw labels into ``0/1`` integers.

    ``--task CancerDiagnosis`` expects numeric labels; ``--task SubtypeClassification``
    maps the positive subtype (``LUAD`` for NSCLC, ``IDC`` for BRCA) to 0 and the
    remaining subtype to 1, following the original protocol.
    """
    if subtype is None:
        return np.array([int(label) for label in labels], dtype=np.int64)
    positive = SUBTYPE_POSITIVE[subtype.lower()]
    return np.array([0 if label.upper() == positive else 1 for label in labels], dtype=np.int64)


def make_folds(
    k: int,
    patients: np.ndarray,
    labels: np.ndarray,
    val_ratio: float = 0.2,
    use_val: bool = True,
    split_dir: str = None,
):
    """Stratified k-fold split with an optional validation split carved out of train.

    Returns a list of ``{"train": idx, "val": idx, "test": idx}`` index arrays.
    """
    folds = []
    skf = StratifiedKFold(n_splits=k)
    for fold, (train_index, test_index) in enumerate(skf.split(patients, labels)):
        train_index = list(train_index)
        if use_val and val_ratio > 0:
            random.shuffle(train_index)
            n_val = int(len(train_index) * val_ratio)
            val_index = train_index[:n_val]
            train_index = train_index[n_val:]
        else:
            val_index = []

        folds.append(
            {
                "train": np.array(train_index, dtype=np.int64),
                "val": np.array(val_index, dtype=np.int64),
                "test": np.array(test_index, dtype=np.int64),
            }
        )

        if split_dir:
            os.makedirs(split_dir, exist_ok=True)
            pd.DataFrame(
                {
                    "train_patients": patients[train_index],
                    "train_labels": labels[train_index],
                }
            ).to_csv(os.path.join(split_dir, f"fold_{fold}.csv"), index=False)

    return folds


class BagDataset(Dataset):
    """A dataset of bags, one ``.pt`` tensor per slide.

    Args:
        patients: slide (or patient) identifiers.
        labels: integer labels aligned with ``patients``.
        feature_dir: directory holding the ``.pt`` feature files.
        match_by_prefix: when ``True`` a feature file is assigned to a patient if the
            patient id is a prefix of the file name (TCGA slides are named after the
            patient, e.g. ``TCGA-05-4244-01Z-00-DX1.<hash>.pt``).
        cache: keep the loaded features in memory.
    """

    def __init__(
        self,
        patients,
        labels,
        feature_dir: str,
        match_by_prefix: bool = False,
        cache: bool = False,
    ):
        self.feature_dir = feature_dir
        self.cache = cache
        self._cache = {}

        if match_by_prefix:
            files = sorted(f for f in os.listdir(feature_dir) if f.endswith(".pt"))
            stems = [os.path.splitext(f)[0] for f in files]
            paths, bag_labels = [], []
            for patient, label in zip(patients, labels):
                matched = [f for f, stem in zip(files, stems) if str(patient) in stem]
                if not matched:
                    raise FileNotFoundError(f"no feature file matches patient {patient!r} in {feature_dir}")
                for name in matched:
                    paths.append(os.path.join(feature_dir, name))
                    bag_labels.append(int(label))
            self.paths = paths
            self.labels = bag_labels
        else:
            self.paths = [os.path.join(feature_dir, f"{patient}.pt") for patient in patients]
            self.labels = [int(label) for label in labels]
            missing = [p for p in self.paths if not os.path.isfile(p)]
            if missing:
                raise FileNotFoundError(f"{len(missing)} feature files are missing, e.g. {missing[0]}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        if self.cache:
            if path not in self._cache:
                self._cache[path] = torch.load(path, weights_only=True)
            feature = self._cache[path]
        else:
            feature = torch.load(path, weights_only=True)
        return feature, self.labels[index]


def build_dataset(task: str, patients, labels, feature_dir: str, cache: bool = False) -> BagDataset:
    """Build the dataset matching the given task."""
    if task == "CancerDiagnosis":
        return BagDataset(patients, labels, feature_dir, match_by_prefix=False, cache=cache)
    if task == "SubtypeClassification":
        return BagDataset(patients, labels, feature_dir, match_by_prefix=True, cache=cache)
    raise ValueError(f"unknown task {task!r}, choose from 'CancerDiagnosis', 'SubtypeClassification'")
