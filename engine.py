"""Training / evaluation loop and metrics."""

from __future__ import annotations

import os
import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve
from tqdm import tqdm


def seed_everything(seed: int = 2024) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def optimal_threshold(fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray, p: float = 0.0):
    """Operating point that minimises ``(fpr - tpr) - p * tpr / (fpr + tpr + 1)``."""
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    return thresholds[np.argmin(loss, axis=0)]


def classification_scores(labels, probs, threshold: float = None):
    """ACC / AUC / F1 / precision / recall of binary predictions.

    When ``threshold`` is ``None`` the operating point is chosen on the given
    predictions (validation set); otherwise the provided threshold is reused (test set).
    """
    labels = np.asarray(labels)
    probs = np.asarray(probs)[:, 1]
    if threshold is None:
        if len(np.unique(labels)) < 2:  # degenerate validation split: no ROC to tune on
            threshold = 0.5
        else:
            fpr, tpr, thresholds = roc_curve(labels, probs, pos_label=1)
            threshold = optimal_threshold(fpr, tpr, thresholds)
    if not np.isfinite(threshold):  # sklearn prepends an infinite threshold
        threshold = 0.5

    auc = roc_auc_score(labels, probs)
    predictions = (probs >= threshold).astype(int)
    precision, recall, f1score, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    acc = accuracy_score(labels, predictions)
    return {
        "acc": acc,
        "auc": auc,
        "f1": f1score,
        "precision": precision,
        "recall": recall,
        "threshold": float(threshold),
    }


class EarlyStopping:
    """Stop training when the validation loss stops improving."""

    def __init__(self, patience: int = 25, stop_epoch: int = 75):
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, epoch: int, val_loss: float) -> None:
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, criterion, device, n_classes: int, threshold: float = None):
    """Average loss and metrics (plus raw probabilities) over a loader."""
    model.eval()
    bag_probs = np.zeros((len(loader), n_classes))
    bag_labels = np.zeros(len(loader))
    total_loss = 0.0

    for batch, (bag, label) in enumerate(loader):
        bag, label = bag.to(device), label.to(device)
        logits, probs, _, _ = model(bag)
        total_loss += criterion(logits, label).item()
        bag_probs[batch] = probs.cpu().numpy()
        bag_labels[batch] = label.item()

    metrics = classification_scores(bag_labels, bag_probs, threshold=threshold)
    metrics["loss"] = total_loss / len(loader)
    metrics["probs"] = bag_probs
    metrics["labels"] = bag_labels
    return metrics


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for bag, label in loader:
        bag, label = bag.to(device), label.to(device)
        logits, _, _, _ = model(bag)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / len(loader)


def train_fold(
    model,
    loaders,
    criterion,
    optimizer,
    scheduler,
    device,
    n_classes: int,
    epochs: int,
    results_dir: str,
    fold: int,
    patience: int = 25,
    stop_epoch: int = 75,
    verbose: bool = True,
):
    """Train one fold and return its test metrics.

    The checkpoint with the best validation AUC is restored before testing.
    """
    os.makedirs(results_dir, exist_ok=True)
    early_stopping = EarlyStopping(patience=patience, stop_epoch=stop_epoch)
    best_auc, best_metrics = -1.0, None

    for epoch in tqdm(range(epochs), desc=f"fold {fold}", colour="GREEN", disable=not verbose):
        train_loss = train_one_epoch(model, loaders["train"], optimizer, criterion, device)
        val_metrics = evaluate(model, loaders["val"], criterion, device, n_classes)
        if scheduler is not None:
            scheduler.step()

        if verbose:
            tqdm.write(
                f"[fold {fold}] epoch {epoch + 1}/{epochs} train_loss={train_loss:.4f} "
                f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f} "
                f"val_auc={val_metrics['auc']:.4f} val_f1={val_metrics['f1']:.4f}"
            )

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            best_metrics = val_metrics
            torch.save({"model": model.state_dict()}, os.path.join(results_dir, f"fold_{fold}_model_best_auc.pt"))

        early_stopping(epoch, val_metrics["loss"])
        if early_stopping.early_stop:
            if verbose:
                tqdm.write(f"[fold {fold}] early stopping at epoch {epoch + 1}")
            break

    checkpoint = torch.load(os.path.join(results_dir, f"fold_{fold}_model_best_auc.pt"), map_location=device)
    model.load_state_dict(checkpoint["model"])

    test_metrics = evaluate(
        model,
        loaders["test"],
        criterion,
        device,
        n_classes=n_classes,
        threshold=best_metrics["threshold"],
    )
    test_metrics["best_epoch_auc"] = best_auc
    return test_metrics
