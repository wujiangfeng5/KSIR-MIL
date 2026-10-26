"""KSIR-MIL: Key Region Selection and Instance Refinement for MIL-based WSI classification.

The model follows a coarse-to-fine diagnostic procedure:

1. **Instance feature squaring** -- patch features of a bag are padded to a square
   grid and split into ``L x L`` regions of ``M x M`` instances each.
2. **Key region selection** -- an attention aggregator scores every instance, the
   Top-K instances are kept, and every region that contains at least one of them
   is treated as a key region.
3. **Instance refinement** -- the Patch Activation Heat (PAH) of every key-region
   instance is computed from the class activation map, instances are grouped into a
   high-PAH and a low-PAH group, and each group is randomly masked at its own ratio.
4. **Bag aggregation** -- the refined instances are aggregated and classified.

Refinement is only applied while training; at inference time the complete instance
sequence is aggregated, so the model stays a drop-in replacement inside a MIL
pipeline.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_weights(module: nn.Module) -> None:
    """Xavier init for linear/conv layers, identity init for LayerNorm."""
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class ABMIL(nn.Module):
    """Additive attention aggregator (Ilse et al., 2018)."""

    def __init__(self, in_dim: int = 512, hidden_dim: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor):
        """x: ``[N, D]`` or ``[B, N, D]`` -> bag feature, attention scores."""
        a_score = self.attention(x).transpose(-1, -2)  # [N] or [B, 1, N]
        bag = torch.matmul(F.softmax(a_score, dim=-1), x)
        return bag, a_score


class ABMILGated(nn.Module):
    """Gated additive attention aggregator (Ilse et al., 2018)."""

    def __init__(self, in_dim: int = 512, hidden_dim: int = 128):
        super().__init__()
        self.attention_a = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Tanh())
        self.attention_b = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Sigmoid())
        self.attention_c = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        a = self.attention_a(x) * self.attention_b(x)
        a_score = self.attention_c(a).transpose(-1, -2)
        bag = torch.matmul(F.softmax(a_score, dim=-1), x)
        return bag, a_score


AGGREGATORS = {"abmil": ABMIL, "abmil_gated": ABMILGated}


def region_partition(x: torch.Tensor, region_size: int):
    """Split ``[B, H, W, D]`` into ``[B * (H/r) * (W/r), r, r, D]`` regions."""
    b, h, w, d = x.shape
    r = region_size
    x = x.view(b, h // r, r, w // r, r, d)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(-1, r, r, d)


def to_region_grid(x: torch.Tensor, region_num: int = None, region_size: int = None):
    """Pad ``x`` into a square grid and partition it into regions.

    Two equivalent parameterisations are supported:

    * ``region_num=L`` -- the grid side length is fixed and the region side length
      becomes ``M = ceil(sqrt(N)) / L`` (used by ``--region_type 1d``);
    * ``region_size=M`` -- the region side length is fixed and the grid side length
      becomes ``L = ceil(sqrt(N)) / M`` (used by ``--region_type 2d``).

    Args:
        x: ``[B, N, D]`` instance features.
        region_num: grid side length ``L``.
        region_size: region side length ``M``.

    Returns:
        ``(regions, L, M)`` where ``regions`` has shape ``[B * L * L, M * M, D]``.
    """
    b, n, d = x.shape
    side = int(math.ceil(math.sqrt(n)))

    if region_num is not None:
        num_regions = int(region_num)
        grid = side + (-side) % num_regions
        region_side = grid // num_regions
    else:
        region_side = int(region_size)
        grid = side + (-side) % region_side
        num_regions = grid // region_side

    num_pad = grid * grid - n
    if num_pad:
        x = torch.cat([x, x.new_zeros(b, num_pad, d)], dim=1)

    regions = region_partition(x.view(b, grid, grid, d), region_side)
    regions = regions.view(b * num_regions * num_regions, region_side * region_side, d)
    return regions, num_regions, region_side


def class_activation_map(classifier: nn.Linear, features: torch.Tensor) -> torch.Tensor:
    """Patch Activation Heat: class-wise CAM followed by a max over classes.

    Args:
        classifier: the bag classifier (``nn.Linear``) providing the weights.
        features: attention-weighted instance features ``[B, N, D]``.

    Returns:
        ``[B, N]`` activation intensity of every instance.
    """
    cam = torch.einsum("bnd,cd->bcn", features, classifier.weight)
    if classifier.bias is not None:
        cam = cam + classifier.bias.view(1, -1, 1)
    cam = F.softmax(cam, dim=1)
    return cam.max(dim=1).values


class KSIRMIL(nn.Module):
    """Key Region Selection and Instance Refinement MIL.

    Args:
        in_dim: patch feature dimension (1024 for ResNet50, 512 for PLIP).
        n_classes: number of bag classes.
        feat_dim: dimension of the projected instance features.
        aggregator: ``"abmil"`` or ``"abmil_gated"``.
        region_type: ``"1d"`` (``region_num`` is the grid side length) or
            ``"2d"`` (``region_size`` is the region side length).
        region_num: grid side length ``L`` for ``region_type="1d"``.
        region_size: region side length ``M`` for ``region_type="2d"``.
        topk: number of Top-K instances driving the key region selection.
        group_mask: enable the instance grouping dynamic mask.
        group_type: which instance group to keep, ``"high"``, ``"low"`` or ``"mix"``.
        high_mask_ratio: masking ratio applied to the high-PAH group.
        low_mask_ratio: masking ratio applied to the low-PAH group.
        dropout: dropout in the feature projection head.
    """

    def __init__(
        self,
        in_dim: int = 1024,
        n_classes: int = 2,
        feat_dim: int = 512,
        aggregator: str = "abmil",
        region_type: str = "1d",
        region_num: int = 2,
        region_size: int = 8,
        topk: int = 8,
        group_mask: bool = True,
        group_type: str = "mix",
        high_mask_ratio: float = 0.5,
        low_mask_ratio: float = 0.5,
        dropout: float = 0.25,
    ):
        super().__init__()
        if aggregator not in AGGREGATORS:
            raise ValueError(f"unknown aggregator {aggregator!r}, choose from {list(AGGREGATORS)}")
        if group_type not in ("high", "low", "mix"):
            raise ValueError(f"unknown group_type {group_type!r}, choose from 'high', 'low', 'mix'")
        if not 0.0 <= high_mask_ratio <= 1.0 or not 0.0 <= low_mask_ratio <= 1.0:
            raise ValueError("masking ratios must lie in [0, 1]")

        self.dim_reduction = nn.Sequential(
            nn.Linear(in_dim, feat_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.aggregator = AGGREGATORS[aggregator](feat_dim)
        self.classifier = nn.Linear(feat_dim, n_classes)

        self.region_type = region_type
        self.region_num = region_num
        self.region_size = region_size
        self.topk = topk
        self.group_mask = group_mask
        self.group_type = group_type
        self.high_mask_ratio = high_mask_ratio
        self.low_mask_ratio = low_mask_ratio

        self.apply(initialize_weights)

    # ------------------------------------------------------------------ regions
    def _squaring(self, x: torch.Tensor):
        if self.region_type == "1d":
            return to_region_grid(x, region_num=self.region_num)
        return to_region_grid(x, region_size=self.region_size)

    def select_key_regions(self, x: torch.Tensor):
        """Coarse stage: keep the regions that contain at least one Top-K instance.

        Args:
            x: projected instance features ``[1, N, D]``.

        Returns:
            ``(roi_instances, region_mask)`` -- the instances of the key regions and
            the indices of the retained regions.
        """
        _, a_score = self.aggregator(x.reshape(-1, x.shape[-1]))
        attention = a_score.squeeze(0)  # [N]

        regions, _, _ = self._squaring(x)
        keep = torch.zeros_like(attention)
        keep[torch.topk(attention, min(self.topk, attention.numel())).indices] = 1.0

        keep_regions, _, _ = self._squaring(keep.view(1, -1, 1))
        region_hit = keep_regions.flatten(1).sum(dim=1) > 0
        region_mask = torch.where(region_hit)[0]
        if region_mask.numel() == 0:  # degenerate case: fall back to the hottest region
            region_mask = keep_regions.flatten(1).sum(dim=1).argmax().view(1)

        return regions[region_mask].reshape(-1, regions.shape[-1]), region_mask

    # ------------------------------------------------------------- significance
    def patch_activation_heat(self, x: torch.Tensor) -> torch.Tensor:
        """PAH of every instance, ``[B, N]``."""
        if x.dim() == 2:
            x = x.unsqueeze(0)
        _, a_score = self.aggregator(x)
        weights = F.softmax(a_score, dim=-1).squeeze(1)  # [B, N]
        weighted = torch.einsum("bnd,bn->bnd", x, weights)
        return class_activation_map(self.classifier, weighted)

    # ---------------------------------------------------------------- refinement
    @staticmethod
    def _random_mask(x: torch.Tensor, drop_ratio: float) -> torch.Tensor:
        if drop_ratio <= 0.0 or x.shape[0] == 0:
            return x
        keep = torch.rand(x.shape[0], device=x.device) >= drop_ratio
        masked = x[keep]
        return masked if masked.shape[0] > 0 else x  # never empty out a whole group

    def refine_instances(self, x: torch.Tensor, roi_instances: torch.Tensor, region_mask: torch.Tensor):
        """Fine stage: group key-region instances by PAH and mask each group."""
        pah = self.patch_activation_heat(x).unsqueeze(-1)  # [1, N, 1]
        pah_regions, _, _ = self._squaring(pah)
        roi_pah = pah_regions[region_mask].reshape(-1)

        threshold = roi_pah.mean()
        high = roi_instances[roi_pah >= threshold]
        low = roi_instances[roi_pah < threshold]

        if self.group_type == "high":
            refined = self._random_mask(high, self.high_mask_ratio)
        elif self.group_type == "low":
            refined = self._random_mask(low, self.low_mask_ratio)
        else:
            refined = torch.cat(
                [
                    self._random_mask(high, self.high_mask_ratio),
                    self._random_mask(low, self.low_mask_ratio),
                ],
                dim=0,
            )
        # a single-instance key region can empty a whole group: keep the bag usable
        return refined if refined.shape[0] else roi_instances

    # ------------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor):
        """x: ``[1, N, in_dim]`` -> ``(logits, probs, preds, attention)``."""
        x = self.dim_reduction(x)
        if self.training:
            roi_instances, region_mask = self.select_key_regions(x)
            x = self.refine_instances(x, roi_instances, region_mask) if self.group_mask else roi_instances

        bag, attention = self.aggregator(x.reshape(-1, x.shape[-1]))
        logits = self.classifier(bag)
        return logits, F.softmax(logits, dim=-1), logits.argmax(dim=-1), attention

    @torch.no_grad()
    def attribute(self, x: torch.Tensor):
        """Inference-time explanation of a single bag.

        Args:
            x: raw patch features ``[1, N, in_dim]``.

        Returns:
            ``(attention, pah, region_mask)`` -- per-instance attention scores,
            per-instance Patch Activation Heat and the key region indices of the
            ``L x L`` grid.
        """
        x = self.dim_reduction(x)
        _, a_score = self.aggregator(x.reshape(-1, x.shape[-1]))
        pah = self.patch_activation_heat(x).squeeze(0)
        _, region_mask = self.select_key_regions(x)
        return a_score.squeeze(0), pah, region_mask
