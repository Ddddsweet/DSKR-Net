import torch
import torch.nn as nn
import torch.nn.functional as F

from nnunet_mednext.training.loss_functions.dice_loss import DC_and_CE_loss


def _unwrap_target(target):
    if isinstance(target, (list, tuple)):
        target = target[0]
    return target


def _to_label(target):
    target = _unwrap_target(target)
    if target.ndim == 5:
        target = target[:, 0]
    return target.long()


def _resize_target_for_multiclass_loss(target_raw, out_shape):
    if target_raw.ndim == 4:
        target_raw = target_raw.unsqueeze(1).float()
    else:
        target_raw = target_raw.float()

    if list(target_raw.shape[2:]) != list(out_shape):
        target_raw = F.interpolate(target_raw, size=out_shape, mode="nearest")

    return target_raw.long()


def one_hot(labels, num_classes):
    return F.one_hot(labels, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()


def binary_boundary_from_mask(mask):
    gx = torch.abs(mask[:, :, 1:] - mask[:, :, :-1])
    gy = torch.abs(mask[:, :, :, 1:] - mask[:, :, :, :-1])
    gz = torch.abs(mask[:, :, :, :, 1:] - mask[:, :, :, :, :-1])

    bx = F.pad(gx, (0, 0, 0, 0, 1, 0))
    by = F.pad(gy, (0, 0, 1, 0, 0, 0))
    bz = F.pad(gz, (1, 0, 0, 0, 0, 0))
    return (bx + by + bz).clamp(0, 1)


def region_probs_from_seg(seg_logits):
    prob = torch.softmax(seg_logits, dim=1)
    if seg_logits.shape[1] >= 4:
        wt = prob[:, 1:2] + prob[:, 2:3] + prob[:, 3:4]
        tc = prob[:, 1:2] + prob[:, 3:4]
        et = prob[:, 3:4]
    else:
        wt = prob[:, 1:].sum(dim=1, keepdim=True)
        tc = prob[:, 1:].sum(dim=1, keepdim=True)
        et = prob[:, -1:]
    return wt, tc, et


def soft_dice_region_loss(prob, target, smooth=1e-5):
    dims = tuple(range(2, prob.ndim))
    inter = (prob * target).sum(dim=dims)
    den = prob.sum(dim=dims) + target.sum(dim=dims)
    dice = (2 * inter + smooth) / (den + smooth)
    return 1.0 - dice.mean()


def focal_tversky_loss_binary_from_prob(prob, target, alpha=0.3, beta=0.7, gamma=0.75):
    target = target.float()
    tp = (prob * target).sum()
    fp = (prob * (1 - target)).sum()
    fn = ((1 - prob) * target).sum()

    tversky = (tp + 1e-6) / (tp + alpha * fp + beta * fn + 1e-6)
    return (1 - tversky) ** gamma


class RSDLossV4(nn.Module):
    """
    兼顾 Dice 与 HD95 的 V4 loss
    """
    def __init__(
        self,
        soft_dice_kwargs,
        ce_kwargs,
        aggregate="sum",
        coarse_weight=0.24,
        edge_weight=0.14,
        band_weight=0.10,
        surface_weight=0.12,
        region_weight=0.20,
        et_weight=0.42,
        delta_bg_weight=0.015,
    ):
        super().__init__()
        self.main_loss = DC_and_CE_loss(
            soft_dice_kwargs=soft_dice_kwargs,
            ce_kwargs=ce_kwargs,
            aggregate=aggregate
        )

        self.coarse_weight = coarse_weight
        self.edge_weight = edge_weight
        self.band_weight = band_weight
        self.surface_weight = surface_weight
        self.region_weight = region_weight
        self.et_weight = et_weight
        self.delta_bg_weight = delta_bg_weight

        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, outputs, target):
        target_raw = _unwrap_target(target)
        labels = _to_label(target)
        num_classes = outputs["seg"].shape[1]

        # main loss
        loss_final = self.main_loss(outputs["seg"], target_raw)

        coarse_target_raw = _resize_target_for_multiclass_loss(
            target_raw, outputs["coarse"].shape[2:]
        )
        loss_coarse = self.main_loss(outputs["coarse"], coarse_target_raw)

        labels_oh = one_hot(labels, num_classes=num_classes)
        if num_classes > 3:
            wt_mask = (labels_oh[:, 1:2] + labels_oh[:, 2:3] + labels_oh[:, 3:4]).clamp(0, 1)
            tc_mask = (labels_oh[:, 1:2] + labels_oh[:, 3:4]).clamp(0, 1)
            et_mask = labels_oh[:, 3:4]
        else:
            wt_mask = labels_oh[:, 1:].sum(dim=1, keepdim=True).clamp(0, 1)
            tc_mask = wt_mask
            et_mask = labels_oh[:, -1:]

        # edge / band / surface targets
        wt_edge = binary_boundary_from_mask(wt_mask)
        tc_edge = binary_boundary_from_mask(tc_mask)
        et_edge = binary_boundary_from_mask(et_mask)

        edge_target = torch.maximum(tc_edge, et_edge)
        band_target = F.max_pool3d(edge_target, kernel_size=3, stride=1, padding=1)
        surface_target = torch.maximum(wt_edge, torch.maximum(tc_edge, et_edge))

        loss_edge = self.bce(outputs["edge"], edge_target)
        loss_band = self.bce(outputs["band"], band_target)
        loss_surface = self.bce(outputs["surface"], surface_target)

        # metric-aligned region supervision
        wt_prob, tc_prob, et_prob = region_probs_from_seg(outputs["seg"])
        loss_region = (
            soft_dice_region_loss(wt_prob, wt_mask) +
            soft_dice_region_loss(tc_prob, tc_mask) +
            soft_dice_region_loss(et_prob, et_mask)
        ) / 3.0

        loss_et = focal_tversky_loss_binary_from_prob(et_prob.clamp(1e-6, 1 - 1e-6), et_mask)

        # suppress foreground residuals in background
        bg_mask = (labels == 0).float().unsqueeze(1)
        fg_delta = outputs["delta"][:, 1:] if outputs["delta"].shape[1] > 1 else outputs["delta"]
        loss_delta_bg = (fg_delta.abs() * bg_mask).mean()

        total = (
            loss_final
            + self.coarse_weight * loss_coarse
            + self.edge_weight * loss_edge
            + self.band_weight * loss_band
            + self.surface_weight * loss_surface
            + self.region_weight * loss_region
            + self.et_weight * loss_et
            + self.delta_bg_weight * loss_delta_bg
        )
        return total