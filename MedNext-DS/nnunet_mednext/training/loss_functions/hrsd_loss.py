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


def labels_to_regions(labels):
    wt = (labels > 0).float().unsqueeze(1)
    tc = ((labels == 1) | (labels == 3)).float().unsqueeze(1)
    et = (labels == 3).float().unsqueeze(1)
    return wt, tc, et


def boundary_from_mask(mask):
    gx = torch.abs(mask[:, :, 1:] - mask[:, :, :-1])
    gy = torch.abs(mask[:, :, :, 1:] - mask[:, :, :, :-1])
    gz = torch.abs(mask[:, :, :, :, 1:] - mask[:, :, :, :, :-1])

    bx = F.pad(gx, (0, 0, 0, 0, 1, 0))
    by = F.pad(gy, (0, 0, 1, 0, 0, 0))
    bz = F.pad(gz, (1, 0, 0, 0, 0, 0))
    bd = (bx + by + bz).clamp(0, 1)
    return bd


def binary_dice_loss_with_logits(logits, target, smooth=1e-5):
    prob = torch.sigmoid(logits)
    dims = tuple(range(2, prob.ndim))
    inter = (prob * target).sum(dim=dims)
    den = prob.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * inter + smooth) / (den + smooth)
    return 1.0 - dice.mean()


class HRSDLoss(nn.Module):
    def __init__(
        self,
        soft_dice_kwargs,
        ce_kwargs,
        aggregate="sum",
        coarse_seg_weight=0.35,
        coarse_region_weight=0.15,
        wt_weight=0.08,
        tc_weight=0.12,
        et_weight=0.18,
        band_weight=0.06,
        contain_weight=0.04,
    ):
        super().__init__()
        self.seg_loss = DC_and_CE_loss(
            soft_dice_kwargs=soft_dice_kwargs,
            ce_kwargs=ce_kwargs,
            aggregate=aggregate
        )

        self.coarse_seg_weight = coarse_seg_weight
        self.coarse_region_weight = coarse_region_weight
        self.wt_weight = wt_weight
        self.tc_weight = tc_weight
        self.et_weight = et_weight
        self.band_weight = band_weight
        self.contain_weight = contain_weight

        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, outputs, target):
        target_raw = _unwrap_target(target)
        labels = _to_label(target)
        wt_t, tc_t, et_t = labels_to_regions(labels)

        # final segmentation must remain dominant
        loss_seg = self.seg_loss(outputs["seg"], target_raw)

        # baseline preserving coarse seg supervision
        loss_coarse_seg = self.seg_loss(outputs["coarse_seg"], target_raw)

        # coarse region prior supervision
        coarse_target = torch.cat([wt_t, tc_t, et_t], dim=1)
        coarse_target = F.interpolate(coarse_target, size=outputs["coarse"].shape[2:], mode="nearest")
        loss_coarse_region = self.bce(outputs["coarse"], coarse_target)

        # hierarchical branch supervision
        loss_wt = self.bce(outputs["wt"], wt_t) + binary_dice_loss_with_logits(outputs["wt"], wt_t)
        loss_tc = self.bce(outputs["tc"], tc_t) + binary_dice_loss_with_logits(outputs["tc"], tc_t)
        loss_et = self.bce(outputs["et"], et_t) + binary_dice_loss_with_logits(outputs["et"], et_t)

        band_t = torch.maximum(boundary_from_mask(tc_t), boundary_from_mask(et_t))
        loss_band = self.bce(outputs["band"], band_t)

        wt_p = torch.sigmoid(outputs["wt"])
        tc_p = torch.sigmoid(outputs["tc"])
        et_p = torch.sigmoid(outputs["et"])
        loss_contain = F.relu(tc_p - wt_p).mean() + F.relu(et_p - tc_p).mean()

        total = (
            loss_seg
            + self.coarse_seg_weight * loss_coarse_seg
            + self.coarse_region_weight * loss_coarse_region
            + self.wt_weight * loss_wt
            + self.tc_weight * loss_tc
            + self.et_weight * loss_et
            + self.band_weight * loss_band
            + self.contain_weight * loss_contain
        )
        return total