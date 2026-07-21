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


def binary_boundary_from_mask(mask):
    gx = torch.abs(mask[:, :, 1:] - mask[:, :, :-1])
    gy = torch.abs(mask[:, :, :, 1:] - mask[:, :, :, :-1])
    gz = torch.abs(mask[:, :, :, :, 1:] - mask[:, :, :, :, :-1])

    bx = F.pad(gx, (0, 0, 0, 0, 1, 0))
    by = F.pad(gy, (0, 0, 1, 0, 0, 0))
    bz = F.pad(gz, (1, 0, 0, 0, 0, 0))
    return (bx + by + bz).clamp(0, 1)


class TopoStateLoss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum",
                 topo_weight=0.2, boundary_weight=0.2):
        super().__init__()
        self.seg_loss = DC_and_CE_loss(
            soft_dice_kwargs=soft_dice_kwargs,
            ce_kwargs=ce_kwargs,
            aggregate=aggregate
        )
        self.topo_weight = topo_weight
        self.boundary_weight = boundary_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, outputs, target):
        target_raw = _unwrap_target(target)
        labels = _to_label(target)

        loss_seg = self.seg_loss(outputs["seg"], target_raw)

        tumor_mask = (labels > 0).float().unsqueeze(1)
        tumor_boundary = binary_boundary_from_mask(tumor_mask)

        topo_target = F.interpolate(
            tumor_mask, size=outputs["topology"].shape[2:], mode="trilinear", align_corners=False
        )

        loss_topo = self.bce(outputs["topology"], topo_target)
        loss_boundary = self.bce(outputs["boundary"], tumor_boundary)

        return loss_seg + self.topo_weight * loss_topo + self.boundary_weight * loss_boundary