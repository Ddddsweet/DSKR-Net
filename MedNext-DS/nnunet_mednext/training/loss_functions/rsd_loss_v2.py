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
    """
    将原始 target 调整到指定空间尺寸，供 DC_and_CE_loss 使用
    target_raw:
        (B, 1, D, H, W) or (B, D, H, W)
    返回:
        与 net_output 空间大小一致的 target
    """
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
    bd = (bx + by + bz).clamp(0, 1)
    return bd


def focal_tversky_loss_binary(logit, target, alpha=0.3, beta=0.7, gamma=0.75):
    prob = torch.sigmoid(logit)
    target = target.float()

    tp = (prob * target).sum()
    fp = (prob * (1 - target)).sum()
    fn = ((1 - prob) * target).sum()

    tversky = (tp + 1e-6) / (tp + alpha * fp + beta * fn + 1e-6)
    return (1 - tversky) ** gamma


class RSDLossV2(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum",
                 coarse_weight=0.3, edge_weight=0.2, et_weight=0.4):
        super().__init__()
        self.main_loss = DC_and_CE_loss(
            soft_dice_kwargs=soft_dice_kwargs,
            ce_kwargs=ce_kwargs,
            aggregate=aggregate
        )
        self.coarse_weight = coarse_weight
        self.edge_weight = edge_weight
        self.et_weight = et_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, outputs, target):
        target_raw = _unwrap_target(target)
        labels = _to_label(target)
        num_classes = outputs["seg"].shape[1]

        # 1) final seg loss: 用原始分辨率 target
        loss_final = self.main_loss(outputs["seg"], target_raw)

        # 2) coarse loss: 先把 target 下采样到 coarse 分辨率
        coarse_target_raw = _resize_target_for_multiclass_loss(
            target_raw, outputs["coarse"].shape[2:]
        )
        loss_coarse = self.main_loss(outputs["coarse"], coarse_target_raw)

        # 3) ET / TC geometry supervision 仍在全分辨率做
        labels_oh = one_hot(labels, num_classes=num_classes)

        if num_classes > 3:
            et_mask = labels_oh[:, 3:4]
            tc_mask = labels_oh[:, 1:2] + labels_oh[:, 3:4]
        else:
            et_mask = labels_oh[:, -1:]
            tc_mask = labels_oh[:, -1:]

        tc_mask = tc_mask.clamp(0, 1)

        edge_target = torch.maximum(
            binary_boundary_from_mask(tc_mask),
            binary_boundary_from_mask(et_mask)
        )
        loss_edge = self.bce(outputs["edge"], edge_target)

        if num_classes > 3:
            et_logit = outputs["seg"][:, 3:4]
        else:
            et_logit = outputs["seg"][:, -1:]

        loss_et = focal_tversky_loss_binary(et_logit, et_mask)

        total = (
            loss_final
            + self.coarse_weight * loss_coarse
            + self.edge_weight * loss_edge
            + self.et_weight * loss_et
        )
        return total