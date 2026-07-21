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


def _resize_binary_target(mask, out_shape):
    if mask.ndim == 4:
        mask = mask.unsqueeze(1).float()
    else:
        mask = mask.float()

    if list(mask.shape[2:]) != list(out_shape):
        mask = F.interpolate(mask, size=out_shape, mode="nearest")
    return mask


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


def focal_tversky_loss_binary(logit, target, alpha=0.3, beta=0.7, gamma=0.75):
    prob = torch.sigmoid(logit)
    target = target.float()

    tp = (prob * target).sum()
    fp = (prob * (1 - target)).sum()
    fn = ((1 - prob) * target).sum()

    tversky = (tp + 1e-6) / (tp + alpha * fp + beta * fn + 1e-6)
    return (1 - tversky) ** gamma


class MSRSDLoss(nn.Module):
    """
    final seg + coarse seg + edge + teacher/student auxiliary + ET loss
    """
    def __init__(
        self,
        soft_dice_kwargs,
        ce_kwargs,
        aggregate="sum",
        coarse_weight=0.28,
        edge_weight=0.15,
        teacher_weight=0.08,
        student_weight=0.10,
        et_weight=0.40,
        delta_bg_weight=0.01
    ):
        super().__init__()
        self.main_loss = DC_and_CE_loss(
            soft_dice_kwargs=soft_dice_kwargs,
            ce_kwargs=ce_kwargs,
            aggregate=aggregate
        )
        self.coarse_weight = coarse_weight
        self.edge_weight = edge_weight
        self.teacher_weight = teacher_weight
        self.student_weight = student_weight
        self.et_weight = et_weight
        self.delta_bg_weight = delta_bg_weight

        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, outputs, target):
        target_raw = _unwrap_target(target)
        labels = _to_label(target)
        num_classes = outputs["seg"].shape[1]

        loss_final = self.main_loss(outputs["seg"], target_raw)

        coarse_target = _resize_target_for_multiclass_loss(
            target_raw, outputs["coarse"].shape[2:]
        )
        loss_coarse = self.main_loss(outputs["coarse"], coarse_target)

        labels_oh = one_hot(labels, num_classes=num_classes)

        if num_classes > 3:
            wt_mask = (labels_oh[:, 1:2] + labels_oh[:, 2:3] + labels_oh[:, 3:4]).clamp(0, 1)
            tc_mask = (labels_oh[:, 1:2] + labels_oh[:, 3:4]).clamp(0, 1)
            et_mask = labels_oh[:, 3:4]
        else:
            wt_mask = labels_oh[:, 1:].sum(dim=1, keepdim=True).clamp(0, 1)
            tc_mask = wt_mask
            et_mask = labels_oh[:, -1:]

        edge_target = torch.maximum(
            binary_boundary_from_mask(tc_mask),
            binary_boundary_from_mask(et_mask)
        )

        loss_edge = self.bce(outputs["edge"], edge_target)

        teacher_target = _resize_binary_target(wt_mask, outputs["teacher_map"].shape[2:])
        loss_teacher = self.bce(outputs["teacher_map"], teacher_target)

        student_band = F.max_pool3d(edge_target, kernel_size=3, stride=1, padding=1)
        student_target = _resize_binary_target(student_band, outputs["student_map"].shape[2:])
        loss_student = self.bce(outputs["student_map"], student_target)

        if num_classes > 3:
            et_logit = outputs["seg"][:, 3:4]
        else:
            et_logit = outputs["seg"][:, -1:]

        loss_et = focal_tversky_loss_binary(et_logit, et_mask)

        bg_mask = (labels == 0).float().unsqueeze(1)
        fg_delta = outputs["delta"][:, 1:] if outputs["delta"].shape[1] > 1 else outputs["delta"]
        loss_delta_bg = (fg_delta.abs() * bg_mask).mean()

        total = (
            loss_final
            + self.coarse_weight * loss_coarse
            + self.edge_weight * loss_edge
            + self.teacher_weight * loss_teacher
            + self.student_weight * loss_student
            + self.et_weight * loss_et
            + self.delta_bg_weight * loss_delta_bg
        )
        return total