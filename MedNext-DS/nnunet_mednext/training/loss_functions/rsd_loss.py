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


def one_hot(labels, num_classes):
    return F.one_hot(labels, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()


def binary_boundary_from_mask(mask):
    # cheap approximation for training-time boundary cue
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


class RSDLoss(nn.Module):
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

        loss_final = self.main_loss(outputs["seg"], target_raw)
        loss_coarse = self.main_loss(outputs["coarse"], target_raw)

        # ET-specific geometry-oriented terms
        labels_oh = one_hot(labels, num_classes=num_classes)
        et_mask = labels_oh[:, 3:4] if num_classes > 3 else labels_oh[:, -1:]

        et_boundary = binary_boundary_from_mask(et_mask)
        edge_loss = self.bce(outputs["edge"], et_boundary)

        et_logit = outputs["seg"][:, 3:4] if num_classes > 3 else outputs["seg"][:, -1:]
        et_ft = focal_tversky_loss_binary(et_logit, et_mask)

        total = loss_final \
                + self.coarse_weight * loss_coarse \
                + self.edge_weight * edge_loss \
                + self.et_weight * et_ft

        return total