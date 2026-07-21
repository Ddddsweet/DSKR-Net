import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import binary_erosion, distance_transform_edt

from nnunet_mednext.training.loss_functions.dice_loss import DC_and_CE_loss


def labels_to_regions(labels: torch.Tensor):
    """
    labels: (B, D, H, W)
    returns: (B, 3, D, H, W) for TC / WT / ET
    """
    tc = ((labels == 1) | (labels == 3)).float()
    wt = ((labels == 1) | (labels == 2) | (labels == 3)).float()
    et = (labels == 3).float()
    return torch.stack([tc, wt, et], dim=1)


def _single_boundary(mask: np.ndarray) -> np.ndarray:
    if mask.max() == 0:
        return np.zeros_like(mask, dtype=np.float32)
    eroded = binary_erosion(mask, structure=np.ones((3, 3, 3)), iterations=1, border_value=0)
    boundary = mask.astype(np.float32) - eroded.astype(np.float32)
    boundary[boundary < 0] = 0
    return boundary.astype(np.float32)


def build_boundary_targets(labels: torch.Tensor, ignore_label: int = -1):
    """
    labels: (B, D, H, W)
    returns: (B, 1, D, H, W)
    """
    labels_np = labels.detach().cpu().numpy()
    out = []
    for b in range(labels_np.shape[0]):
        lb = labels_np[b]
        valid = lb != ignore_label
        tumor = (lb > 0) & valid
        bd = _single_boundary(tumor.astype(np.uint8))
        out.append(torch.from_numpy(bd[None]))
    out = torch.stack(out, dim=0).float().to(labels.device)
    return out


def _single_sdf(mask: np.ndarray) -> np.ndarray:
    if mask.max() == 0:
        return np.zeros_like(mask, dtype=np.float32)

    posmask = mask.astype(bool)
    negmask = ~posmask

    posdis = distance_transform_edt(posmask).astype(np.float32)
    negdis = distance_transform_edt(negmask).astype(np.float32)
    sdf = posdis - negdis

    mx = np.max(np.abs(sdf))
    if mx > 0:
        sdf = sdf / mx
    return sdf.astype(np.float32)


def build_sdf_targets(labels: torch.Tensor, ignore_label: int = -1):
    """
    labels: (B, D, H, W)
    returns: (B, 3, D, H, W) for TC / WT / ET
    """
    regions = labels_to_regions(labels).detach().cpu().numpy()
    out = []
    for b in range(regions.shape[0]):
        sdf_b = []
        for c in range(regions.shape[1]):
            sdf = _single_sdf(regions[b, c] > 0.5)
            sdf_b.append(torch.from_numpy(sdf))
        sdf_b = torch.stack(sdf_b, dim=0)
        out.append(sdf_b)
    out = torch.stack(out, dim=0).float().to(labels.device)
    return out


class GeoStateLoss(nn.Module):
    def __init__(
        self,
        soft_dice_kwargs,
        ce_kwargs,
        aggregate="sum",
        boundary_weight: float = 0.2,
        sdf_weight: float = 0.3,
        ignore_label: int = -1,
        seg_only: bool = True,
    ):
        super().__init__()
        self.seg_loss = DC_and_CE_loss(
            soft_dice_kwargs=soft_dice_kwargs,
            ce_kwargs=ce_kwargs,
            aggregate=aggregate
        )
        self.boundary_weight = boundary_weight
        self.sdf_weight = sdf_weight
        self.ignore_label = ignore_label
        self.seg_only = seg_only

        self.bce = nn.BCEWithLogitsLoss()

    def _unwrap_target(self, target):
        """
        nnUNet/MedNeXt 里 target 可能是:
        1) tensor
        2) [tensor]                  # deep supervision关闭时也可能这样
        3) [tensor, tensor, ...]     # deep supervision开启
        这里统一取最高分辨率那个。
        """
        if isinstance(target, (list, tuple)):
            if len(target) == 0:
                raise ValueError("target is an empty list/tuple")
            target = target[0]
        return target

    def forward(self, pred_dict, target):
        """
        pred_dict:
            pred_dict["seg"]:      (B, C, D, H, W)
            pred_dict["boundary"]: (B, 1, D, H, W)
            pred_dict["sdf"]:      (B, 3, D, H, W)

        target:
            may be:
              - tensor (B, 1, D, H, W)
              - tensor (B, D, H, W)
              - list/tuple containing the above
        """
        target = self._unwrap_target(target)

        if not torch.is_tensor(target):
            raise TypeError(f"target must be tensor after unwrap, got {type(target)}")

        # 给 DC_and_CE_loss 用的原始 target 形式
        target_for_seg_loss = target

        if target.ndim == 5:
            target_seg = target[:, 0].long()
        elif target.ndim == 4:
            target_seg = target.long()
        else:
            raise ValueError(f"unexpected target ndim: {target.ndim}, target shape: {target.shape}")

        seg_logits = pred_dict["seg"]
        seg_loss = self.seg_loss(seg_logits, target_for_seg_loss)

        total = seg_loss
        loss_boundary = torch.tensor(0.0, device=seg_logits.device)
        loss_sdf = torch.tensor(0.0, device=seg_logits.device)

        if not self.seg_only:
            boundary_target = build_boundary_targets(target_seg, ignore_label=self.ignore_label)
            sdf_target = build_sdf_targets(target_seg, ignore_label=self.ignore_label)

            boundary_logits = pred_dict["boundary"]
            sdf_logits = pred_dict["sdf"]

            loss_boundary = self.bce(boundary_logits, boundary_target)
            loss_sdf = F.smooth_l1_loss(sdf_logits, sdf_target)

            total = total + self.boundary_weight * loss_boundary + self.sdf_weight * loss_sdf

        return total