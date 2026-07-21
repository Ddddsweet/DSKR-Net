#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从训练好的 MedNeXt 模型中抽取 voxel 级特征（通常是 softmax 概率），
并与对应的标签一起保存为 npz，用于后续训练 XGBoost 精修器。

使用方式（示例）：
    python tools/extract_xgb_features.py \
        --task Task021_BrainTumour \
        --model 3d_fullres \
        --trainer nnUNetTrainerV2_MedNeXt_S_kernel3 \
        --plans nnUNetPlansv2.1_trgSp_1x1x1
"""

import os
import glob
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from nnunet_mednext.network_architecture.mednextv1.MedNextV1 import MedNeXt


# -------------------------------------------------------------------------
# 工具函数
# -------------------------------------------------------------------------
def get_env(name, default=None, must_exist=False):
    v = os.environ.get(name, default)
    if must_exist and v is None:
        raise RuntimeError(f"环境变量 {name} 未设置且未提供默认值")
    return v


def find_stage0_folder(task_preprocessed_dir: str, prefer_3d: bool = True) -> str:
    """
    在 nnUNet_preprocessed/TaskXXX 下找到包含 'stage0' 的子目录
    - prefer_3d=True 时优先选择不含 '2D' 的目录（给 3d_fullres 用）
    - 否则优先选择含 '2D' 的目录
    """
    candidates = [
        d for d in glob.glob(os.path.join(task_preprocessed_dir, "*"))
        if os.path.isdir(d) and "stage0" in os.path.basename(d)
    ]
    if not candidates:
        raise RuntimeError(
            f"在 {task_preprocessed_dir} 下找不到包含 'stage0' 的子目录，"
            f"请确认 nnUNet_preprocessed 结构是否正确"
        )

    if prefer_3d:
        c3d = [d for d in candidates if "2D" not in os.path.basename(d)]
        if c3d:
            return sorted(c3d)[0]
    else:
        c2d = [d for d in candidates if "2D" in os.path.basename(d)]
        if c2d:
            return sorted(c2d)[0]

    # 实在没得选就随便取一个
    return sorted(candidates)[0]


def pad_to_multiple_of_factor(tensor, factor=16):
    """
    将 (B, C, D, H, W) pad 到各向都为 factor 的倍数。
    返回: padded_tensor, pad_info
        pad_info: (pd0, pd1, ph0, ph1, pw0, pw1)
    """
    assert tensor.ndim == 5
    B, C, D, H, W = tensor.shape

    def _pad(dim):
        pad = (factor - dim % factor) % factor
        return pad // 2, pad - pad // 2

    pd0, pd1 = _pad(D)
    ph0, ph1 = _pad(H)
    pw0, pw1 = _pad(W)

    padding = (pw0, pw1, ph0, ph1, pd0, pd1)  # F.pad 的顺序是 (W_left, W_right, H_left, H_right, D_left, D_right)
    padded = F.pad(tensor, padding, mode="constant", value=0.0)
    return padded, (pd0, pd1, ph0, ph1, pw0, pw1)


def crop_back_to_original(tensor, pad_info, orig_shape):
    """
    将 (B, C, D_pad, H_pad, W_pad) 按 pad_info crop 回原始 DHW。
    orig_shape: (D, H, W)
    """
    B, C, Dp, Hp, Wp = tensor.shape
    pd0, pd1, ph0, ph1, pw0, pw1 = pad_info
    D, H, W = orig_shape

    d_start = pd0
    h_start = ph0
    w_start = pw0
    d_end = d_start + D
    h_end = h_start + H
    w_end = w_start + W

    cropped = tensor[:, :, d_start:d_end, h_start:h_end, w_start:w_end]
    assert cropped.shape[2:] == orig_shape, (
        f"crop 后尺寸 {cropped.shape[2:]} 与期望 {orig_shape} 不一致"
    )
    return cropped


# -------------------------------------------------------------------------
# 与 checkpoint 结构对齐的 MedNeXt
# -------------------------------------------------------------------------
def build_network(ckpt_file: str):
    """
    构建与训练时一致的 MedNeXt 结构，并加载权重
    （⚠️ 你的 MedNeXt 源码要尽量和当时训练这个 ckpt 的版本一致）
    """
    net = MedNeXt(
        in_channels=4,
        n_channels=32,
        n_classes=4,             # 背景 + 3 类
        exp_r=2,
        kernel_size=3,
        enc_kernel_size=None,
        dec_kernel_size=None,
        deep_supervision=True,
        do_res=True,
        do_res_up_down=True,
        checkpoint_style=None,
        block_counts=[2, 2, 2, 2, 2, 2, 2, 2, 2],
        norm_type="group",
        dim="3d",
        grn=False,
        use_fdkan_in_down=False,
        use_msca_in_up=False,
        fdkan_groups=4,
        fdkan_Q=6,
        fdkan_K=8,
        drop_path_max=0.1,
    )

    assert os.path.isfile(ckpt_file), f"找不到 checkpoint: {ckpt_file}"
    ckpt = torch.load(ckpt_file, map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    missing, unexpected = net.load_state_dict(state_dict, strict=False)
    print(f"load_state_dict missing keys: {len(missing)}")
    if missing:
        print("  MISSING:", *missing[:20], sep="\n  ")
    print(f"load_state_dict unexpected keys: {len(unexpected)}")
    if unexpected:
        print("  UNEXPECTED:", *unexpected[:20], sep="\n  ")

    net.cuda()
    net.eval()
    return net


# -------------------------------------------------------------------------
# 数据集
# -------------------------------------------------------------------------
class Stage0NPZDataset(Dataset):
    """
    读取 nnUNet_preprocessed/TaskXXX/<stage0> 下的 .npz：
        arr['data']: (C, D, H, W)，最后一通道为标签
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.files = sorted(
            f for f in glob.glob(os.path.join(data_dir, "*.npz"))
            if os.path.basename(f) != "dataset.json"
        )
        if not self.files:
            raise RuntimeError(
                f"在 {data_dir} 中没有找到任何 .npz 文件，请检查预处理是否完成"
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        case_id = os.path.splitext(os.path.basename(path))[0]

        arr = np.load(path)
        if "data" not in arr:
            raise RuntimeError(f"{path} 中不存在 key='data'")

        data = arr["data"]  # (C, D, H, W)  或者 (C, H, W) 看你的预处理
        if data.ndim == 3:
            # 万一是 (C, H, W)，我们加一个 D=1 维度，变为 (C, 1, H, W)
            data = data[:, None, :, :]

        if data.ndim != 4:
            raise RuntimeError(
                f"{path} data 维度为 {data.shape}，期望 (C, D, H, W)"
            )

        img = data[:-1].astype(np.float32)   # (C_img, D, H, W)
        seg = data[-1].astype(np.int16)      # (D, H, W)

        return torch.from_numpy(img), torch.from_numpy(seg), case_id


# -------------------------------------------------------------------------
# 主流程
# -------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="从 MedNeXt 抽取 voxel 级特征，用于训练 XGBoost 精修器"
    )
    p.add_argument("--task", type=str, required=True,
                   help="任务 ID，例如 Task021_BrainTumour")
    p.add_argument("--model", type=str, default="3d_fullres",
                   help="nnUNet 模式（默认 3d_fullres）")
    p.add_argument("--trainer", type=str,
                   default="nnUNetTrainerV2_MedNeXt_S_kernel3",
                   help="trainer 名称")
    p.add_argument("--plans", type=str,
                   default="nnUNetPlansv2.1_trgSp_1x1x1",
                   help="plans 名称")
    p.add_argument("--batch_size", type=int, default=1,
                   help="一般 1 即可，按体积大小酌情调整")
    return p.parse_args()


def main():
    args = parse_args()

    task = args.task
    model = args.model
    trainer = args.trainer
    plans = args.plans

    results_base = get_env("RESULTS_FOLDER", "./nnUNet_results")
    preprocessed_base = get_env("nnUNet_preprocessed", must_exist=True)

    result_dir = os.path.join(
        results_base, "nnUNet", model, task, f"{trainer}__{plans}"
    )
    ckpt_file = os.path.join(result_dir, "all", "model_final_checkpoint.model")
    task_preprocessed_dir = os.path.join(preprocessed_base, task)

    print(f"使用模型权重: {ckpt_file}")
    print(f"数据集目录: {task_preprocessed_dir}")

    prefer_3d = "3d" in model.lower()
    stage0_dir = find_stage0_folder(task_preprocessed_dir, prefer_3d=prefer_3d)
    print(f"使用 stage0 数据目录: {stage0_dir}")

    # 特征输出目录
    feat_out_dir = os.path.join(result_dir, "all", "xgb_features")
    os.makedirs(feat_out_dir, exist_ok=True)
    print(f"特征输出目录: {feat_out_dir}")

    # 构建网络与数据集
    net = build_network(ckpt_file)
    dataset = Stage0NPZDataset(stage0_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    net.eval()
    with torch.no_grad():
        for img, seg, case_id in loader:
            # img: (B=1, C, D, H, W)
            img = img.cuda(non_blocking=True)
            seg = seg.cuda(non_blocking=True)[0]  # (D, H, W)
            orig_shape = tuple(seg.shape)         # (D, H, W)

            # ------------------- padding 到 16 的倍数 -------------------
            img_padded, pad_info = pad_to_multiple_of_factor(img, factor=16)

            # ------------------- 网络前向 -------------------
            out = net(img_padded)
            if isinstance(out, (list, tuple)):
                logits = out[0]  # (1, C_out, Dp, Hp, Wp)
            else:
                logits = out

            probs = torch.softmax(logits, dim=1)  # (1, C_out, Dp, Hp, Wp)

            # ------------------- crop 回原始尺寸 -------------------
            probs = crop_back_to_original(probs, pad_info, orig_shape)
            probs = probs[0].cpu().numpy()  # (C_out, D, H, W)

            # ------------------- 展平为 (N, C) & 标签 (N,) -------------------
            C, D, H, W = probs.shape
            feat = probs.reshape(C, -1).transpose(1, 0).astype(np.float32)  # (N, C)
            label = seg.cpu().numpy().reshape(-1).astype(np.int16)          # (N,)

            assert feat.shape[0] == label.shape[0], \
                f"feat N={feat.shape[0]} 与 label N={label.shape[0]} 不一致"

            save_path = os.path.join(feat_out_dir, f"case_{case_id}.npz")
            np.savez_compressed(save_path, feat=feat, label=label)
            print(
                f"保存特征: {save_path}, "
                f"feat 形状={feat.shape}, label 形状={label.shape}"
            )

    print("全部 case 特征抽取完成。")


if __name__ == "__main__":
    main()
