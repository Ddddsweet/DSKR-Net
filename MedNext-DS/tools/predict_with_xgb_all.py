#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用 MedNeXt + 训练好的 XGBoost 精修器，对 Task021_BrainTumour 的预处理 npz 做预测，
并把 XGBoost 精修后的分割结果，映射回 labelsTr 同空间，保存为 .nii.gz 文件。

用法示例：

python tools/predict_with_xgb_all.py \
    --task Task021_BrainTumour \
    --model 3d_fullres \
    --trainer nnUNetTrainerV2_MedNeXt_S_kernel3 \
    --plans nnUNetPlansv2.1_trgSp_1x1x1
"""

import os
import glob
import argparse
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk

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

    # F.pad 的顺序是 (W_left, W_right, H_left, H_right, D_left, D_right)
    padding = (pw0, pw1, ph0, ph1, pd0, pd1)
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


def build_network(ckpt_file: str):
    """
    构建与训练时一致的 MedNeXt 结构，并加载权重
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

    # 兼容 PyTorch 2.6 的 weights_only 改动
    try:
        ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    except TypeError:
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
# 数据集（和 extract_xgb_features.py 保持一致）
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

        data = arr["data"]  # (C, D, H, W)  或者 (C, H, W)
        if data.ndim == 3:
            data = data[:, None, :, :]  # (C, 1, H, W)

        if data.ndim != 4:
            raise RuntimeError(
                f"{path} data 维度为 {data.shape}，期望 (C, D, H, W)"
            )

        img = data[:-1].astype(np.float32)   # (C_img, D, H, W)
        seg = data[-1].astype(np.int16)      # (D, H, W)

        return torch.from_numpy(img), torch.from_numpy(seg), case_id


# -------------------------------------------------------------------------
# 参数
# -------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="使用 MedNeXt + XGBoost 精修器预测，并映射回 labelsTr 空间"
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
                   help="一般 1 即可")
    p.add_argument("--chunk_size", type=int, default=200000,
                   help="XGBoost 预测时每次处理的 voxel 数，防止一次性太大")
    return p.parse_args()


# -------------------------------------------------------------------------
# 主流程
# -------------------------------------------------------------------------
def main():
    args = parse_args()

    task = args.task
    model = args.model
    trainer = args.trainer
    plans = args.plans

    results_base = get_env("RESULTS_FOLDER", "./nnUNet_results")
    preprocessed_base = get_env("nnUNet_preprocessed", must_exist=True)
    raw_base = get_env("nnUNet_raw_data_base", must_exist=True)

    # 结果、预处理、原始标签目录
    result_dir = os.path.join(
        results_base, "nnUNet", model, task, f"{trainer}__{plans}"
    )
    ckpt_file = os.path.join(result_dir, "all", "model_final_checkpoint.model")
    task_preprocessed_dir = os.path.join(preprocessed_base, task)
    label_dir = os.path.join(
        raw_base, "nnUNet_raw_data", task, "labelsTr"
    )

    print(f"使用模型权重: {ckpt_file}")
    print(f"预处理数据根目录: {task_preprocessed_dir}")
    print(f"标签目录 (labelsTr): {label_dir}")

    prefer_3d = "3d" in model.lower()
    stage0_dir = find_stage0_folder(task_preprocessed_dir, prefer_3d=prefer_3d)
    print(f"使用 stage0 数据目录: {stage0_dir}")

    # XGBoost 模型
    xgb_model_path = os.path.join(result_dir, "all", "xgb_model", "xgb_refiner.pkl")
    if not os.path.isfile(xgb_model_path):
        raise RuntimeError(f"找不到 XGBoost 模型 {xgb_model_path}")
    with open(xgb_model_path, "rb") as f:
        xgb_model = pickle.load(f)
    print(f"已加载 XGBoost 模型: {xgb_model_path}")

    # 输出目录：XGBoost 精修后的预测（已映射到 labelsTr 空间）
    out_dir = os.path.join(result_dir, "all", "xgb_preds")
    os.makedirs(out_dir, exist_ok=True)
    print(f"精修结果将保存到: {out_dir}")

    # MedNeXt 网络
    net = build_network(ckpt_file)

    # 数据集
    dataset = Stage0NPZDataset(stage0_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    net.eval()
    with torch.no_grad():
        for img, seg, case_id in loader:
            # ----------------- 处理 case_id 字符串 -----------------
            # 原来有类似 ('BraTS2021_01637',) 这种情况，这里统一清洗成 BraTS2021_01637
            if isinstance(case_id, (list, tuple)):
                case_str = str(case_id[0])
            else:
                case_str = str(case_id)
            case_str = case_str.strip("()' ")

            print(f"\n===== 处理 case: {case_str} =====")

            # 找对应的 labelsTr（用于最后映射回去）
            gt_path_gz = os.path.join(label_dir, f"{case_str}.nii.gz")
            gt_path_ni = os.path.join(label_dir, f"{case_str}.nii")
            if os.path.exists(gt_path_gz):
                gt_path = gt_path_gz
            elif os.path.exists(gt_path_ni):
                gt_path = gt_path_ni
            else:
                print(f"[WARN] 在 labelsTr 中找不到 {case_str}，跳过该 case。")
                continue

            gt_img = sitk.ReadImage(gt_path)
            gt_arr = sitk.GetArrayFromImage(gt_img).astype(np.int16)  # (Z, Y, X)

            # ----------------- 网络预测（在 stage0 空间）-----------------
            img = img.cuda(non_blocking=True)     # (1, C, D, H, W)
            seg = seg.cuda(non_blocking=True)[0]  # (D, H, W)
            orig_shape = tuple(seg.shape)

            # 1) pad 到 16 的倍数，避免 MedNeXt 解码阶段尺寸错误
            img_padded, pad_info = pad_to_multiple_of_factor(img, factor=16)

            # 2) 前向
            out = net(img_padded)
            if isinstance(out, (list, tuple)):
                logits = out[0]  # (1, C_out, Dp, Hp, Wp)
            else:
                logits = out

            probs = torch.softmax(logits, dim=1)   # (1, C_out, Dp, Hp, Wp)

            # 3) crop 回原始 stage0 尺寸
            probs = crop_back_to_original(probs, pad_info, orig_shape)
            probs = probs[0]  # (C_out, D, H, W)

            C, D, H, W = probs.shape
            N = D * H * W

            # 4) 展平成 (N, C) 喂给 XGBoost
            probs_np = probs.cpu().numpy().reshape(C, N).transpose(1, 0)  # (N, C)

            preds_list = []
            for start in range(0, N, args.chunk_size):
                end = min(N, start + args.chunk_size)
                X_chunk = probs_np[start:end].astype(np.float32)
                y_chunk = xgb_model.predict(X_chunk)   # (chunk_size,)
                preds_list.append(y_chunk)

            y_pred_all = np.concatenate(preds_list, axis=0)   # (N,)
            xgb_seg = y_pred_all.reshape(D, H, W).astype(np.int16)  # (D, H, W)

            # ----------------- 映射到 labelsTr 空间 -----------------
            # 目标：生成 full_pred，形状与 gt_arr 一致，几何信息与 gt_img 一致
            full_pred = np.zeros_like(gt_arr, dtype=np.int16)  # (Z, Y, X)

            # 找 labelsTr 中的肿瘤前景 bounding box（WT > 0）
            nz = np.where(gt_arr > 0)
            if len(nz[0]) == 0:
                # 这个 case 没有肿瘤：直接保持全 0 即可
                print("  [INFO] GT 中没有肿瘤，预测也保持全 0。")
            else:
                z0, z1 = nz[0].min(), nz[0].max() + 1
                y0, y1 = nz[1].min(), nz[1].max() + 1
                x0, x1 = nz[2].min(), nz[2].max() + 1

                Dz = z1 - z0
                Dy = y1 - y0
                Dx = x1 - x0

                # 将 xgb_seg 插值到和 GT 肿瘤 bbox 一样的尺寸
                xgb_seg_t = torch.from_numpy(xgb_seg[None, None].astype(np.float32))  # (1,1,D,H,W)
                xgb_resized = F.interpolate(
                    xgb_seg_t,
                    size=(Dz, Dy, Dx),
                    mode="nearest"
                )[0, 0].cpu().numpy().astype(np.int16)

                # 填回 full_pred 对应的肿瘤 bbox 区域
                full_pred[z0:z1, y0:y1, x0:x1] = xgb_resized

            # ----------------- 保存为 .nii.gz（几何信息拷贝 labelsTr） -----------------
            pred_img = sitk.GetImageFromArray(full_pred.astype(np.uint16))
            pred_img.CopyInformation(gt_img)

            out_path = os.path.join(out_dir, f"{case_str}_xgb.nii.gz")
            sitk.WriteImage(pred_img, out_path)
            print(f"✅ 精修结果已保存: {out_path}")

    print("\n🎉 全部 case 预测 + XGBoost 精修（映射到 labelsTr 空间）完成！")
    print(f"你现在可以在 matrics.py 里设置：")
    print(f"  PRED_DIR  = \"{out_dir}\"")
    print(f"  LABEL_DIR = \"{label_dir}\"")
    print("然后重新运行 matrics.py 计算 WT/TC/ET Dice。")


if __name__ == "__main__":
    main()
