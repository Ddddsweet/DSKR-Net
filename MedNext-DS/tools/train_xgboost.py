#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
在从 MedNeXt 抽取出的 voxel 级特征上训练 XGBoost 精修器。

使用方式（示例）：
    python tools/train_xgboost.py \
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
from xgboost import XGBClassifier


# -------------------------------------------------------------------------
# 工具函数
# -------------------------------------------------------------------------
def get_env(name, default=None, must_exist=False):
    v = os.environ.get(name, default)
    if must_exist and v is None:
        raise RuntimeError(f"环境变量 {name} 未设置且未提供默认值")
    return v


def load_all_features(folder: str):
    """
    读取 folder 下所有 case_*.npz，拼成一个大矩阵：
        feat: (N, C)
        label: (N,)
    """
    pattern = os.path.join(folder, "case_*.npz")
    files = sorted(glob.glob(pattern))

    print(f"特征目录: {folder}")
    print(f"找到特征文件数: {len(files)}")

    if not files:
        raise RuntimeError(
            f"在 {folder} 中没有找到 case_*.npz，说明特征还没抽取成功"
        )

    feats = []
    labels = []
    for f in files:
        arr = np.load(f)
        if "feat" not in arr or "label" not in arr:
            raise RuntimeError(f"{f} 中缺少 'feat' 或 'label' 键")

        X = arr["feat"]   # (N, C)
        y = arr["label"]  # (N,)

        if X.ndim != 2:
            raise RuntimeError(f"{f} 中 feat 维度为 {X.shape}，期望 (N, C)")
        if y.ndim != 1:
            raise RuntimeError(f"{f} 中 label 维度为 {y.shape}，期望 (N,)")

        if X.shape[0] != y.shape[0]:
            raise RuntimeError(
                f"{f} 中 feat 和 label 第 0 维不一致: {X.shape[0]} vs {y.shape[0]}"
            )

        feats.append(X)
        labels.append(y)

    X_all = np.concatenate(feats, axis=0)
    y_all = np.concatenate(labels, axis=0)

    print(f"合并后特征形状: {X_all.shape}, 标签形状: {y_all.shape}")
    return X_all, y_all


# -------------------------------------------------------------------------
# 参数
# -------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="在 MedNeXt 抽取的 voxel 特征上训练 XGBoost 精修器"
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
    p.add_argument("--max_depth", type=int, default=6,
                   help="XGBoost 最大树深度")
    p.add_argument("--n_estimators", type=int, default=200,
                   help="XGBoost 树的数量")
    p.add_argument("--learning_rate", type=float, default=0.1,
                   help="XGBoost 学习率")
    p.add_argument("--subsample", type=float, default=0.8,
                   help="XGBoost subsample")
    p.add_argument("--colsample_bytree", type=float, default=0.8,
                   help="XGBoost colsample_bytree")

    # 总采样上限：防止 30 亿 voxel 直接打爆内存
    p.add_argument("--max_samples", type=int, default=10000000,
                   help="训练时最多使用多少 voxel，总量超过会做下采样 + 类别均衡")

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

    result_dir = os.path.join(
        results_base, "nnUNet", model, task, f"{trainer}__{plans}"
    )
    feat_dir = os.path.join(result_dir, "all", "xgb_features")

    # 1. 读入所有特征
    X, y = load_all_features(feat_dir)

    # 2. 过滤掉 ignore label（例如 -1）
    mask = y >= 0
    num_ignored = int((~mask).sum())
    if num_ignored > 0:
        print(f"过滤掉 y < 0 的 voxel 数量: {num_ignored}")
        X = X[mask]
        y = y[mask]

    print(f"过滤 ignore 后特征形状: X={X.shape}, y={y.shape}")

    # 3. 类别统计
    classes = np.unique(y)
    print(f"过滤后唯一标签: {classes}")
    if classes.min() < 0:
        raise RuntimeError(
            f"过滤后仍然存在负标签: {classes}，这不应该发生，请检查数据。"
        )
    num_classes = int(len(classes))
    print(f"类别数: {num_classes}")

    # 4. 做「类别均衡」下采样，而不是对全部样本做全局随机采样
    N = X.shape[0]
    if N > args.max_samples:
        print(
            f"当前样本数 {N} > max_samples={args.max_samples}，"
            f"将按类别均衡方式下采样..."
        )
        max_per_class = args.max_samples // num_classes
        all_idx = []

        for c in classes:
            idx_c = np.where(y == c)[0]
            n_c = min(len(idx_c), max_per_class)
            if n_c > 0:
                sel = np.random.choice(idx_c, size=n_c, replace=False)
                all_idx.append(sel)
            print(f"  类别 {c}: 总数={len(idx_c)}, 采样={n_c}")

        if not all_idx:
            raise RuntimeError("均衡采样后没有任何样本，请检查标签分布。")

        all_idx = np.concatenate(all_idx)
        np.random.shuffle(all_idx)

        X = X[all_idx]
        y = y[all_idx]
        print(f"均衡下采样后形状: X={X.shape}, y={y.shape}")

        # 重新看看采样后的类别分布
        uniq, cnt = np.unique(y, return_counts=True)
        print("均衡采样后各类别 voxel 数：")
        for uc, cnum in zip(uniq, cnt):
            print(f"  类别 {uc}: {cnum}")

    # 5. 训练 XGBoost
    print("开始训练 XGBoost 模型...")

    clf = XGBClassifier(
        max_depth=args.max_depth,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        objective="multi:softmax",
        num_class=num_classes,
        tree_method="hist",
        n_jobs=8,
        eval_metric="mlogloss",
    )

    clf.fit(X, y)
    print("XGBoost 模型训练完成。")

    # 6. 保存模型
    save_dir = os.path.join(result_dir, "all", "xgb_model")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "xgb_refiner.pkl")

    with open(save_path, "wb") as f:
        pickle.dump(clf, f)

    print(f"XGBoost 模型已保存到: {save_path}")


if __name__ == "__main__":
    main()
