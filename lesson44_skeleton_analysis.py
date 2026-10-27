
from pathlib import Path
import csv

import numpy as np
import torch
import matplotlib.pyplot as plt

from scipy import ndimage as ndi
from skimage.morphology import skeletonize

from models.resnet_unet import ResNetUNet
from datasets.crack_dataset import CrackDataset
from datasets.splits import get_train_val_pairs


# =====================================
# 1. 配置
# =====================================

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "lesson44"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CONFIGS = {
    "Baseline": {
        "path": ROOT / "outputs" / "lesson42"
                / "best_baseline.pth",
        "threshold": 0.73
    },
    "Contrast Aug": {
        "path": ROOT / "outputs" / "lesson42"
                / "best_contrast_aug.pth",
        "threshold": 0.84
    }
}

# 8邻域连通
CONNECTIVITY = np.ones((3, 3), dtype=bool)

# 判断GT骨架是否被识别时的容差
TOLERANCE = 2

# 预测骨架距离GT超过3像素，视为疑似额外路径
FAR_DISTANCE = 3

# 忽略长度不足5像素的零碎候选片段
MIN_COMPONENT = 5

WIDTH_LABELS = [
    "Thin <=3px",
    "Medium 3-6px",
    "Thick >6px"
]


# =====================================
# 2. 加载数据和模型
# =====================================

_, val_pairs = get_train_val_pairs(
    ROOT / "data", seed=42
)

dataset = CrackDataset(
    val_pairs,
    training=False,
    augment=False
)

models = {}

for name, cfg in CONFIGS.items():

    model = ResNetUNet(
        pretrained=False,
        freeze_encoder=True
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(
            cfg["path"],
            map_location=DEVICE,
            weights_only=True
        )
    )

    model.eval()
    models[name] = model

print("设备:", DEVICE)
print("验证集图像:", len(dataset))


# =====================================
# 3. 连通区域统计
# =====================================

def count_components(binary, min_pixels=5):

    labels, _ = ndi.label(
        binary,
        structure=CONNECTIVITY
    )

    sizes = np.bincount(
        labels.ravel()
    )[1:]

    return int(
        np.count_nonzero(sizes >= min_pixels)
    )


# =====================================
# 4. 统计容器
# =====================================

stats = {}

for name in CONFIGS:

    stats[name] = {
        "width_total": np.zeros(3, dtype=np.int64),
        "width_exact": np.zeros(3, dtype=np.int64),
        "width_tol": np.zeros(3, dtype=np.int64),
        "gt_skel_total": 0,
        "gt_skel_found": 0,
        "pred_skel_total": 0,
        "far_skel_total": 0,
        "missing_segments": 0,
        "far_fragments": 0
    }

records = []
csv_rows = []


# =====================================
# 5. 遍历验证集
# =====================================

with torch.inference_mode():

    for index in range(len(dataset)):

        image_t, mask_t = dataset[index]

        image = image_t.permute(
            1, 2, 0
        ).numpy()

        gt = mask_t[0].numpy() > 0.5

        # GT骨架
        gt_skel = skeletonize(gt)

        # 距离变换，估计局部Mask宽度
        distance = ndi.distance_transform_edt(gt)
        width_map = 2.0 * distance

        x = image_t.unsqueeze(0).to(DEVICE)

        sample = {
            "name": val_pairs[index][0].name,
            "image": image,
            "gt": gt,
            "gt_skel": gt_skel,
            "models": {}
        }

        # 两组模型使用同一张验证图
        for name, model in models.items():

            threshold = CONFIGS[name]["threshold"]

            prob = torch.sigmoid(
                model(x)
            )[0, 0].cpu().numpy()

            pred = prob > threshold

            # 预测骨架
            pred_skel = skeletonize(pred)

            # 允许2像素的8邻域空间偏差
            pred_dilated = ndi.binary_dilation(
                pred,
                structure=CONNECTIVITY,
                iterations=TOLERANCE
            )

            # GT骨架未被预测覆盖的区域
            missed = gt_skel & ~pred_dilated

            # 预测骨架远离GT的区域
            gt_dilated = ndi.binary_dilation(
                gt,
                structure=CONNECTIVITY,
                iterations=FAR_DISTANCE
            )

            far = pred_skel & ~gt_dilated

            # 统计不同宽度的GT骨架
            width_on_skel = width_map[gt_skel]

            width_masks = [
                width_on_skel <= 3,
                (width_on_skel > 3)
                & (width_on_skel <= 6),
                width_on_skel > 6
            ]

            exact = pred[gt_skel]
            tolerant = pred_dilated[gt_skel]

            s = stats[name]

            for k, selection in enumerate(width_masks):

                s["width_total"][k] += selection.sum()

                s["width_exact"][k] += (
                    exact & selection
                ).sum()

                s["width_tol"][k] += (
                    tolerant & selection
                ).sum()

            s["gt_skel_total"] += gt_skel.sum()

            s["gt_skel_found"] += (
                gt_skel & pred_dilated
            ).sum()

            s["pred_skel_total"] += pred_skel.sum()

            s["far_skel_total"] += far.sum()

            missing_segments = count_components(
                missed,
                MIN_COMPONENT
            )

            far_fragments = count_components(
                far,
                MIN_COMPONENT
            )

            s["missing_segments"] += missing_segments
            s["far_fragments"] += far_fragments

            skeleton_recall = (
                (gt_skel & pred_dilated).sum()
                / max(gt_skel.sum(), 1)
            )

            far_ratio = (
                far.sum()
                / max(pred_skel.sum(), 1)
            )

            sample["models"][name] = {
                "pred": pred,
                "pred_skel": pred_skel,
                "missed": missed,
                "far": far,
                "skeleton_recall": skeleton_recall,
                "far_ratio": far_ratio
            }

            csv_rows.append({
                "image": sample["name"],
                "model": name,
                "centerline_recall_2px": skeleton_recall,
                "far_skeleton_ratio": far_ratio,
                "missing_segments": missing_segments,
                "far_fragments": far_fragments,
                "gt_components": count_components(
                    gt_skel,
                    MIN_COMPONENT
                ),
                "pred_components": count_components(
                    pred_skel,
                    MIN_COMPONENT
                )
            })

        records.append(sample)


# =====================================
# 6. 输出整体统计
# =====================================

print("\n========== 骨架结构比较 ==========")

for name, s in stats.items():

    centerline_recall = (
        s["gt_skel_found"]
        / max(s["gt_skel_total"], 1)
    )

    far_ratio = (
        s["far_skel_total"]
        / max(s["pred_skel_total"], 1)
    )

    print(f"\n模型: {name}")

    print(
        f"GT骨架召回率(容差2px): "
        f"{centerline_recall:.4f}"
    )

    print(
        f"远离GT的预测骨架比例: "
        f"{far_ratio:.4f}"
    )

    print(
        f"漏检路径片段候选数: "
        f"{s['missing_segments']}"
    )

    print(
        f"疑似额外路径片段数: "
        f"{s['far_fragments']}"
    )

    print("\n不同Mask宽度的骨架召回率:")

    for k, label in enumerate(WIDTH_LABELS):

        total = s["width_total"][k]

        if total == 0:
            print(label, ": 无样本")
            continue

        exact = s["width_exact"][k] / total
        tolerant = s["width_tol"][k] / total

        print(
            f"{label:15s} | "
            f"骨架点={total:6d} | "
            f"严格={exact:.4f} | "
            f"容差2px={tolerant:.4f}"
        )


# =====================================
# 7. 保存逐图统计
# =====================================

with open(
    OUT / "per_image_geometry.csv",
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(csv_rows[0].keys())
    )

    writer.writeheader()
    writer.writerows(csv_rows)


# =====================================
# 8. 绘制宽度分组结果
# =====================================

fig, axes = plt.subplots(
    1, 2,
    figsize=(12, 4.5),
    constrained_layout=True
)

xpos = np.arange(3)
bar_width = 0.35

for j, name in enumerate(CONFIGS):

    s = stats[name]

    total = s["width_total"]

    exact = np.divide(
        s["width_exact"],
        total,
        out=np.zeros(3, dtype=float),
        where=total > 0
    )

    tolerant = np.divide(
        s["width_tol"],
        total,
        out=np.zeros(3, dtype=float),
        where=total > 0
    )

    offset = (j - 0.5) * bar_width

    axes[0].bar(
        xpos + offset,
        exact,
        bar_width,
        label=name
    )

    axes[1].bar(
        xpos + offset,
        tolerant,
        bar_width,
        label=name
    )

for ax, title in zip(
    axes,
    ["Exact Centerline Recall",
     "Centerline Recall (2px tolerance)"]
):

    ax.set_title(title)
    ax.set_xticks(xpos)
    ax.set_xticklabels(WIDTH_LABELS)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Recall")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)

plt.savefig(
    OUT / "width_recall.png",
    dpi=160
)


# =====================================
# 9. 可视化典型困难图像
# =====================================

# 优先选择第41课发现的困难样本
target_names = [
    "IMG11-1.jpg",
    "11122-2.jpg",
    "IMG39-1.jpg"
]

selected = []

for name in target_names:
    for r in records:
        if r["name"] == name:
            selected.append(r)
            break

# 如果上述图像没有匹配到，自动选择困难样本
if not selected:

    selected = sorted(
        records,
        key=lambda r: r["models"]["Baseline"][
            "skeleton_recall"
        ]
    )[:3]


def create_overlay(sample, model_name):

    image = sample["image"].copy()

    gt_skel = sample["gt_skel"]

    result = sample["models"][model_name]

    missed = result["missed"]
    far = result["far"]

    # 绿色：真实裂纹骨架位置
    image[gt_skel] = [0.1, 0.85, 0.2]

    # 青色：容差2px后仍漏检
    image[missed] = [0.0, 0.9, 1.0]

    # 红色：距离GT超过3px的预测骨架
    image[far] = [1.0, 0.15, 0.1]

    return image


fig, axes = plt.subplots(
    len(selected),
    4,
    figsize=(16, 4 * len(selected)),
    squeeze=False,
    constrained_layout=True
)

for i, sample in enumerate(selected):

    axes[i, 0].imshow(sample["image"])
    axes[i, 0].set_title(
        sample["name"] + "\nOriginal"
    )

    axes[i, 1].imshow(
        sample["gt"],
        cmap="gray",
        vmin=0,
        vmax=1
    )
    axes[i, 1].set_title("Ground Truth")

    for col, name in [
        (2, "Baseline"),
        (3, "Contrast Aug")
    ]:

        result = sample["models"][name]

        axes[i, col].imshow(
            create_overlay(sample, name)
        )

        axes[i, col].set_title(
            f"{name}\n"
            f"CenterlineR="
            f"{result['skeleton_recall']:.3f}"
        )

    for ax in axes[i]:
        ax.axis("off")

plt.savefig(
    OUT / "geometry_examples.png",
    dpi=160
)

plt.show()

print("\n分析完成。结果目录:", OUT)
