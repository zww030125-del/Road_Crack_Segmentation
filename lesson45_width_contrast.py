from pathlib import Path
import csv

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from scipy import ndimage as ndi
from skimage.morphology import skeletonize

from models.resnet_unet import ResNetUNet
from datasets.crack_dataset import CrackDataset
from datasets.splits import get_train_val_pairs


# ========================================
# 1. 实验设置
# ========================================

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "lesson45"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

THRESHOLD = 0.73
TOLERANCE = 2

# 来自第41课的局部对比度分界点
LOW_CUT = 0.0779
HIGH_CUT = 0.2028

WIDTH_NAMES = [
    "Thin <=3px",
    "Medium 3-6px",
    "Thick >6px"
]

CONTRAST_NAMES = [
    "Low",
    "Medium",
    "High"
]

CONNECTIVITY = np.ones((3, 3), dtype=bool)


# ========================================
# 2. 读取验证数据
# ========================================

_, val_pairs = get_train_val_pairs(
    ROOT / "data",
    seed=42
)

dataset = CrackDataset(
    val_pairs,
    training=False,
    augment=False
)

model = ResNetUNet(
    pretrained=False,
    freeze_encoder=True
).to(DEVICE)

model.load_state_dict(
    torch.load(
        ROOT / "outputs" / "lesson42"
        / "best_baseline.pth",
        map_location=DEVICE,
        weights_only=True
    )
)

model.eval()

print("设备:", DEVICE)
print("验证集:", len(dataset))


# ========================================
# 3. 计算局部暗线对比度
# ========================================

def get_contrast(image):
    """
    image: [H, W, 3], 0~1
    """

    gray = (
        0.299 * image[:, :, 0]
        + 0.587 * image[:, :, 1]
        + 0.114 * image[:, :, 2]
    )

    gray_t = torch.from_numpy(
        gray.astype(np.float32)
    )[None, None]

    padded = F.pad(
        gray_t,
        (10, 10, 10, 10),
        mode="replicate"
    )

    local_mean = F.avg_pool2d(
        padded,
        kernel_size=21,
        stride=1
    )[0, 0].numpy()

    return np.maximum(local_mean - gray, 0)


# ========================================
# 4. 用于统计的9个分组
# ========================================

counts = np.zeros((3, 3), dtype=np.int64)
found_exact = np.zeros((3, 3), dtype=np.int64)
found_tolerant = np.zeros((3, 3), dtype=np.int64)

samples = []


# ========================================
# 5. 逐张图像分析
# ========================================

with torch.inference_mode():

    for index in range(len(dataset)):

        image_t, mask_t = dataset[index]

        image = image_t.permute(
            1, 2, 0
        ).numpy()

        gt = mask_t[0].numpy() > 0.5

        x = image_t.unsqueeze(0).to(DEVICE)

        prob = torch.sigmoid(
            model(x)
        )[0, 0].cpu().numpy()

        pred = prob > THRESHOLD

        # ------- 真实骨架 -------
        gt_skel = skeletonize(gt)

        # ------- Mask宽度 -------
        distance = ndi.distance_transform_edt(gt)
        width_map = 2.0 * distance

        # ------- 局部对比度 -------
        contrast_map = get_contrast(image)

        # ------- 允许2px偏差 -------
        pred_dilated = ndi.binary_dilation(
            pred,
            structure=CONNECTIVITY,
            iterations=TOLERANCE
        )

        # 获取真实骨架坐标
        yy, xx = np.nonzero(gt_skel)

        if len(yy) == 0:
            continue

        widths = width_map[yy, xx]
        contrasts = contrast_map[yy, xx]

        # 宽度分组：
        # 0: <=3
        # 1: (3,6]
        # 2: >6
        width_group = np.digitize(
            widths,
            [3.0, 6.0],
            right=True
        )

        # 对比度分组
        contrast_group = np.digitize(
            contrasts,
            [LOW_CUT, HIGH_CUT],
            right=True
        )

        # 9个组合分别编号0~8
        group_id = width_group * 3 + contrast_group

        # 严格检出
        exact = pred[yy, xx]

        # 2px容差检出
        tolerant = pred_dilated[yy, xx]

        counts += np.bincount(
            group_id,
            minlength=9
        ).reshape(3, 3)

        found_exact += np.bincount(
            group_id[exact],
            minlength=9
        ).reshape(3, 3)

        found_tolerant += np.bincount(
            group_id[tolerant],
            minlength=9
        ).reshape(3, 3)

        # ------- 找出细且低对比度的漏检 -------
        thin_low = (
            gt_skel
            & (width_map <= 3)
            & (contrast_map <= LOW_CUT)
        )

        missed = gt_skel & ~pred_dilated
        thin_low_missed = thin_low & missed

        samples.append({
            "name": val_pairs[index][0].name,
            "image": image,
            "gt": gt,
            "pred": pred,
            "gt_skel": gt_skel,
            "missed": missed,
            "thin_low_missed": thin_low_missed,
            "thin_low_count": int(thin_low.sum()),
            "thin_low_miss_count": int(
                thin_low_missed.sum()
            )
        })


# ========================================
# 6. 统计Recall
# ========================================

recall_exact = np.divide(
    found_exact,
    counts,
    out=np.full((3, 3), np.nan),
    where=counts > 0
)

recall_tolerant = np.divide(
    found_tolerant,
    counts,
    out=np.full((3, 3), np.nan),
    where=counts > 0
)

missed_counts = counts - found_tolerant

total_points = int(counts.sum())
total_missed = int(missed_counts.sum())

print("\n========== 联合诊断结果 ==========")

print("真实骨架点总数:", total_points)
print("容差2px未覆盖点数:", total_missed)

print("\n各组合：骨架点数 / Recall(2px)")

for i, width_name in enumerate(WIDTH_NAMES):

    print(f"\n{width_name}")

    for j, contrast_name in enumerate(CONTRAST_NAMES):

        n = counts[i, j]
        r = recall_tolerant[i, j]

        if n > 0:
            print(
                f"  {contrast_name:8s} "
                f"| Count={n:6d} "
                f"| Recall={r:.4f} "
                f"| Missed={missed_counts[i, j]}"
            )
        else:
            print(
                f"  {contrast_name:8s} "
                "| 无骨架点"
            )


# ========================================
# 7. 重点统计：细+低对比度
# ========================================

thin_low_total = int(counts[0, 0])
thin_low_missed = int(missed_counts[0, 0])

print("\n========== 细+低对比度 ==========")

if total_points > 0:
    print(
        "占全部骨架点比例:",
        f"{thin_low_total / total_points:.2%}"
    )

if total_missed > 0:
    print(
        "占全部漏检骨架点比例:",
        f"{thin_low_missed / total_missed:.2%}"
    )

if thin_low_total > 0:
    print(
        "该组合骨架召回率:",
        f"{recall_tolerant[0, 0]:.4f}"
    )


# ========================================
# 8. 保存联合统计CSV
# ========================================

csv_rows = []

for i in range(3):
    for j in range(3):

        csv_rows.append({
            "width": WIDTH_NAMES[i],
            "contrast": CONTRAST_NAMES[j],
            "gt_skeleton_points": int(counts[i, j]),
            "exact_recall": recall_exact[i, j],
            "tolerant_recall": recall_tolerant[i, j],
            "missed_points": int(missed_counts[i, j])
        })

with open(
    OUT / "joint_statistics.csv",
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


# ========================================
# 9. 画两个热力图
# ========================================

fig, axes = plt.subplots(
    1, 2,
    figsize=(12, 5),
    constrained_layout=True
)

plots = [
    (
        recall_tolerant,
        "Centerline Recall (2px)",
        "viridis",
        0,
        1
    ),
    (
        missed_counts,
        "Missed Centerline Points",
        "magma",
        0,
        None
    )
]

for ax, (data, title, cmap, vmin, vmax) in zip(
    axes, plots
):

    im = ax.imshow(
        data,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))

    ax.set_xticklabels(CONTRAST_NAMES)
    ax.set_yticklabels(WIDTH_NAMES)

    ax.set_xlabel("Local Contrast")
    ax.set_ylabel("Mask Width")
    ax.set_title(title)

    for i in range(3):
        for j in range(3):

            if counts[i, j] == 0:
                label = "N/A"
            elif title.startswith("Centerline"):
                label = (
                    f"{data[i, j]:.3f}\n"
                    f"n={counts[i, j]}"
                )
            else:
                label = (
                    f"{int(data[i, j])}\n"
                    f"n={counts[i, j]}"
                )

            ax.text(
                j, i, label,
                ha="center",
                va="center",
                fontsize=9,
                color="white"
            )

    fig.colorbar(
        im,
        ax=ax,
        shrink=0.8
    )

plt.savefig(
    OUT / "joint_heatmap.png",
    dpi=160
)


# ========================================
# 10. 可视化细+低对比度的困难样本
# ========================================

# 按这类漏检骨架点数量排序
selected = sorted(
    samples,
    key=lambda s: s["thin_low_miss_count"],
    reverse=True
)[:3]

fig, axes = plt.subplots(
    len(selected),
    4,
    figsize=(16, 4 * len(selected)),
    squeeze=False,
    constrained_layout=True
)

for i, sample in enumerate(selected):

    image = sample["image"]
    gt = sample["gt"]
    pred = sample["pred"]

    overlay = image.copy()

    # 绿色：全部真实骨架
    green = ndi.binary_dilation(
        sample["gt_skel"],
        iterations=1
    )

    # 青色：其他漏检骨架点
    other_missed = (
        sample["missed"]
        & ~sample["thin_low_missed"]
    )

    cyan = ndi.binary_dilation(
        other_missed,
        iterations=1
    )

    # 红色：细+低对比度的漏检点
    red = ndi.binary_dilation(
        sample["thin_low_missed"],
        iterations=1
    )

    overlay[green] = [0.0, 0.85, 0.2]
    overlay[cyan] = [0.0, 0.9, 1.0]
    overlay[red] = [1.0, 0.1, 0.1]

    panels = [
        ("Original", image, None),
        ("GT", gt, "gray"),
        ("Prediction", pred, "gray"),
        ("Missed Centerline", overlay, None)
    ]

    for j, (title, arr, cmap) in enumerate(panels):

        ax = axes[i, j]

        ax.imshow(
            arr,
            cmap=cmap,
            vmin=0 if cmap else None,
            vmax=1 if cmap else None
        )

        ax.set_title(
            f"{title}\n{sample['name']}"
            if j == 0 else title
        )

        ax.axis("off")

plt.savefig(
    OUT / "thin_low_examples.png",
    dpi=160
)

plt.show()

print("\n实验完成，结果保存在:", OUT)