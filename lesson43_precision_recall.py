
from pathlib import Path
import csv

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from models.resnet_unet import ResNetUNet
from datasets.crack_dataset import CrackDataset
from datasets.splits import get_train_val_pairs


# ============================================
# 1. 配置
# ============================================

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "lesson43"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEED = 42
TARGET_PRECISION = 0.845

# 与第42课保持相同的局部对比度分界
LOW_CUT = 0.0779

# 搜索0.05~0.95，步长0.01
thresholds = (
    torch.arange(5, 96, device=DEVICE)
    .float() / 100
)

CHECKPOINTS = {
    "Baseline": ROOT / "outputs" / "lesson42"
                / "best_baseline.pth",

    "Contrast Aug": ROOT / "outputs" / "lesson42"
                    / "best_contrast_aug.pth"
}


# ============================================
# 2. 验证集
# ============================================

_, val_pairs = get_train_val_pairs(
    ROOT / "data",
    seed=SEED
)

val_dataset = CrackDataset(
    val_pairs,
    training=False,
    augment=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0
)

print("设备:", DEVICE)
print("验证集数量:", len(val_dataset))


# ============================================
# 3. 局部亮度对比度
# ============================================

def calculate_contrast(images):

    gray = (
        0.299 * images[:, 0:1]
        + 0.587 * images[:, 1:2]
        + 0.114 * images[:, 2:3]
    )

    padded = F.pad(
        gray,
        (10, 10, 10, 10),
        mode="replicate"
    )

    local_mean = F.avg_pool2d(
        padded,
        kernel_size=21,
        stride=1
    )

    return (local_mean - gray).clamp_min(0)


# ============================================
# 4. 统计不同阈值的指标
# ============================================

@torch.inference_mode()
def evaluate_model(model):

    model.eval()

    tp = torch.zeros_like(thresholds)
    fp = torch.zeros_like(thresholds)
    fn = torch.zeros_like(thresholds)

    low_tp = torch.zeros_like(thresholds)
    low_count = 0

    for images, masks in val_loader:

        images = images.to(DEVICE)
        gt = masks.to(DEVICE) > 0.5

        prob = torch.sigmoid(model(images))

        # [91, B, 1, H, W]
        pred = (
            prob.unsqueeze(0)
            > thresholds[:, None, None, None, None]
        )

        gt_expanded = gt.unsqueeze(0)

        tp += (pred & gt_expanded).sum(
            dim=(1, 2, 3, 4)
        )

        fp += (pred & ~gt_expanded).sum(
            dim=(1, 2, 3, 4)
        )

        fn += (~pred & gt_expanded).sum(
            dim=(1, 2, 3, 4)
        )

        # 低对比度真实裂纹像素
        contrast = calculate_contrast(images)

        low = (
            gt & (contrast <= LOW_CUT)
        )

        low_count += low.sum().item()

        low_tp += (
            pred & low.unsqueeze(0)
        ).sum(dim=(1, 2, 3, 4))

    precision = tp / (tp + fp).clamp_min(1)
    recall = tp / (tp + fn).clamp_min(1)

    dice = (
        2 * tp /
        (2 * tp + fp + fn).clamp_min(1)
    )

    low_recall = low_tp / max(low_count, 1)

    return {
        "threshold": thresholds.cpu().numpy(),
        "precision": precision.cpu().numpy(),
        "recall": recall.cpu().numpy(),
        "dice": dice.cpu().numpy(),
        "low_recall": low_recall.cpu().numpy()
    }


# ============================================
# 5. 两套模型分别评价
# ============================================

results = {}

for name, checkpoint in CHECKPOINTS.items():

    print(f"\n正在评价: {name}")

    model = ResNetUNet(
        pretrained=False,
        freeze_encoder=True
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(
            checkpoint,
            map_location=DEVICE,
            weights_only=True
        )
    )

    result = evaluate_model(model)
    results[name] = result

    # -------- 各自最佳Dice --------
    best_idx = result["dice"].argmax()

    print("\n【最佳Dice】")
    print(
        f"Threshold: "
        f"{result['threshold'][best_idx]:.2f}"
    )
    print(
        f"Precision: "
        f"{result['precision'][best_idx]:.4f}"
    )
    print(
        f"Recall: "
        f"{result['recall'][best_idx]:.4f}"
    )
    print(
        f"Dice: "
        f"{result['dice'][best_idx]:.4f}"
    )
    print(
        f"Low Recall: "
        f"{result['low_recall'][best_idx]:.4f}"
    )

    # -------- 相近Precision对比 --------
    # 选择Precision最接近目标值的阈值
    matched_idx = abs(
        result["precision"] - TARGET_PRECISION
    ).argmin()

    print(
        f"\n【Precision接近{TARGET_PRECISION:.3f}】"
    )

    print(
        f"Threshold: "
        f"{result['threshold'][matched_idx]:.2f}"
    )
    print(
        f"Precision: "
        f"{result['precision'][matched_idx]:.4f}"
    )
    print(
        f"Recall: "
        f"{result['recall'][matched_idx]:.4f}"
    )
    print(
        f"Dice: "
        f"{result['dice'][matched_idx]:.4f}"
    )
    print(
        f"Low Recall: "
        f"{result['low_recall'][matched_idx]:.4f}"
    )

    p_gap = abs(
        result["precision"][matched_idx]
        - TARGET_PRECISION
    )

    if p_gap > 0.01:
        print(
            "注意：最接近目标的Precision仍存在"
            "超过0.01的差异，暂不视为严格匹配。"
        )

    # -------- 保存每个阈值的统计 --------
    csv_path = OUT / (
        name.lower().replace(" ", "_") + "_metrics.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "threshold",
            "precision",
            "recall",
            "dice",
            "low_recall"
        ])

        for i in range(len(result["threshold"])):

            writer.writerow([
                result["threshold"][i],
                result["precision"][i],
                result["recall"][i],
                result["dice"][i],
                result["low_recall"][i]
            ])


# ============================================
# 6. 绘制比较曲线
# ============================================

fig, axes = plt.subplots(
    1, 3,
    figsize=(16, 4.5),
    constrained_layout=True
)

for name, r in results.items():

    # Precision-Recall曲线
    axes[0].plot(
        r["recall"],
        r["precision"],
        marker=".",
        markersize=3,
        label=name
    )

    # Dice-Threshold曲线
    axes[1].plot(
        r["threshold"],
        r["dice"],
        label=name
    )

    # 固定Precision附近的低对比度召回表现
    axes[2].plot(
        r["precision"],
        r["low_recall"],
        marker=".",
        markersize=3,
        label=name
    )


axes[0].set_title("Precision-Recall")
axes[0].set_xlabel("Recall")
axes[0].set_ylabel("Precision")

axes[1].set_title("Dice vs Threshold")
axes[1].set_xlabel("Threshold")
axes[1].set_ylabel("Dice")

axes[2].set_title("Low-Contrast Recall")
axes[2].set_xlabel("Precision")
axes[2].set_ylabel("Low-Contrast Recall")

for ax in axes:
    ax.grid(alpha=0.2)
    ax.legend()

fig.savefig(
    OUT / "model_comparison.png",
    dpi=160
)

plt.show()

print("\n对照实验完成")
print("结果保存目录:", OUT)
