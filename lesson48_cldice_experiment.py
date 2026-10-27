
from pathlib import Path
import random
import csv

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from torch.utils.data import DataLoader

from models.resnet_unet import ResNetUNet
from datasets.crack_dataset import CrackDataset
from datasets.splits import get_train_val_pairs
from losses.soft_cldice import soft_cldice_loss


# =====================================
# 1. 实验配置
# =====================================

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "lesson48"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SPLIT_SEED = 42
TRAIN_SEED = 42

EPOCHS = 15
BATCH_SIZE = 2
LR = 0.001

FIXED_THRESHOLD = 0.73
TARGET_PRECISION = 0.845

CLDICE_WEIGHT = 0.25
SKELETON_ITER = 10

LOW_CUT = 0.0779
NEIGHBORS = np.ones((3, 3), dtype=bool)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================
# 2. 数据集
# =====================================

train_pairs, val_pairs = get_train_val_pairs(
    ROOT / "data",
    seed=SPLIT_SEED
)

val_dataset = CrackDataset(
    val_pairs,
    training=False,
    augment=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# =====================================
# 3. 损失函数
# =====================================

bce = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([5.0], device=DEVICE)
)


def dice_loss(logits, masks):
    prob = torch.sigmoid(logits)
    dims = (1, 2, 3)

    intersection = (prob * masks).sum(dim=dims)
    denominator = (
        prob.sum(dim=dims) + masks.sum(dim=dims)
    )

    dice = (
        2 * intersection + 1
    ) / (denominator + 1)

    return 1 - dice.mean()


def calculate_loss(logits, masks, use_cldice):
    loss = bce(logits, masks)
    loss = loss + 0.5 * dice_loss(logits, masks)

    if use_cldice:
        loss = loss + CLDICE_WEIGHT * soft_cldice_loss(
            logits, masks,
            iterations=SKELETON_ITER
        )

    return loss


# =====================================
# 4. 验证Dice
# =====================================

@torch.inference_mode()
def validate(model):
    model.eval()

    tp = fp = fn = 0

    for images, masks in val_loader:
        images = images.to(DEVICE)
        gt = masks.to(DEVICE) > 0.5

        pred = (
            torch.sigmoid(model(images))
            > FIXED_THRESHOLD
        )

        tp += (pred & gt).sum().item()
        fp += (pred & ~gt).sum().item()
        fn += (~pred & gt).sum().item()

    return 2 * tp / max(2 * tp + fp + fn, 1)


# =====================================
# 5. 配对训练
# =====================================

def train_one(name, use_cldice):
    print(f"\n========== {name} ==========")

    # 两组重新设置相同随机种子
    set_seed(TRAIN_SEED)

    train_dataset = CrackDataset(
        train_pairs,
        training=True,
        augment=False
    )

    loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(
            TRAIN_SEED
        )
    )

    model = ResNetUNet(
        pretrained=True,
        freeze_encoder=True
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        (
            p for p in model.parameters()
            if p.requires_grad
        ),
        lr=LR
    )

    best_dice = -1
    history = []

    for epoch in range(EPOCHS):
        model.train()
        model.encoder.eval()

        total_loss = 0.0

        for images, masks in loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()

            logits = model(images)

            loss = calculate_loss(
                logits, masks, use_cldice
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)

        avg_loss = total_loss / len(train_dataset)
        val_dice = validate(model)

        history.append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "dice": val_dice
        })

        print(
            f"{name} | Epoch {epoch+1:02d}/{EPOCHS}"
            f" | Loss={avg_loss:.4f}"
            f" | ValDice={val_dice:.4f}",
            flush=True
        )

        if val_dice > best_dice:
            best_dice = val_dice

            torch.save(
                model.state_dict(),
                OUT / f"best_{name}.pth"
            )

    with open(
        OUT / f"{name}_history.csv",
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(history[0].keys())
        )
        writer.writeheader()
        writer.writerows(history)

    print(
        f"{name} 最佳Dice(0.73): {best_dice:.4f}"
    )

    return history


history_base = train_one(
    "baseline", use_cldice=False
)

history_cldice = train_one(
    "cldice", use_cldice=True
)


# =====================================
# 6. 收集验证集概率
# =====================================

@torch.inference_mode()
def collect_predictions(name):

    model = ResNetUNet(
        pretrained=False,
        freeze_encoder=True
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(
            OUT / f"best_{name}.pth",
            map_location=DEVICE,
            weights_only=True
        )
    )

    model.eval()

    probs, masks = [], []

    for images, gt in val_loader:
        pred_prob = torch.sigmoid(
            model(images.to(DEVICE))
        )[:, 0].cpu().numpy()

        probs.append(pred_prob)
        masks.append(gt[:, 0].numpy() > 0.5)

    return np.concatenate(probs), np.concatenate(masks)


# =====================================
# 7. 阈值与PR分析
# =====================================

THRESHOLDS = np.arange(5, 96) / 100


def scan_thresholds(prob, gt):
    rows = []

    for t in THRESHOLDS:
        pred = prob > t

        tp = np.count_nonzero(pred & gt)
        fp = np.count_nonzero(pred & ~gt)
        fn = np.count_nonzero(~pred & gt)

        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        dice = 2 * tp / max(2 * tp + fp + fn, 1)

        rows.append({
            "threshold": t,
            "precision": p,
            "recall": r,
            "dice": dice
        })

    return rows


def choose_matched(rows):
    feasible = [
        r for r in rows
        if r["precision"] >= TARGET_PRECISION
    ]

    if feasible:
        return max(feasible, key=lambda x: x["recall"])

    return min(
        rows,
        key=lambda x: abs(
            x["precision"] - TARGET_PRECISION
        )
    )


# =====================================
# 8. 评价真实骨架和路径
# =====================================

def geometry_metrics(prob, gt, threshold):

    thin_hit = thin_total = 0
    low_hit = low_total = 0

    gt_skel_hit = gt_skel_total = 0
    pred_skel_inside = pred_skel_total = 0

    far_skel_total = 0
    missing_segments = 0

    for i in range(len(prob)):
        pred = prob[i] > threshold
        mask = gt[i]

        gt_skel = skeletonize(mask)
        pred_skel = skeletonize(pred)

        # GT宽度和局部对比度
        width = 2 * ndi.distance_transform_edt(mask)

        image_t, _ = val_dataset[i]
        image = image_t.permute(1, 2, 0).numpy()

        gray = (
            0.299 * image[:, :, 0]
            + 0.587 * image[:, :, 1]
            + 0.114 * image[:, :, 2]
        )

        mean = ndi.uniform_filter(
            gray, size=21, mode="nearest"
        )
        contrast = np.maximum(mean - gray, 0)

        thin = gt_skel & (width <= 3)
        thin_low = thin & (contrast <= LOW_CUT)

        # 容差2px
        pred_near = ndi.binary_dilation(
            pred,
            structure=NEIGHBORS,
            iterations=2
        )

        thin_hit += np.count_nonzero(thin & pred_near)
        thin_total += np.count_nonzero(thin)

        low_hit += np.count_nonzero(
            thin_low & pred_near
        )
        low_total += np.count_nonzero(thin_low)

        # 严格的硬骨架clDice组成部分
        gt_skel_hit += np.count_nonzero(
            gt_skel & pred
        )
        gt_skel_total += np.count_nonzero(gt_skel)

        pred_skel_inside += np.count_nonzero(
            pred_skel & mask
        )
        pred_skel_total += np.count_nonzero(pred_skel)

        # 远离GT的预测骨架
        gt_near = ndi.binary_dilation(
            mask,
            structure=NEIGHBORS,
            iterations=3
        )

        far_skel_total += np.count_nonzero(
            pred_skel & ~gt_near
        )

        # 未覆盖真实骨架上的连续候选片段
        missed = gt_skel & ~pred_near

        labels, _ = ndi.label(
            missed,
            structure=NEIGHBORS
        )
        sizes = np.bincount(
            labels.ravel()
        )[1:]

        missing_segments += int(
            np.count_nonzero(sizes >= 5)
        )

    tprec = (
        pred_skel_inside / max(pred_skel_total, 1)
    )
    tsens = (
        gt_skel_hit / max(gt_skel_total, 1)
    )

    hard_cldice = (
        2 * tprec * tsens / (tprec + tsens)
        if tprec + tsens > 0 else 0.0
    )

    return {
        "hard_cldice": hard_cldice,
        "thin_recall_2px": thin_hit / max(thin_total, 1),
        "thin_low_recall_2px": low_hit / max(low_total, 1),
        "missing_segments": missing_segments,
        "far_skeleton_ratio": (
            far_skel_total / max(pred_skel_total, 1)
        )
    }


# =====================================
# 9. 输出最终结果
# =====================================

results = {}
summary = []

for name in ["baseline", "cldice"]:

    prob, gt = collect_predictions(name)
    curve = scan_thresholds(prob, gt)

    best = max(curve, key=lambda x: x["dice"])
    matched = choose_matched(curve)

    geo = geometry_metrics(
        prob, gt, matched["threshold"]
    )

    results[name] = {
        "prob": prob,
        "curve": curve,
        "best": best,
        "matched": matched,
        "geometry": geo
    }

    row = {
        "model": name,
        "best_dice": best["dice"],
        "best_threshold": best["threshold"],
        "matched_threshold": matched["threshold"],
        "precision": matched["precision"],
        "recall": matched["recall"],
        "matched_dice": matched["dice"],
        **geo
    }

    summary.append(row)

    print(f"\n========== {name} ==========")

    for key, value in row.items():
        if isinstance(value, (float, np.floating)):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


with open(
    OUT / "summary.csv",
    "w", newline="", encoding="utf-8-sig"
) as f:
    writer = csv.DictWriter(
        f, fieldnames=list(summary[0].keys())
    )
    writer.writeheader()
    writer.writerows(summary)


# =====================================
# 10. 训练与PR曲线
# =====================================

fig, axes = plt.subplots(
    1, 2, figsize=(12, 4.5),
    constrained_layout=True
)

for name, history in [
    ("baseline", history_base),
    ("cldice", history_cldice)
]:
    axes[0].plot(
        [r["epoch"] for r in history],
        [r["dice"] for r in history],
        label=name
    )

for name, result in results.items():
    curve = result["curve"]

    axes[1].plot(
        [r["recall"] for r in curve],
        [r["precision"] for r in curve],
        label=name
    )

axes[0].set_title("Validation Dice")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Dice")

axes[1].set_title("Precision-Recall")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")

for ax in axes:
    ax.grid(alpha=0.2)
    ax.legend()

plt.savefig(
    OUT / "training_comparison.png",
    dpi=160
)


# =====================================
# 11. 典型困难图像对比
# =====================================

names = [
    "IMG11-1.jpg",
    "11150.jpg",
    "11178-3.jpg"
]

selected = [
    i for i, pair in enumerate(val_pairs)
    if pair[0].name in names
]

if selected:
    fig, axes = plt.subplots(
        len(selected), 4,
        figsize=(15, 4 * len(selected)),
        squeeze=False,
        constrained_layout=True
    )

    for row, idx in enumerate(selected):
        image_t, gt_t = val_dataset[idx]

        image = image_t.permute(
            1, 2, 0
        ).numpy()

        axes[row, 0].imshow(image)
        axes[row, 0].set_title(
            val_pairs[idx][0].name
        )

        axes[row, 1].imshow(
            gt_t[0].numpy(),
            cmap="gray", vmin=0, vmax=1
        )
        axes[row, 1].set_title("Ground Truth")

        for col, name in [
            (2, "baseline"),
            (3, "cldice")
        ]:
            r = results[name]
            threshold = r["matched"]["threshold"]
            pred = r["prob"][idx] > threshold

            axes[row, col].imshow(
                pred, cmap="gray", vmin=0, vmax=1
            )
            axes[row, col].set_title(
                f"{name}\nThreshold={threshold:.2f}"
            )

        for ax in axes[row]:
            ax.axis("off")

    plt.savefig(
        OUT / "difficult_examples.png",
        dpi=160
    )

plt.show()

print("\n第48课完成:", OUT)
