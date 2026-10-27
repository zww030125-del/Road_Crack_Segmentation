
from pathlib import Path
import random
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from models.resnet_unet import ResNetUNet
from datasets.crack_dataset import CrackDataset
from datasets.splits import get_train_val_pairs


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "lesson42"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEED = 42
EPOCHS = 15
BATCH_SIZE = 2
THRESHOLD = 0.73

# 上一课得到的局部对比度分界点
LOW_CUT = 0.0779
HIGH_CUT = 0.2028


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================
# 1. 光照与对比度增强
# =====================================

class ContrastDataset(Dataset):

    def __init__(self, base, enabled, seed=42):
        self.base = base
        self.enabled = enabled

        # 独立随机数发生器，避免扰乱
        # 原始Dataset的随机裁剪序列
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):

        image, mask = self.base[idx]

        if self.enabled and self.rng.random() < 0.7:

            alpha = float(
                self.rng.uniform(0.65, 1.10)
            )

            beta = float(
                self.rng.uniform(-0.05, 0.05)
            )

            mean = image.mean(
                dim=(1, 2), keepdim=True
            )

            image = (
                (image - mean) * alpha
                + mean + beta
            ).clamp(0, 1)

        return image, mask


# =====================================
# 2. 数据划分
# =====================================

train_pairs, val_pairs = get_train_val_pairs(
    ROOT / "data",
    seed=SEED
)


# =====================================
# 3. Loss
# =====================================

def dice_loss(logits, masks):

    prob = torch.sigmoid(logits)
    dims = (1, 2, 3)

    intersection = (
        prob * masks
    ).sum(dim=dims)

    denominator = (
        prob.sum(dim=dims)
        + masks.sum(dim=dims)
    )

    dice = (
        2 * intersection + 1
    ) / (denominator + 1)

    return 1 - dice.mean()


bce = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([5.0], device=DEVICE)
)


def criterion(logits, masks):
    return (
        bce(logits, masks)
        + 0.5 * dice_loss(logits, masks)
    )


# =====================================
# 4. 验证并统计低对比度召回率
# =====================================

@torch.inference_mode()
def validate(model, loader):

    model.eval()

    tp = fp = fn = 0
    low_tp = low_gt = 0
    mid_tp = mid_gt = 0
    high_tp = high_gt = 0

    for images, masks in loader:

        images = images.to(DEVICE)
        gt = masks.to(DEVICE) > 0.5

        prob = torch.sigmoid(model(images))
        pred = prob > THRESHOLD

        tp += (pred & gt).sum().item()
        fp += (pred & ~gt).sum().item()
        fn += (~pred & gt).sum().item()

        # 与上一课一致的局部对比度计算
        gray = (
            0.299 * images[:, 0:1]
            + 0.587 * images[:, 1:2]
            + 0.114 * images[:, 2:3]
        )

        padded = F.pad(
            gray, (10, 10, 10, 10),
            mode="replicate"
        )

        local_mean = F.avg_pool2d(
            padded, kernel_size=21, stride=1
        )

        contrast = (
            local_mean - gray
        ).clamp_min(0)

        low = gt & (contrast <= LOW_CUT)

        mid = (
            gt
            & (contrast > LOW_CUT)
            & (contrast <= HIGH_CUT)
        )

        high = gt & (contrast > HIGH_CUT)

        low_tp += (pred & low).sum().item()
        low_gt += low.sum().item()

        mid_tp += (pred & mid).sum().item()
        mid_gt += mid.sum().item()

        high_tp += (pred & high).sum().item()
        high_gt += high.sum().item()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    dice = 2 * tp / max(2 * tp + fp + fn, 1)

    return {
        "precision": precision,
        "recall": recall,
        "dice": dice,
        "low_recall": low_tp / max(low_gt, 1),
        "mid_recall": mid_tp / max(mid_gt, 1),
        "high_recall": high_tp / max(high_gt, 1)
    }


# =====================================
# 5. 训练一组模型
# =====================================

def train_experiment(name, use_augmentation):

    print(f"\n========== {name} ==========")

    set_seed(SEED)

    base_train = CrackDataset(
        train_pairs,
        training=True,
        augment=False
    )

    train_dataset = ContrastDataset(
        base_train,
        enabled=use_augmentation,
        seed=SEED
    )

    val_dataset = CrackDataset(
        val_pairs,
        training=False,
        augment=False
    )

    generator = torch.Generator().manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=generator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
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
        lr=0.001
    )

    best_dice = -1
    best_result = None
    history = []

    for epoch in range(EPOCHS):

        model.train()
        model.encoder.eval()

        total_loss = 0

        for images, masks in train_loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, masks)

            loss.backward()
            optimizer.step()

            total_loss += (
                loss.item() * images.size(0)
            )

        avg_loss = total_loss / len(train_dataset)

        result = validate(model, val_loader)

        history.append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            **result
        })

        print(
            f"{name} | Epoch {epoch+1:02d}/{EPOCHS} "
            f"| Loss={avg_loss:.4f} "
            f"| Dice={result['dice']:.4f} "
            f"| LowR={result['low_recall']:.4f}",
            flush=True
        )

        if result["dice"] > best_dice:

            best_dice = result["dice"]
            best_result = result.copy()

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

    return best_result


# =====================================
# 6. 受控对照实验
# =====================================

baseline = train_experiment(
    "baseline", use_augmentation=False
)

augmented = train_experiment(
    "contrast_aug", use_augmentation=True
)


print("\n========== 最终比较 ==========")

for metric in [
    "dice",
    "precision",
    "recall",
    "low_recall",
    "mid_recall",
    "high_recall"
]:

    print(
        f"{metric:12s} | "
        f"Baseline: {baseline[metric]:.4f} | "
        f"Augmented: {augmented[metric]:.4f} | "
        f"Change: {augmented[metric]-baseline[metric]:+.4f}"
    )
