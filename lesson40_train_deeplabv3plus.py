
from pathlib import Path
import csv
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp

from datasets.crack_dataset import CrackDataset
from datasets.splits import get_train_val_pairs


# =====================================
# 1. 基本配置
# =====================================

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "deeplab_experiment"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEED = 42
EPOCHS = 25
BATCH_SIZE = 2
LR = 0.001

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)

print("设备:", DEVICE)


# =====================================
# 2. 数据集
# =====================================

train_pairs, val_pairs = get_train_val_pairs(
    ROOT / "data", seed=SEED
)

train_dataset = CrackDataset(
    train_pairs,
    training=True,
    augment=False
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

print("训练集:", len(train_dataset))
print("验证集:", len(val_dataset))


# =====================================
# 3. 构建DeepLabV3+
# =====================================

model = smp.DeepLabV3Plus(
    encoder_name="resnet18",
    encoder_weights="imagenet",
    encoder_output_stride=16,
    decoder_channels=128,
    decoder_atrous_rates=(3, 6, 9),
    in_channels=3,
    classes=1,
    activation=None
).to(DEVICE)

# 冻结预训练编码器
for param in model.encoder.parameters():
    param.requires_grad = False

total_params = sum(
    p.numel() for p in model.parameters()
)

trainable_params = sum(
    p.numel() for p in model.parameters()
    if p.requires_grad
)

print("总参数量:", total_params)
print("可训练参数量:", trainable_params)


# =====================================
# 4. ImageNet输入归一化
# =====================================

mean = torch.tensor(
    [0.485, 0.456, 0.406],
    device=DEVICE
).view(1, 3, 1, 1)

std = torch.tensor(
    [0.229, 0.224, 0.225],
    device=DEVICE
).view(1, 3, 1, 1)


def preprocess(images):
    return (images - mean) / std


# =====================================
# 5. BCE + Dice Loss
# =====================================

bce = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([5.0], device=DEVICE)
)


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


def criterion(logits, masks):
    return bce(logits, masks) + 0.5 * dice_loss(
        logits, masks
    )


optimizer = torch.optim.Adam(
    (p for p in model.parameters() if p.requires_grad),
    lr=LR
)


# =====================================
# 6. 验证
# =====================================

@torch.inference_mode()
def validate():

    model.eval()

    tp = fp = fn = 0

    for images, masks in val_loader:

        images = preprocess(images.to(DEVICE))
        masks = masks.to(DEVICE)

        logits = model(images)
        prob = torch.sigmoid(logits)

        pred = prob > 0.5
        gt = masks > 0.5

        tp += (pred & gt).sum().item()
        fp += (pred & ~gt).sum().item()
        fn += (~pred & gt).sum().item()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    denominator = 2 * tp + fp + fn

    dice = (
        2 * tp / denominator
        if denominator > 0 else 1.0
    )

    return precision, recall, dice


# =====================================
# 7. 正式训练
# =====================================

best_dice = -1.0
history = []

for epoch in range(EPOCHS):

    model.train()

    # 编码器冻结，同时固定其BatchNorm统计量
    model.encoder.eval()

    total_loss = 0.0

    for images, masks in train_loader:

        images = preprocess(images.to(DEVICE))
        masks = masks.to(DEVICE)

        optimizer.zero_grad()

        logits = model(images)

        loss = criterion(logits, masks)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    avg_loss = total_loss / len(train_dataset)

    precision, recall, val_dice = validate()

    history.append({
        "epoch": epoch + 1,
        "train_loss": avg_loss,
        "precision": precision,
        "recall": recall,
        "val_dice": val_dice
    })

    print(
        f"Epoch {epoch+1:02d}/{EPOCHS} | "
        f"Loss: {avg_loss:.4f} | "
        f"P: {precision:.4f} | "
        f"R: {recall:.4f} | "
        f"Dice: {val_dice:.4f}",
        flush=True
    )

    if val_dice > best_dice:

        best_dice = val_dice

        torch.save(
            model.state_dict(),
            OUT / "best_deeplabv3plus.pth"
        )

# =====================================
# 8. 保存训练日志
# =====================================

with open(
    OUT / "training_history.csv",
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "epoch",
            "train_loss",
            "precision",
            "recall",
            "val_dice"
        ]
    )

    writer.writeheader()
    writer.writerows(history)

print("\nDeepLabV3+训练完成")
print(f"最佳验证Dice: {best_dice:.4f}")
print("权重保存目录:", OUT)
