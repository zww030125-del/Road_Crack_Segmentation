
from pathlib import Path
import csv
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.mini_unet import MiniUNet
from models.mini_unet_pp import MiniUNetPP

from datasets.crack_dataset import CrackDataset
from datasets.splits import get_train_val_pairs


# =====================================
# 1. 配置
# =====================================

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "unetpp_experiment"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEED = 42
EPOCHS = 25
BATCH_SIZE = 2

print("训练设备:", DEVICE)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================
# 2. 数据划分
# =====================================

train_pairs, val_pairs = get_train_val_pairs(
    ROOT / "data",
    seed=SEED
)

print("训练集:", len(train_pairs))
print("验证集:", len(val_pairs))


# =====================================
# 3. 定义损失函数
# =====================================

def dice_loss(logits, targets):

    probs = torch.sigmoid(logits)

    dims = (1, 2, 3)

    intersection = (
        probs * targets
    ).sum(dim=dims)

    denominator = (
        probs.sum(dim=dims)
        + targets.sum(dim=dims)
    )

    dice = (
        2 * intersection + 1
    ) / (denominator + 1)

    return 1 - dice.mean()


bce = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor(
        [5.0], device=DEVICE
    )
)


def criterion(logits, masks):
    return (
        bce(logits, masks)
        + 0.5 * dice_loss(logits, masks)
    )


# =====================================
# 4. 验证函数
# =====================================

@torch.inference_mode()
def validate(model, loader):

    model.eval()

    tp = fp = fn = 0

    for images, masks in loader:

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        prob = torch.sigmoid(model(images))

        pred = prob > 0.5
        gt = masks > 0.5

        tp += (pred & gt).sum().item()
        fp += (pred & ~gt).sum().item()
        fn += (~pred & gt).sum().item()

    denominator = 2 * tp + fp + fn

    return (
        2 * tp / denominator
        if denominator > 0 else 1.0
    )


# =====================================
# 5. 训练单个模型
# =====================================

def train_model(name, model_class):

    # 两组实验使用相同随机种子
    set_seed(SEED)

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

    # 独立的随机生成器，固定训练样本顺序
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

    model = model_class().to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    params = sum(
        p.numel() for p in model.parameters()
    )

    print("\n" + "=" * 45)
    print("模型:", name)
    print("参数量:", params)

    best_dice = -1.0
    history = []

    for epoch in range(EPOCHS):

        model.train()
        total_loss = 0.0

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

        train_loss = (
            total_loss / len(train_dataset)
        )

        val_dice = validate(
            model, val_loader
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_dice": val_dice
        })

        print(
            f"{name} | "
            f"Epoch {epoch+1:02d}/{EPOCHS} | "
            f"Loss: {train_loss:.4f} | "
            f"Val Dice: {val_dice:.4f}",
            flush=True
        )

        if val_dice > best_dice:

            best_dice = val_dice

            torch.save(
                model.state_dict(),
                OUTPUT_DIR / f"best_{name}.pth"
            )

    # 保存完整训练曲线数据
    csv_path = OUTPUT_DIR / f"{name}_history.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_dice"
            ]
        )

        writer.writeheader()
        writer.writerows(history)

    print(
        f"\n{name} 最佳验证Dice: "
        f"{best_dice:.4f}"
    )

    return best_dice


# =====================================
# 6. 完成两个模型的对照
# =====================================

results = {}

results["MiniUNet"] = train_model(
    "MiniUNet", MiniUNet
)

results["MiniUNetPP"] = train_model(
    "MiniUNetPP", MiniUNetPP
)

print("\n========== 最终比较 ==========")

for name, dice in results.items():
    print(f"{name}: {dice:.4f}")
