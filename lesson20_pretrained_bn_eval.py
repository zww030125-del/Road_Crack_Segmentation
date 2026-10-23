
from pathlib import Path
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from resnet_unet import ResNetUNet
from lesson07_dataset import (
    CrackDataset,
    train_pairs,
    val_loader
)

# ==================================
# 1. 配置
# ==================================
# True：ImageNet预训练
# False：完全随机初始化
USE_PRETRAINED = True

EXPERIMENT_NAME = (
    "pretrained_bn_eval"
    if USE_PRETRAINED
    else "scratch_full"
)
random.seed(42)
torch.manual_seed(42)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

EPOCHS = 15
BATCH_SIZE = 1

output_dir = Path(__file__).resolve().parent / "outputs"
output_dir.mkdir(exist_ok=True)

train_dataset = CrackDataset(
    train_pairs,
    training=True,
    augment=False
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

# ==================================
# 2. 加载预训练模型并冻结编码器
# ==================================

model = ResNetUNet(
    pretrained=USE_PRETRAINED,
    freeze_encoder=False
).to(device)

total_params = sum(
    p.numel() for p in model.parameters()
)

trainable_params = sum(
    p.numel() for p in model.parameters()
    if p.requires_grad
)

print("训练设备:", device)
print("总参数量:", total_params)
print("可训练参数量:", trainable_params)

# ==================================
# 3. BCE + Dice Loss
# ==================================

bce_loss = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([5.0], device=device)
)

def dice_loss(logits, targets):
    probs = torch.sigmoid(logits)

    dims = (1, 2, 3)

    intersection = (probs * targets).sum(dim=dims)
    denominator = (
        probs.sum(dim=dims) + targets.sum(dim=dims)
    )

    dice = (
        2 * intersection + 1.0
    ) / (denominator + 1.0)

    return 1 - dice.mean()

def criterion(logits, targets):
    return (
        bce_loss(logits, targets)
        + 0.5 * dice_loss(logits, targets)
    )

# 只优化可训练的参数
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)

# ==================================
# 4. 训练
# ==================================

best_dice = -1.0

for epoch in range(EPOCHS):
    model.train()

    # 固定预训练编码器的BatchNorm运行统计量
    for module in model.encoder.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
    total_loss = 0.0

    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    avg_loss = total_loss / len(train_dataset)

    # ==================================
    # 5. 验证
    # ==================================

    model.eval()

    tp, fp, fn = 0, 0, 0

    with torch.inference_mode():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)

            pred = torch.sigmoid(logits) > 0.5
            gt = masks > 0.5

            tp += (pred & gt).sum().item()
            fp += (pred & ~gt).sum().item()
            fn += (~pred & gt).sum().item()

    val_dice = (
        2 * tp / (2 * tp + fp + fn)
        if (2 * tp + fp + fn) > 0 else 1.0
    )

    print(
        f"{EXPERIMENT_NAME} | "
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss: {avg_loss:.4f} | "
        f"Val Dice: {val_dice:.4f}",
        flush=True
    )

    if val_dice > best_dice:
        best_dice = val_dice

        torch.save(
            model.state_dict(),
            output_dir / "best_resnet18_pretrained_bn_eval.pth"
        )

print("\n冻结编码器训练完成")
print(f"最佳验证Dice: {best_dice:.4f}")
