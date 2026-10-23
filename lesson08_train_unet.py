
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from unet_model import MiniUNet

# 直接复用上一节的数据加载器
from lesson07_dataset import train_loader, val_loader

torch.manual_seed(42)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("训练设备:", device)

# ====================================
# 1. 定义U-Net
# ====================================


model = MiniUNet().to(device)

# ====================================
# 2. 损失函数和优化器
# ====================================

# 初次实验，对少数类裂纹增加损失权重
criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([5.0], device=device)
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

EPOCHS = 25

output_dir = Path(__file__).resolve().parent / "outputs"
output_dir.mkdir(exist_ok=True)

train_losses = []
val_losses = []
val_dices = []

best_dice = -1.0

# ====================================
# 3. 正式训练
# ====================================

for epoch in range(EPOCHS):

    model.train()

    total_train_loss = 0.0
    train_count = 0

    for images, masks in train_loader:

        images = images.to(device)
        masks = masks.to(device)

        # 前向传播
        logits = model(images)
        loss = criterion(logits, masks)

        # 反向传播 + 参数更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item() * images.size(0)
        train_count += images.size(0)

    avg_train_loss = total_train_loss / train_count

    # ====================================
    # 4. 验证集评价
    # ====================================

    model.eval()

    total_val_loss = 0.0
    val_count = 0

    intersection = 0.0
    pred_pixels = 0.0
    true_pixels = 0.0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss = criterion(logits, masks)

            total_val_loss += loss.item() * images.size(0)
            val_count += images.size(0)

            probabilities = torch.sigmoid(logits)

            predictions = probabilities > 0.5
            targets = masks > 0.5

            intersection += (
                predictions & targets
            ).sum().item()

            pred_pixels += predictions.sum().item()
            true_pixels += targets.sum().item()

    avg_val_loss = total_val_loss / val_count

    # 数据集整体像素Dice
    val_dice = (
        2 * intersection + 1e-6
    ) / (
        pred_pixels + true_pixels + 1e-6
    )

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    val_dices.append(val_dice)

    print(
        f"Epoch {epoch+1:02d}/{EPOCHS} | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Val Loss: {avg_val_loss:.4f} | "
        f"Val Dice: {val_dice:.4f}",
        flush=True
    )

    # 保存验证Dice最高的模型
    if val_dice > best_dice:
        best_dice = val_dice

        torch.save(
            model.state_dict(),
            output_dir / "best_unet_25epochs.pth"
        )

print("\n训练完成！")
print(f"最佳验证Dice: {best_dice:.4f}")

# ====================================
# 5. 绘制训练曲线
# ====================================

epochs = range(1, EPOCHS + 1)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].plot(epochs, train_losses, label="Train Loss")
axes[0].plot(epochs, val_losses, label="Val Loss")
axes[0].set_title("Training and Validation Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(epochs, val_dices, marker="o")
axes[1].set_title("Validation Dice")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Dice")
axes[1].set_ylim(0, 1)
axes[1].grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    output_dir / "training_curves.png",
    dpi=150
)

plt.show()

