
from pathlib import Path
import torch
import matplotlib.pyplot as plt

from unet_model import MiniUNet
from lesson07_dataset import val_loader

ROOT = Path(__file__).resolve().parent
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# 保持两个模型使用相同阈值
THRESHOLD = 0.3

# 两个模型的权重路径
models = {
    "BCE + Dice": ROOT / "outputs" / "best_unet_bce_dice.pth",
    "BCE + Dice + Sampling": ROOT / "outputs" / "best_unet_sampling.pth"
}

# 固定同一张验证图像进行可视化
images, masks = next(iter(val_loader))

sample = images[0:1].to(device)
image_np = images[0].permute(1, 2, 0).numpy()
mask_np = masks[0, 0].numpy()

fig, axes = plt.subplots(1, 4, figsize=(15, 4))

axes[0].imshow(image_np)
axes[0].set_title("Original Image")

axes[1].imshow(mask_np, cmap="gray", vmin=0, vmax=1)
axes[1].set_title("Ground Truth")

# 分别加载模型，逐个预测
for i, (name, model_path) in enumerate(models.items()):

    if not model_path.exists():
        raise FileNotFoundError(
            f"找不到模型：{model_path}"
        )

    model = MiniUNet().to(device)

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device,
            weights_only=True
        )
    )

    model.eval()

    # 当前样本预测
    with torch.inference_mode():
        logits = model(sample)
        probabilities = torch.sigmoid(logits)
        prediction = (probabilities > THRESHOLD)

    pred_np = prediction[0, 0].cpu().numpy()

    # 当前样本Dice
    tp = ((pred_np == 1) & (mask_np == 1)).sum()
    total = pred_np.sum() + mask_np.sum()

    sample_dice = (
        2 * tp / total if total > 0 else 1.0
    )

    axes[i + 2].imshow(
        pred_np,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[i + 2].set_title(
        f"{name}\nDice={sample_dice:.4f}"
    )

    # 整个验证集评价
    TP, FP, FN = 0, 0, 0

    with torch.inference_mode():
        for batch_images, batch_masks in val_loader:

            batch_images = batch_images.to(device)
            batch_masks = batch_masks.to(device)

            batch_logits = model(batch_images)

            batch_pred = (
                torch.sigmoid(batch_logits) > THRESHOLD
            )

            batch_gt = batch_masks > 0.5

            TP += (batch_pred & batch_gt).sum().item()
            FP += (batch_pred & ~batch_gt).sum().item()
            FN += (~batch_pred & batch_gt).sum().item()

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    dice = 2 * TP / (2 * TP + FP + FN + 1e-8)

    print(f"\n{name}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Val Dice:  {dice:.4f}")

for ax in axes:
    ax.axis("off")

plt.tight_layout()

plt.savefig(
    ROOT / "outputs" / "compare_8_25.png",
    dpi=150
)

plt.show()
