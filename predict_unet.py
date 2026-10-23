
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from unet_model import MiniUNet
from lesson07_dataset import val_loader

# ==================================
# 1. 基本配置
# ==================================

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "outputs" / "best_unet.pth"

THRESHOLD = 0.2

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("运行设备:", device)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"找不到模型文件：{MODEL_PATH}"
    )

# ==================================
# 2. 创建U-Net并加载训练好的权重
# ==================================

model = MiniUNet().to(device)

state_dict = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

model.load_state_dict(state_dict)

model.eval()

print("模型权重加载成功！")

# ==================================
# 3. 从验证集读取图像
# ==================================

images, masks = next(iter(val_loader))

images_gpu = images.to(device)

# ==================================
# 4. 使用模型预测
# ==================================

with torch.inference_mode():

    # 网络输出Logits
    logits = model(images_gpu)

    # 转换为裂纹概率
    probabilities = torch.sigmoid(logits)

    # 根据阈值得到二值Mask
    predictions = (
        probabilities > THRESHOLD
    ).float()

# 将预测结果从GPU转回CPU
predictions = predictions.cpu()

# ==================================
# 5. 取第一张图像进行分析
# ==================================

image = images[0].permute(1, 2, 0).numpy()

true_mask = masks[0, 0].numpy()

pred_mask = predictions[0, 0].numpy()

# 计算当前图像Dice
intersection = (true_mask * pred_mask).sum()

denominator = true_mask.sum() + pred_mask.sum()

dice = (
    (2 * intersection / denominator)
    if denominator > 0 else 1.0
)

print(f"当前预测阈值: {THRESHOLD}")
print(f"当前图像Dice: {dice:.4f}")

# ==================================
# 6. 绘制预测结果
# ==================================

fig, axes = plt.subplots(
    1, 3, figsize=(12, 4)
)

axes[0].imshow(image)
axes[0].set_title("Original Image")

axes[1].imshow(
    true_mask,
    cmap="gray",
    vmin=0,
    vmax=1
)
axes[1].set_title("Ground Truth")

axes[2].imshow(
    pred_mask,
    cmap="gray",
    vmin=0,
    vmax=1
)
axes[2].set_title(
    f"Prediction (Dice={dice:.3f})"
)

for ax in axes:
    ax.axis("off")

plt.tight_layout()

# 保存结果
output_path = ROOT / "outputs" / "prediction_comparison.png"

plt.savefig(output_path, dpi=150)

plt.show()
import numpy as np

# 使用同一次预测的概率图
prob_map = probabilities[0, 0].cpu().numpy()
gt = masks[0, 0].numpy() > 0.5

thresholds = [0.2, 0.3, 0.5]
results = {}

fig, axes = plt.subplots(1, 4, figsize=(14, 4))

axes[0].imshow(gt, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Ground Truth")

for i, t in enumerate(thresholds):

    pred = prob_map > t
    results[t] = pred

    tp = np.logical_and(pred, gt).sum()

    precision = tp / (pred.sum() + 1e-8)
    recall = tp / (gt.sum() + 1e-8)
    dice = 2 * tp / (pred.sum() + gt.sum() + 1e-8)

    print(
        f"阈值={t:.1f} | "
        f"预测裂纹像素={pred.sum()} | "
        f"Precision={precision:.4f} | "
        f"Recall={recall:.4f} | "
        f"Dice={dice:.4f}"
    )

    axes[i + 1].imshow(
        pred, cmap="gray", vmin=0, vmax=1
    )
    axes[i + 1].set_title(
        f"Threshold={t:.1f}\nDice={dice:.3f}"
    )

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()

# 检查两个阈值具体相差多少像素
diff = np.count_nonzero(
    results[0.2] != results[0.3]
)

print(f"\n阈值0.2和0.3相差的像素数量：{diff}")
# ==================================
# 显示U-Net原始预测概率
# ==================================

prob_map = probabilities[0, 0].cpu().numpy()

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].imshow(
    masks[0, 0].numpy(),
    cmap="gray",
    vmin=0,
    vmax=1
)
axes[0].set_title("Ground Truth")

im = axes[1].imshow(
    prob_map,
    cmap="viridis",
    vmin=0,
    vmax=1
)
axes[1].set_title("Prediction Probability")

plt.colorbar(im, ax=axes[1])

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()