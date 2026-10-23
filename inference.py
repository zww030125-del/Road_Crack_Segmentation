
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

from unet_model import MiniUNet


# =====================================
# 1. 配置
# =====================================

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "outputs" / "best_unet_bce_dice.pth"

# 可以替换成任何本地图片路径
IMAGE_PATH = ROOT / "data" / "test_img" / "111212-1.jpg"

# 验证集选出的阈值
THRESHOLD = 0.76

OUTPUT_DIR = ROOT / "outputs" / "inference"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("运行设备:", device)


# =====================================
# 2. 加载训练好的模型
# =====================================

model = MiniUNet().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )
)

model.eval()

print("模型加载成功")


# =====================================
# 3. 读取原始图像
# =====================================

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"找不到图片: {IMAGE_PATH}"
    )

image = Image.open(IMAGE_PATH).convert("RGB")

image_np = np.array(image)

height, width = image_np.shape[:2]

print("原图尺寸:", width, "×", height)

# 转换成Tensor
x = torch.from_numpy(
    image_np.astype(np.float32) / 255.0
).permute(2, 0, 1).unsqueeze(0)

x = x.to(device)

# =====================================
# 4. 尺寸填充
# =====================================

# 当前Mini U-Net下采样两次
# 高和宽需要能被4整除
pad_h = (-height) % 4
pad_w = (-width) % 4

if pad_h or pad_w:
    x = F.pad(
        x,
        (0, pad_w, 0, pad_h),
        mode="replicate"
    )


# =====================================
# 5. 模型推理
# =====================================

with torch.inference_mode():

    logits = model(x)

    probabilities = torch.sigmoid(logits)

    # 裁回原始尺寸
    prob_map = probabilities[
        0, 0, :height, :width
    ].cpu().numpy()

# 概率图 -> 二值Mask
pred_mask = prob_map > THRESHOLD

print("预测完成")
print("预测裂纹像素数量:", int(pred_mask.sum()))


# =====================================
# 6. 生成红色叠加图
# =====================================

overlay = image_np.copy()

overlay[pred_mask] = (
    0.5 * image_np[pred_mask]
    + 0.5 * np.array([255, 0, 0])
).astype(np.uint8)


# =====================================
# 7. 保存预测结果
# =====================================

name = IMAGE_PATH.stem

# 二值Mask：0背景，255裂纹
mask_image = Image.fromarray(
    pred_mask.astype(np.uint8) * 255
)

mask_image.save(
    OUTPUT_DIR / f"{name}_mask.png"
)

# 概率图：灰度值0~255
prob_image = Image.fromarray(
    np.clip(prob_map * 255, 0, 255).astype(np.uint8)
)

prob_image.save(
    OUTPUT_DIR / f"{name}_probability.png"
)

Image.fromarray(overlay).save(
    OUTPUT_DIR / f"{name}_overlay.png"
)


# =====================================
# 8. 可视化
# =====================================

fig, axes = plt.subplots(
    1, 4, figsize=(14, 4)
)

axes[0].imshow(image_np)
axes[0].set_title("Original")

axes[1].imshow(
    prob_map, cmap="viridis",
    vmin=0, vmax=1
)
axes[1].set_title("Probability")

axes[2].imshow(
    pred_mask, cmap="gray",
    vmin=0, vmax=1
)
axes[2].set_title("Prediction Mask")

axes[3].imshow(overlay)
axes[3].set_title("Overlay")

for ax in axes:
    ax.axis("off")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / f"{name}_overview.png",
    dpi=150
)

plt.show()

print("结果保存位置:", OUTPUT_DIR)
