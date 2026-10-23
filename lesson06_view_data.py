
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 1. 定位数据集
project_dir = Path(__file__).resolve().parent
data_dir = project_dir / "data"

image_dir = data_dir / "train_img"
mask_dir = data_dir / "train_lab"

# 2. 检查文件夹
if not image_dir.is_dir() or not mask_dir.is_dir():
    raise FileNotFoundError(
        "找不到train_img或train_lab文件夹，"
        "请检查data文件夹的位置。"
    )

# 3. 列出图像文件
extensions = {".jpg", ".jpeg", ".png", ".bmp"}

images = sorted(
    p for p in image_dir.iterdir()
    if p.suffix.lower() in extensions
)

masks = {
    p.stem: p
    for p in mask_dir.iterdir()
    if p.suffix.lower() in extensions
}

print("训练原图数量：", len(images))
print("训练标签数量：", len(masks))

# 4. 根据文件名匹配图像和标签
pairs = [
    (p, masks[p.stem])
    for p in images
    if p.stem in masks
]

print("成功匹配的图像对数：", len(pairs))

if not pairs:
    raise RuntimeError(
        "没有匹配到图像与Mask，请检查文件名。"
    )

# 5. 读取第一组图像
image_path, mask_path = pairs[0]

image = Image.open(image_path).convert("RGB")
mask = Image.open(mask_path).convert("L")

image_array = np.array(image)
mask_array = np.array(mask)

print("原图文件：", image_path.name)
print("标签文件：", mask_path.name)
print("原图尺寸：", image_array.shape)
print("标签尺寸：", mask_array.shape)
print("标签像素值：", np.unique(mask_array)[:20])

# 检查尺寸是否一致
if image_array.shape[:2] != mask_array.shape:
    raise ValueError("原图与Mask尺寸不一致！")

# 6. 转换成二值标签：0背景，1裂纹
binary_mask = (mask_array > 0).astype(np.uint8)

# 7. 生成叠加图
overlay = image_array.copy()

crack_pixels = binary_mask == 1
overlay[crack_pixels] = (
    0.5 * overlay[crack_pixels]
    + 0.5 * np.array([255, 0, 0])
).astype(np.uint8)

# 8. 可视化
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(image_array)
axes[0].set_title("Original Image")

axes[1].imshow(binary_mask, cmap="gray", vmin=0, vmax=1)
axes[1].set_title("Ground Truth Mask")

axes[2].imshow(overlay)
axes[2].set_title("Overlay (Red = Crack)")

for ax in axes:
    ax.axis("off")

plt.tight_layout()

# 保存结果
output_dir = project_dir / "outputs"
output_dir.mkdir(exist_ok=True)

plt.savefig(output_dir / "sample_preview.png", dpi=150)
plt.show()
