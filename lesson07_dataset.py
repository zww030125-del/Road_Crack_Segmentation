
from pathlib import Path
import random

import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader

# -------------------------------
# 1. 基本配置
# -------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

CROP_SIZE = 256
BATCH_SIZE = 2

# -------------------------------
# 2. 配对原图与Mask
# -------------------------------
image_dir = DATA_DIR / "train_img"
mask_dir = DATA_DIR / "train_lab"

extensions = {".jpg", ".jpeg", ".png", ".bmp"}

image_paths = sorted(
    p for p in image_dir.iterdir()
    if p.suffix.lower() in extensions
)

mask_map = {
    p.stem: p
    for p in mask_dir.iterdir()
    if p.suffix.lower() in extensions
}

pairs = []

for image_path in image_paths:
    if image_path.stem not in mask_map:
        raise ValueError(f"缺少标签: {image_path.name}")

    pairs.append((image_path, mask_map[image_path.stem]))

print("图像总数:", len(pairs))

# -------------------------------
# 3. 划分训练集与验证集
# -------------------------------
random.Random(42).shuffle(pairs)

split_index = int(len(pairs) * 0.8)

train_pairs = pairs[:split_index]
val_pairs = pairs[split_index:]

print("训练集:", len(train_pairs))
print("验证集:", len(val_pairs))

# -------------------------------
# 4. 自定义Dataset
# -------------------------------
class CrackDataset(Dataset):

    def __init__(
            self,
            pairs,
            training=True,
            augment=False,
            thin_focus_prob=0.0
    ):
        self.pairs = pairs
        self.training = training
        self.augment = augment
        self.thin_focus_prob = thin_focus_prob

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):

        image_path, mask_path = self.pairs[index]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if image.size != mask.size:
            raise ValueError("图像与Mask尺寸不一致")

        width, height = image.size

        if width < CROP_SIZE or height < CROP_SIZE:
            raise ValueError(
                f"图像过小: {image_path.name}, {image.size}"
            )

        # 训练集随机裁剪，验证集中心裁剪

        # 先默认使用普通随机裁剪
        if self.training:
            left = random.randint(0, width - CROP_SIZE)
            top = random.randint(0, height - CROP_SIZE)

            # 按设定概率使用细裂纹引导裁剪
            if random.random() < self.thin_focus_prob:

                mask_array = np.asarray(mask)

                # 计算7×7邻域的平均前景密度
                local_mean = np.asarray(
                    mask.filter(ImageFilter.BoxBlur(3))
                )

                # 寻找局部前景较稀疏的裂纹像素
                candidate = (
                    (mask_array > 127)
                    & (local_mean < 90)
                )

                ys, xs = np.where(candidate)

                if len(xs) > 0:
                    idx = random.randrange(len(xs))

                    cx = int(xs[idx])
                    cy = int(ys[idx])

                    # 让候选像素位于裁剪区域内部
                    offset_x = random.randint(32, 224)
                    offset_y = random.randint(32, 224)

                    left = max(
                        0,
                        min(cx - offset_x, width - CROP_SIZE)
                    )

                    top = max(
                        0,
                        min(cy - offset_y, height - CROP_SIZE)
                    )

        else:
            # 验证集仍使用中心裁剪
            left = (width - CROP_SIZE) // 2
            top = (height - CROP_SIZE) // 2


        box = (
            left,
            top,
            left + CROP_SIZE,
            top + CROP_SIZE
        )

        image = image.crop(box)
        mask = mask.crop(box)
        # ==================================
        # 数据增强：仅用于训练集
        # ==================================

        if self.training and self.augment:

            # 1. 50%概率水平翻转
            if random.random() < 0.5:
                image = image.transpose(
                    Image.Transpose.FLIP_LEFT_RIGHT
                )
                mask = mask.transpose(
                    Image.Transpose.FLIP_LEFT_RIGHT
                )

            # 2. 50%概率垂直翻转
            if random.random() < 0.5:
                image = image.transpose(
                    Image.Transpose.FLIP_TOP_BOTTOM
                )
                mask = mask.transpose(
                    Image.Transpose.FLIP_TOP_BOTTOM
                )

            # 3. 随机旋转0、90、180或270度
            rotation = random.choice([
                None,
                Image.Transpose.ROTATE_90,
                Image.Transpose.ROTATE_180,
                Image.Transpose.ROTATE_270
            ])

            if rotation is not None:
                image = image.transpose(rotation)
                mask = mask.transpose(rotation)

        # RGB图像转换为Tensor，数值0~1
        image_np = np.array(image, dtype=np.float32) / 255.0

        image_tensor = torch.from_numpy(
            image_np
        ).permute(2, 0, 1)

        # 二值Mask，0为背景，1为裂纹
        mask_np = (
            np.array(mask) > 127
        ).astype(np.float32)

        mask_tensor = torch.from_numpy(
            mask_np
        ).unsqueeze(0)

        return image_tensor, mask_tensor

# -------------------------------
# 5. 创建DataLoader
# -------------------------------
train_dataset = CrackDataset(
    train_pairs,
    training=True,
    augment=True
)

val_dataset = CrackDataset(
    val_pairs,
    training=False,
    augment=False
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# -------------------------------
# 6. 取出一个Batch
# -------------------------------
if __name__ == "__main__":
    images, masks = next(iter(train_loader))

    print("\n一个Batch的形状:")
    print("Images:", images.shape)
    print("Masks:", masks.shape)
    print("Mask像素值:", torch.unique(masks))

    # -------------------------------
    # 7. 检查GPU传输
    # -------------------------------
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    images_gpu = images.to(device)

    print("运行设备:", images_gpu.device)

    # -------------------------------
    # 8. 可视化一个Batch
    # -------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    for i in range(2):

        image = images[i].permute(1, 2, 0).numpy()
        mask = masks[i, 0].numpy()

        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f"Image {i + 1}")

        axes[i, 1].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[i, 1].set_title(f"Mask {i + 1}")

        axes[i, 0].axis("off")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.show()
