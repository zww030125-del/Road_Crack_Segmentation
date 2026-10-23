
import random

import numpy as np
import torch

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


# =====================================
# 1. 基本配置
# =====================================

CROP_SIZE = 256


# =====================================
# 2. 裂纹分割数据集
# =====================================

class CrackDataset(Dataset):

    def __init__(
        self,
        pairs,
        training=True,
        augment=False,
        crop_size=CROP_SIZE
    ):
        """
        pairs:
            [(image_path, mask_path), ...]

        training:
            True  -> 随机裁剪
            False -> 中心裁剪

        augment:
            是否使用随机翻转与90度旋转

        crop_size:
            图像裁剪尺寸
        """

        self.pairs = pairs
        self.training = training
        self.augment = augment
        self.crop_size = crop_size

    # =================================
    # 3. 数据集长度
    # =================================

    def __len__(self):
        return len(self.pairs)

    # =================================
    # 4. 图像和Mask同步增强
    # =================================

    def apply_augmentation(self, image, mask):

        # 随机水平翻转
        if random.random() < 0.5:
            image = np.fliplr(image)
            mask = np.fliplr(mask)

        # 随机垂直翻转
        if random.random() < 0.5:
            image = np.flipud(image)
            mask = np.flipud(mask)

        # 随机旋转0、90、180或270度
        k = random.randint(0, 3)

        image = np.rot90(image, k)
        mask = np.rot90(mask, k)

        return image, mask

    # =================================
    # 5. 读取一个样本
    # =================================

    def __getitem__(self, idx):

        image_path, mask_path = self.pairs[idx]

        # 读取RGB原始图像
        with Image.open(image_path) as img:
            image = np.array(
                img.convert("RGB")
            )

        # 读取灰度Mask
        with Image.open(mask_path) as mask_img:
            mask = np.array(
                mask_img.convert("L")
            )

        h, w = image.shape[:2]
        size = self.crop_size

        if mask.shape != (h, w):
            raise ValueError(
                f"图像与Mask尺寸不一致: {image_path}"
            )

        if h < size or w < size:
            raise ValueError(
                f"图像尺寸小于裁剪尺寸: {image_path}"
            )

        # =============================
        # 6. 确定裁剪位置
        # =============================

        if self.training:

            # 训练集：随机裁剪
            top = random.randint(0, h - size)
            left = random.randint(0, w - size)

        else:

            # 验证集：中心裁剪
            top = (h - size) // 2
            left = (w - size) // 2

        # 图像和Mask使用完全相同的裁剪区域
        image = image[
            top:top + size,
            left:left + size
        ]

        mask = mask[
            top:top + size,
            left:left + size
        ]

        # =============================
        # 7. 数据增强
        # =============================

        if self.training and self.augment:

            image, mask = self.apply_augmentation(
                image, mask
            )

        # =============================
        # 8. 转换为Tensor
        # =============================

        # 保证数组内存连续
        image = np.ascontiguousarray(image)
        mask = np.ascontiguousarray(mask)

        # RGB图像: HWC -> CHW
        # 像素值转换到0~1
        image_tensor = torch.from_numpy(
            image.astype(np.float32) / 255.0
        ).permute(2, 0, 1)

        # Mask二值化：背景0、裂纹1
        mask_binary = (
            mask > 127
        ).astype(np.float32)

        mask_tensor = torch.from_numpy(
            mask_binary
        ).unsqueeze(0)

        return image_tensor, mask_tensor
