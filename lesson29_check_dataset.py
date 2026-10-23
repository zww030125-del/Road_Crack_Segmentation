
from pathlib import Path
import torch

# 原来的代码
from lesson07_dataset import (
    CrackDataset as OldCrackDataset,
    train_pairs as old_train_pairs,
    val_pairs as old_val_pairs
)

# 新的代码
from datasets.crack_dataset import (
    CrackDataset as NewCrackDataset
)

from datasets.splits import (
    get_train_val_pairs
)


ROOT = Path(__file__).resolve().parent

new_train_pairs, new_val_pairs = (
    get_train_val_pairs(ROOT / "data")
)


# =====================================
# 1. 检查数据集划分
# =====================================

def pair_names(pairs):
    return [
        (Path(img).name, Path(mask).name)
        for img, mask in pairs
    ]


assert pair_names(old_train_pairs) == pair_names(
    new_train_pairs
), "训练集划分与原来不一致"

assert pair_names(old_val_pairs) == pair_names(
    new_val_pairs
), "验证集划分与原来不一致"

print("训练/验证集划分一致")
print("训练集:", len(new_train_pairs))
print("验证集:", len(new_val_pairs))


# =====================================
# 2. 检查验证样本预处理
# =====================================

old_dataset = OldCrackDataset(
    old_val_pairs,
    training=False
)

new_dataset = NewCrackDataset(
    new_val_pairs,
    training=False
)

for i in range(len(old_dataset)):

    old_image, old_mask = old_dataset[i]
    new_image, new_mask = new_dataset[i]

    assert torch.equal(
        old_image, new_image
    ), f"第{i}张图像预处理不一致"

    assert torch.equal(
        old_mask, new_mask
    ), f"第{i}张Mask预处理不一致"


print("全部验证样本预处理一致")
print("Dataset模块重构成功")
