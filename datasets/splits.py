
from pathlib import Path
import random


def get_train_val_pairs(
    data_root,
    train_ratio=0.8,
    seed=42
):
    data_root = Path(data_root)

    img_dir = data_root / "train_img"
    mask_dir = data_root / "train_lab"

    images = sorted(img_dir.glob("*.jpg"))

    if not images:
        raise FileNotFoundError(
            f"没有找到训练图像: {img_dir}"
        )

    pairs = []

    for img_path in images:

        mask_path = mask_dir / (
            img_path.stem + ".png"
        )

        if not mask_path.exists():
            raise FileNotFoundError(
                f"缺少Mask: {mask_path}"
            )

        pairs.append((img_path, mask_path))

    # 固定随机种子，保证可复现
    random.Random(seed).shuffle(pairs)

    split_idx = int(
        len(pairs) * train_ratio
    )

    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    return train_pairs, val_pairs
