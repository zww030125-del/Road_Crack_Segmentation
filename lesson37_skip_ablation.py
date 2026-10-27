
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from models.resnet_unet import ResNetUNet
from datasets.crack_dataset import CrackDataset
from datasets.splits import get_train_val_pairs


ROOT = Path(__file__).resolve().parent

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

THRESHOLD = 0.73
SAMPLE_INDEX = 0

OUTPUT_DIR = ROOT / "outputs" / "skip_ablation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================
# 1. 数据与模型
# =====================================

_, val_pairs = get_train_val_pairs(ROOT / "data")

dataset = CrackDataset(
    val_pairs,
    training=False,
    augment=False
)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0
)

model = ResNetUNet(
    pretrained=False,
    freeze_encoder=True
).to(DEVICE)

model.load_state_dict(
    torch.load(
        ROOT / "outputs" / "best_resnet18_frozen.pth",
        map_location=DEVICE,
        weights_only=True
    )
)

model.eval()


# =====================================
# 2. 定义消融条件
# =====================================

# 每个解码模块的输入：
# [上采样特征, 编码器跳跃特征]
# 下面记录最后拼接的跳跃特征通道数

CASES = [
    ("Baseline", None, 0),
    ("No e3 skip", model.dec4, 256),
    ("No e2 skip", model.dec3, 128),
    ("No e1 skip", model.dec2, 64),
    ("No e0 skip", model.dec1, 64),
    ("No RGB skip", model.dec0, 3)
]


def disable_skip(skip_channels):
    """
    将解码模块输入中来自跳跃连接的部分置零。
    不影响编码器继续提取特征。
    """

    def hook(module, inputs):
        x = inputs[0].clone()

        # 拼接结果的最后几个通道属于skip
        x[:, -skip_channels:, :, :] = 0

        return (x,)

    return hook


# =====================================
# 3. 评价函数
# =====================================

def evaluate_case(name, module, skip_channels):

    handle = None

    if module is not None:
        handle = module.register_forward_pre_hook(
            disable_skip(skip_channels)
        )

    try:
        tp = fp = fn = 0

        with torch.inference_mode():
            for images, masks in loader:

                images = images.to(DEVICE)
                masks = masks.to(DEVICE)

                prob = torch.sigmoid(model(images))

                pred = prob > THRESHOLD
                gt = masks > 0.5

                tp += (pred & gt).sum().item()
                fp += (pred & ~gt).sum().item()
                fn += (~pred & gt).sum().item()

            # 获取示例图像对应的概率图
            image, _ = dataset[SAMPLE_INDEX]

            sample_prob = torch.sigmoid(
                model(image.unsqueeze(0).to(DEVICE))
            )[0, 0].cpu().numpy()

        dice = (
            2 * tp / (2 * tp + fp + fn)
            if (2 * tp + fp + fn) > 0 else 1.0
        )

        print(f"{name:15s} | Val Dice: {dice:.4f}")

        return sample_prob

    finally:
        if handle is not None:
            handle.remove()


# =====================================
# 4. 执行全部条件
# =====================================

results = {}

for name, module, channels in CASES:
    results[name] = evaluate_case(
        name, module, channels
    )


# =====================================
# 5. 可视化
# =====================================

image, mask = dataset[SAMPLE_INDEX]

image_np = image.permute(1, 2, 0).numpy()
mask_np = mask[0].numpy()

fig, axes = plt.subplots(
    2, 4, figsize=(15, 8),
    constrained_layout=True
)

axes = axes.flatten()

axes[0].imshow(image_np)
axes[0].set_title("Original")

axes[1].imshow(
    mask_np, cmap="gray", vmin=0, vmax=1
)
axes[1].set_title("Ground Truth")

for index, (name, prob) in enumerate(
    results.items(), start=2
):

    axes[index].imshow(
        prob,
        cmap="viridis",
        vmin=0,
        vmax=1
    )

    axes[index].set_title(name)

for ax in axes:
    ax.axis("off")

plt.savefig(
    OUTPUT_DIR / "skip_ablation.png",
    dpi=160
)

plt.show()

print("\n消融实验完成")
