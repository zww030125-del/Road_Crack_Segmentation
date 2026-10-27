
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp

from models.resnet_unet import ResNetUNet
from datasets.crack_dataset import CrackDataset
from datasets.splits import get_train_val_pairs


# =====================================
# 1. 配置
# =====================================

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "deeplab_experiment"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SAMPLE_INDEX = 0
RESNET_THRESHOLD = 0.73

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


# =====================================
# 2. 加载DeepLabV3+
# =====================================

deeplab = smp.DeepLabV3Plus(
    encoder_name="resnet18",
    encoder_weights=None,
    encoder_output_stride=16,
    decoder_channels=128,
    decoder_atrous_rates=(3, 6, 9),
    in_channels=3,
    classes=1,
    activation=None
).to(DEVICE)

deeplab.load_state_dict(
    torch.load(
        OUT / "best_deeplabv3plus.pth",
        map_location=DEVICE,
        weights_only=True
    )
)

deeplab.eval()


# =====================================
# 3. 加载ResNet18 U-Net
# =====================================

resnet = ResNetUNet(
    pretrained=False,
    freeze_encoder=True
).to(DEVICE)

resnet.load_state_dict(
    torch.load(
        ROOT / "outputs" / "best_resnet18_frozen.pth",
        map_location=DEVICE,
        weights_only=True
    )
)

resnet.eval()


# =====================================
# 4. DeepLab输入预处理
# =====================================

mean = torch.tensor(
    [0.485, 0.456, 0.406],
    device=DEVICE
).view(1, 3, 1, 1)

std = torch.tensor(
    [0.229, 0.224, 0.225],
    device=DEVICE
).view(1, 3, 1, 1)


def preprocess(x):
    return (x - mean) / std


# =====================================
# 5. 验证集阈值搜索
# =====================================

thresholds = torch.arange(
    5, 96, device=DEVICE
).float() / 100

tp = torch.zeros_like(thresholds)
fp = torch.zeros_like(thresholds)
fn = torch.zeros_like(thresholds)

with torch.inference_mode():

    for images, masks in loader:

        images = images.to(DEVICE)
        gt = (masks.to(DEVICE) > 0.5).unsqueeze(0)

        logits = deeplab(preprocess(images))
        prob = torch.sigmoid(logits)

        pred = (
            prob.unsqueeze(0)
            > thresholds[:, None, None, None, None]
        )

        tp += (pred & gt).sum(
            dim=(1, 2, 3, 4)
        )

        fp += (pred & ~gt).sum(
            dim=(1, 2, 3, 4)
        )

        fn += (~pred & gt).sum(
            dim=(1, 2, 3, 4)
        )


precision = tp / (tp + fp).clamp_min(1)
recall = tp / (tp + fn).clamp_min(1)

dice = (
    2 * tp /
    (2 * tp + fp + fn).clamp_min(1)
)

best_idx = dice.argmax().item()

best_threshold = thresholds[best_idx].item()

print("\nDeepLabV3+阈值搜索结果")
print(f"最佳阈值: {best_threshold:.2f}")
print(f"Precision: {precision[best_idx].item():.4f}")
print(f"Recall: {recall[best_idx].item():.4f}")
print(f"最佳Dice: {dice[best_idx].item():.4f}")


# =====================================
# 6. 同一张图像的预测对比
# =====================================

image, mask = dataset[SAMPLE_INDEX]

x = image.unsqueeze(0).to(DEVICE)

with torch.inference_mode():

    deeplab_prob = torch.sigmoid(
        deeplab(preprocess(x))
    )[0, 0].cpu().numpy()

    resnet_prob = torch.sigmoid(
        resnet(x)
    )[0, 0].cpu().numpy()


image_np = image.permute(1, 2, 0).numpy()

gt_np = mask[0].numpy()

resnet_pred = (
    resnet_prob > RESNET_THRESHOLD
)

deeplab_pred = (
    deeplab_prob > best_threshold
)


fig, axes = plt.subplots(
    1, 4,
    figsize=(16, 4),
    constrained_layout=True
)

axes[0].imshow(image_np)
axes[0].set_title("Original")

axes[1].imshow(
    gt_np, cmap="gray", vmin=0, vmax=1
)
axes[1].set_title("Ground Truth")

axes[2].imshow(
    resnet_pred, cmap="gray", vmin=0, vmax=1
)
axes[2].set_title(
    f"ResNet18 U-Net\nThreshold={RESNET_THRESHOLD}"
)

axes[3].imshow(
    deeplab_pred, cmap="gray", vmin=0, vmax=1
)
axes[3].set_title(
    f"DeepLabV3+\nThreshold={best_threshold:.2f}"
)

for ax in axes:
    ax.axis("off")

plt.savefig(
    OUT / "deeplab_vs_resnet.png",
    dpi=160
)

plt.show()

print("\n预测对比完成")
