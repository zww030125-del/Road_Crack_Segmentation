
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from models.mini_unet import MiniUNet
from models.mini_unet_pp import MiniUNetPP
from datasets.crack_dataset import CrackDataset
from datasets.splits import get_train_val_pairs


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "unetpp_experiment"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

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

# 0.05 ~ 0.95
thresholds = (
    torch.arange(5, 96, device=DEVICE).float() / 100
)

configs = {
    "MiniUNet": (
        MiniUNet,
        OUTPUT_DIR / "best_MiniUNet.pth"
    ),
    "MiniUNetPP": (
        MiniUNetPP,
        OUTPUT_DIR / "best_MiniUNetPP.pth"
    )
}


@torch.inference_mode()
def evaluate(model):

    model.eval()

    tp = torch.zeros_like(thresholds)
    fp = torch.zeros_like(thresholds)
    fn = torch.zeros_like(thresholds)

    for images, masks in loader:

        images = images.to(DEVICE)
        gt = (masks.to(DEVICE) > 0.5).unsqueeze(0)

        prob = torch.sigmoid(model(images))

        # 同一批概率同时比较91个阈值
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

    dice = (
        2 * tp /
        (2 * tp + fp + fn).clamp_min(1)
    )

    precision = tp / (tp + fp).clamp_min(1)
    recall = tp / (tp + fn).clamp_min(1)

    best_idx = dice.argmax().item()

    return {
        "threshold": thresholds[best_idx].item(),
        "dice": dice[best_idx].item(),
        "precision": precision[best_idx].item(),
        "recall": recall[best_idx].item(),
        "dice_at_05": dice[45].item()
    }


models = {}
results = {}

for name, (model_class, path) in configs.items():

    model = model_class().to(DEVICE)

    model.load_state_dict(
        torch.load(
            path,
            map_location=DEVICE,
            weights_only=True
        )
    )

    model.eval()
    models[name] = model

    result = evaluate(model)
    results[name] = result

    print(f"\n模型: {name}")
    print(f"阈值0.5的Dice: {result['dice_at_05']:.4f}")
    print(f"最佳阈值: {result['threshold']:.2f}")
    print(f"Precision: {result['precision']:.4f}")
    print(f"Recall: {result['recall']:.4f}")
    print(f"最佳验证Dice: {result['dice']:.4f}")


# =====================================
# 同一张验证图像的预测比较
# =====================================

SAMPLE_INDEX = 0

image, mask = dataset[SAMPLE_INDEX]

image_np = image.permute(1, 2, 0).numpy()
mask_np = mask[0].numpy()

fig, axes = plt.subplots(
    1, 4,
    figsize=(15, 4),
    constrained_layout=True
)

axes[0].imshow(image_np)
axes[0].set_title("Original")

axes[1].imshow(mask_np, cmap="gray", vmin=0, vmax=1)
axes[1].set_title("Ground Truth")

for i, name in enumerate(configs, start=2):

    model = models[name]
    threshold = results[name]["threshold"]

    with torch.inference_mode():
        prob = torch.sigmoid(
            model(image.unsqueeze(0).to(DEVICE))
        )[0, 0].cpu().numpy()

    pred = prob > threshold

    axes[i].imshow(pred, cmap="gray", vmin=0, vmax=1)
    axes[i].set_title(
        f"{name}\nThreshold={threshold:.2f}"
    )

for ax in axes:
    ax.axis("off")

plt.savefig(
    OUTPUT_DIR / "threshold_comparison.png",
    dpi=160
)

plt.show()
