
from pathlib import Path
import csv

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

from unet_model import MiniUNet

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUT = ROOT / "outputs"

THRESHOLD = 0.76

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ==================================
# 1. 读取测试结果，寻找困难样本
# ==================================

with open(
    OUTPUT / "test_results.csv",
    encoding="utf-8-sig"
) as f:
    records = list(csv.DictReader(f))

records.sort(key=lambda r: float(r["dice"]))

# 最差3张 + 最好1张
selected = records[:3] + records[-1:]

# ==================================
# 2. 加载模型
# ==================================

model = MiniUNet().to(device)

model.load_state_dict(
    torch.load(
        OUTPUT / "best_unet_bce_dice.pth",
        map_location=device,
        weights_only=True
    )
)

model.eval()

# ==================================
# 3. 整张图像预测
# ==================================

@torch.inference_mode()
def predict(image):

    h, w = image.shape[:2]

    x = torch.from_numpy(
        image.astype(np.float32) / 255.0
    ).permute(2, 0, 1).unsqueeze(0)

    x = x.to(device)

    # 保证输入尺寸可被4整除
    pad_h = (-h) % 4
    pad_w = (-w) % 4

    if pad_h or pad_w:
        x = F.pad(
            x,
            (0, pad_w, 0, pad_h),
            mode="replicate"
        )

    logits = model(x)

    return torch.sigmoid(logits)[
        0, 0, :h, :w
    ].cpu().numpy()


# ==================================
# 4. 绘制困难样本
# ==================================

fig, axes = plt.subplots(
    len(selected), 5,
    figsize=(16, 3.5 * len(selected))
)

for row, record in enumerate(selected):

    name = record["image"]

    image_path = DATA / "test_img" / name

    mask_path = (
        DATA / "test_lab" /
        (Path(name).stem + ".png")
    )

    image = np.array(
        Image.open(image_path).convert("RGB")
    )

    gt = np.array(
        Image.open(mask_path).convert("L")
    ) > 127

    prob = predict(image)
    pred = prob > THRESHOLD

    fp = pred & ~gt
    fn = ~pred & gt

    error_map = image.copy()
    error_map[fp] = [255, 0, 0]
    error_map[fn] = [0, 180, 255]

    axes[row, 0].imshow(image)
    axes[row, 1].imshow(gt, cmap="gray", vmin=0, vmax=1)
    axes[row, 2].imshow(pred, cmap="gray", vmin=0, vmax=1)
    axes[row, 3].imshow(
        prob, cmap="viridis", vmin=0, vmax=1
    )
    axes[row, 4].imshow(error_map)

    axes[row, 0].set_ylabel(
        f"{name}\nDice={float(record['dice']):.3f}",
        fontsize=10
    )

    print(
        f"{name}: "
        f"真实裂纹像素={gt.sum()}, "
        f"预测裂纹像素={pred.sum()}"
    )

titles = [
    "Original",
    "Ground Truth",
    "Prediction",
    "Probability",
    "Red FP / Cyan FN"
]

for col, title in enumerate(titles):
    axes[0, col].set_title(title)

for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()

plt.savefig(
    OUTPUT / "error_analysis.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()
