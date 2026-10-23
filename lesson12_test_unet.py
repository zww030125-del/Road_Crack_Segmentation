
from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image

from unet_model import MiniUNet


# =====================================
# 1. 基本配置
# =====================================

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(exist_ok=True)

MODEL_PATH = OUTPUT / "best_unet_bce_dice.pth"

# 使用验证集选择的阈值，不在测试集上调参
THRESHOLD = 0.76

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("运行设备:", device)
print("测试阈值:", THRESHOLD)


# =====================================
# 2. 读取并配对测试集
# =====================================

image_dir = DATA / "test_img"
mask_dir = DATA / "test_lab"

extensions = {".jpg", ".jpeg", ".png", ".bmp"}

image_paths = sorted(
    p for p in image_dir.iterdir()
    if p.suffix.lower() in extensions
)

mask_paths = {
    p.stem: p
    for p in mask_dir.iterdir()
    if p.suffix.lower() in extensions
}

pairs = []

for image_path in image_paths:
    if image_path.stem not in mask_paths:
        raise ValueError(
            f"找不到对应标签: {image_path.name}"
        )

    pairs.append(
        (image_path, mask_paths[image_path.stem])
    )

if not pairs:
    raise RuntimeError("测试集为空，请检查文件夹")

print("测试图像数量:", len(pairs))


# =====================================
# 3. 加载已经训练好的模型
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
# 4. 整张图像预测
# =====================================

@torch.inference_mode()
def predict_full_image(image_np):

    h, w = image_np.shape[:2]

    # HWC -> NCHW
    x = torch.from_numpy(
        image_np.astype(np.float32) / 255.0
    ).permute(2, 0, 1).unsqueeze(0)

    x = x.to(device)

    # U-Net下采样两次，尺寸需要是4的倍数
    pad_h = (-h) % 4
    pad_w = (-w) % 4

    if pad_h > 0 or pad_w > 0:
        x = F.pad(
            x,
            (0, pad_w, 0, pad_h),
            mode="replicate"
        )

    logits = model(x)

    probabilities = torch.sigmoid(logits)

    # 去掉临时填充的边缘
    prob_map = probabilities[
        0, 0, :h, :w
    ].cpu().numpy()

    return prob_map


# =====================================
# 5. 计算评价指标
# =====================================

def calculate_metrics(tp, fp, fn):

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else (1.0 if fn == 0 else 0.0)
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0 else 1.0
    )

    dice = (
        2 * tp / (2 * tp + fp + fn)
        if 2 * tp + fp + fn > 0 else 1.0
    )

    iou = (
        tp / (tp + fp + fn)
        if tp + fp + fn > 0 else 1.0
    )

    return precision, recall, dice, iou


# =====================================
# 6. 遍历整个测试集
# =====================================

total_tp = 0
total_fp = 0
total_fn = 0

records = []
preview = None

for index, (image_path, mask_path) in enumerate(pairs):

    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    if image.size != mask.size:
        raise ValueError(
            f"原图与标签尺寸不一致: {image_path.name}"
        )

    image_np = np.asarray(image)
    gt = np.asarray(mask) > 127

    # 模型预测
    prob_map = predict_full_image(image_np)
    pred = prob_map > THRESHOLD

    # 统计像素分类结果
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())

    total_tp += tp
    total_fp += fp
    total_fn += fn

    precision, recall, dice, iou = calculate_metrics(
        tp, fp, fn
    )

    records.append({
        "image": image_path.name,
        "precision": precision,
        "recall": recall,
        "dice": dice,
        "iou": iou
    })

    # 保存第一张图用于可视化
    if preview is None:
        preview = (
            image_np.copy(),
            gt.copy(),
            pred.copy(),
            image_path.name
        )

    if (index + 1) % 20 == 0:
        print(f"已测试: {index + 1}/{len(pairs)}")


# =====================================
# 7. 整个测试集评价结果
# =====================================

precision, recall, dice, iou = calculate_metrics(
    total_tp, total_fp, total_fn
)

print("\n===== 独立测试集结果 =====")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"Dice:      {dice:.4f}")
print(f"IoU:       {iou:.4f}")

# 同时计算逐图平均Dice
mean_dice = np.mean([r["dice"] for r in records])

print(f"Mean Image Dice: {mean_dice:.4f}")

# 保存每张图像的指标
with open(
    OUTPUT / "test_results.csv",
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "image", "precision",
            "recall", "dice", "iou"
        ]
    )

    writer.writeheader()
    writer.writerows(records)


# =====================================
# 8. 可视化：原图、标签、预测、错误
# =====================================

image_np, gt, pred, image_name = preview

# FP：误检，FN：漏检
fp_mask = np.logical_and(pred, ~gt)
fn_mask = np.logical_and(~pred, gt)

error_overlay = image_np.copy()

error_overlay[fp_mask] = [255, 0, 0]
error_overlay[fn_mask] = [0, 180, 255]

fig, axes = plt.subplots(
    1, 4, figsize=(15, 4)
)

axes[0].imshow(image_np)
axes[0].set_title("Original")

axes[1].imshow(gt, cmap="gray", vmin=0, vmax=1)
axes[1].set_title("Ground Truth")

axes[2].imshow(pred, cmap="gray", vmin=0, vmax=1)
axes[2].set_title("U-Net Prediction")

axes[3].imshow(error_overlay)
axes[3].set_title("Red: FP | Cyan: FN")

for ax in axes:
    ax.axis("off")

fig.suptitle(image_name)
plt.tight_layout()

plt.savefig(
    OUTPUT / "test_prediction.png",
    dpi=150
)

plt.show()

print("\n测试完成，结果已保存")
