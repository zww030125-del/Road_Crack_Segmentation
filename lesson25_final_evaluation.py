
from pathlib import Path
import csv

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from unet_model import MiniUNet
from resnet_unet import ResNetUNet


# =====================================
# 1. 路径配置
# =====================================

ROOT = Path(__file__).resolve().parent

TEST_IMG = ROOT / "data" / "test_img"
TEST_LAB = ROOT / "data" / "test_lab"

OUTPUT_DIR = ROOT / "outputs" / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("设备:", device)


# =====================================
# 2. 模型配置
# =====================================

MODELS = [
    {
        "name": "MiniUNet",
        "class": MiniUNet,
        "weight": ROOT / "outputs" / "best_unet_bce_dice.pth",
        "threshold": 0.76
    },
    {
        "name": "ResNet18UNet",
        "class": lambda: ResNetUNet(
            pretrained=False,
            freeze_encoder=True
        ),
        "weight": ROOT / "outputs" / "best_resnet18_frozen.pth",
        "threshold": 0.73
    }
]


# =====================================
# 3. 匹配测试图像和Mask
# =====================================

test_pairs = []

for img_path in sorted(TEST_IMG.glob("*.jpg")):

    mask_path = TEST_LAB / (img_path.stem + ".png")

    if not mask_path.exists():
        raise FileNotFoundError(
            f"缺少对应标签: {mask_path}"
        )

    test_pairs.append((img_path, mask_path))

print("测试图像数量:", len(test_pairs))

assert len(test_pairs) == 237, (
    f"预期237张，实际找到{len(test_pairs)}张"
)


# =====================================
# 4. 计算评价指标
# =====================================

def calculate_metrics(tp, fp, fn):

    precision = (
        tp / (tp + fp)
        if tp + fp > 0 else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0 else 0.0
    )

    denominator = 2 * tp + fp + fn

    dice = (
        2 * tp / denominator
        if denominator > 0 else 1.0
    )

    iou = (
        tp / (tp + fp + fn)
        if tp + fp + fn > 0 else 1.0
    )

    return precision, recall, dice, iou


# =====================================
# 5. 单张图像推理
# =====================================

@torch.inference_mode()
def predict_image(model, image_np, threshold):

    h, w = image_np.shape[:2]

    x = torch.from_numpy(
        image_np.astype(np.float32) / 255.0
    ).permute(2, 0, 1).unsqueeze(0)

    x = x.to(device)

    # 统一填充至32的倍数
    pad_h = (-h) % 32
    pad_w = (-w) % 32

    if pad_h or pad_w:
        x = F.pad(
            x,
            (0, pad_w, 0, pad_h),
            mode="replicate"
        )

    logits = model(x)

    probabilities = torch.sigmoid(logits)

    probabilities = probabilities[
        0, 0, :h, :w
    ]

    prediction = probabilities > threshold

    return prediction.cpu().numpy()


# =====================================
# 6. 评价单个模型
# =====================================

def evaluate_model(config):

    name = config["name"]
    threshold = config["threshold"]

    print(f"\n开始评价: {name}")

    model = config["class"]().to(device)

    model.load_state_dict(
        torch.load(
            config["weight"],
            map_location=device,
            weights_only=True
        )
    )

    model.eval()

    total_tp = 0
    total_fp = 0
    total_fn = 0

    results = []

    for index, (img_path, mask_path) in enumerate(
        test_pairs, 1
    ):

        # 读取原始RGB图像
        with Image.open(img_path) as img:
            image_np = np.array(img.convert("RGB"))

        # 读取真实Mask
        with Image.open(mask_path) as mask_img:
            mask_np = np.array(
                mask_img.convert("L")
            )

        gt = mask_np > 127

        assert gt.shape == image_np.shape[:2], (
            f"图像与标签尺寸不一致: {img_path.name}"
        )

        # 模型预测
        pred = predict_image(
            model, image_np, threshold
        )

        # 统计TP、FP、FN
        tp = int(np.logical_and(pred, gt).sum())
        fp = int(np.logical_and(pred, ~gt).sum())
        fn = int(np.logical_and(~pred, gt).sum())

        total_tp += tp
        total_fp += fp
        total_fn += fn

        precision, recall, dice, iou = (
            calculate_metrics(tp, fp, fn)
        )

        results.append({
            "image": img_path.name,
            "precision": precision,
            "recall": recall,
            "dice": dice,
            "iou": iou,
            "tp": tp,
            "fp": fp,
            "fn": fn
        })

        if index % 50 == 0:
            print(
                f"已完成 {index}/{len(test_pairs)}"
            )

    # 整个测试集累计像素指标
    precision, recall, dice, iou = (
        calculate_metrics(
            total_tp, total_fp, total_fn
        )
    )

    mean_dice = float(
        np.mean([r["dice"] for r in results])
    )

    # 保存每张图像的评价结果
    csv_path = OUTPUT_DIR / f"{name}_test_results.csv"

    with open(
        csv_path, "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image", "precision", "recall",
                "dice", "iou", "tp", "fp", "fn"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 45)
    print("模型:", name)
    print("阈值:", threshold)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Micro Dice: {dice:.4f}")
    print(f"IoU:       {iou:.4f}")
    print(f"Mean Image Dice: {mean_dice:.4f}")
    print("结果文件:", csv_path)

    return {
        "model": name,
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "dice": dice,
        "iou": iou,
        "mean_image_dice": mean_dice
    }


# =====================================
# 7. 评价全部模型
# =====================================

all_results = []

for config in MODELS:
    result = evaluate_model(config)
    all_results.append(result)


# =====================================
# 8. 保存模型对比表
# =====================================

summary_path = OUTPUT_DIR / "model_comparison.csv"

with open(
    summary_path, "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(all_results[0].keys())
    )

    writer.writeheader()
    writer.writerows(all_results)

print("\n全部评价完成")
print("模型对比表:", summary_path)
