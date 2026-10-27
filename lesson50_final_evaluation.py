
from pathlib import Path
import csv
import gc

import numpy as np
import torch
from PIL import Image

from models.resnet_unet import ResNetUNet
from datasets.splits import get_train_val_pairs


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "final_review"

MASK_OUT = OUT / "test_masks"
OVERLAY_OUT = OUT / "test_overlays"

for folder in [OUT, MASK_OUT, OVERLAY_OUT]:
    folder.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

THRESHOLDS = np.arange(5, 96) / 100

CHECKPOINTS = {
    "ResNet18_Frozen":
        ROOT / "outputs" / "best_resnet18_frozen.pth",

    "ResNet18_Finetuned":
        ROOT / "outputs" / "best_resnet18_finetuned.pth",

    "Hard_Sampling_43":
        ROOT / "outputs" / "lesson47"
        / "seed_43" / "best_hard_sampling.pth",

    "clDice_42":
        ROOT / "outputs" / "lesson48"
        / "best_cldice.pth",

    "Full_Image":
        ROOT / "outputs" / "final_full_image"
        / "best_full_image.pth"
}


def load_pair(image_path, mask_path):
    with Image.open(image_path) as im:
        image = np.asarray(
            im.convert("RGB"), dtype=np.uint8
        )

    with Image.open(mask_path) as im:
        mask = np.asarray(
            im.convert("L")
        ) > 127

    if image.shape[:2] != mask.shape:
        raise ValueError(
            f"图像和标签尺寸不一致: {image_path}"
        )

    return image, mask


def create_model(checkpoint):
    model = ResNetUNet(
        pretrained=False,
        freeze_encoder=True
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(
            checkpoint,
            map_location=DEVICE,
            weights_only=True
        )
    )

    model.eval()
    return model


@torch.inference_mode()
def predict(model, image):
    x = torch.from_numpy(
        image.transpose(2, 0, 1).copy()
    ).float().div(255.0)

    x = x.unsqueeze(0).to(DEVICE)

    prob = torch.sigmoid(model(x))

    return prob[0, 0].cpu().numpy()


def calculate_metrics(tp, fp, fn):
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    denom = 2 * tp + fp + fn

    dice = (
        2 * tp / denom if denom > 0 else 1.0
    )

    iou = (
        tp / max(tp + fp + fn, 1)
        if denom > 0 else 1.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "dice": dice,
        "iou": iou
    }


def save_csv(path, rows):
    if not rows:
        return

    with open(
        path, "w", newline="",
        encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)


# ===========================================
# 1. 完整验证集：统一模型选择
# ===========================================

_, val_pairs = get_train_val_pairs(
    ROOT / "data", seed=42
)

selection_rows = []

print("设备:", DEVICE)
print("完整验证图像:", len(val_pairs))

for name, checkpoint in CHECKPOINTS.items():

    if not checkpoint.exists():
        print("跳过缺失权重:", checkpoint)
        continue

    print("\n验证模型:", name)

    model = create_model(checkpoint)

    all_probs = []
    all_masks = []

    for image_path, mask_path in val_pairs:
        image, gt = load_pair(
            image_path, mask_path
        )

        prob = predict(model, image)

        all_probs.append(prob.ravel())
        all_masks.append(gt.ravel())

    probabilities = np.concatenate(all_probs)
    ground_truth = np.concatenate(all_masks)

    best = None

    for threshold in THRESHOLDS:

        pred = probabilities > threshold

        tp = int(np.count_nonzero(
            pred & ground_truth
        ))

        fp = int(np.count_nonzero(
            pred & ~ground_truth
        ))

        fn = int(np.count_nonzero(
            ~pred & ground_truth
        ))

        metrics = calculate_metrics(
            tp, fp, fn
        )

        candidate = {
            "model": name,
            "threshold": float(threshold),
            **metrics
        }

        if (
            best is None
            or candidate["dice"] > best["dice"]
        ):
            best = candidate

    selection_rows.append(best)

    print(
        f"Threshold={best['threshold']:.2f} | "
        f"Precision={best['precision']:.4f} | "
        f"Recall={best['recall']:.4f} | "
        f"Dice={best['dice']:.4f} | "
        f"IoU={best['iou']:.4f}"
    )

    del model, probabilities, ground_truth
    del all_probs, all_masks
    gc.collect()

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


if not selection_rows:
    raise RuntimeError("没有找到任何可评价的模型权重")

save_csv(
    OUT / "full_validation_comparison.csv",
    selection_rows
)

# 验证集Dice最高的模型
winner = max(
    selection_rows,
    key=lambda row: row["dice"]
)

winner_name = winner["model"]
winner_threshold = winner["threshold"]

print("\n========== 最终模型选择 ==========")
print("模型:", winner_name)
print("阈值:", winner_threshold)
print("完整验证Dice:", winner["dice"])


# ===========================================
# 2. 读取官方237张测试集
# ===========================================

test_image_dir = ROOT / "data" / "test_img"
test_mask_dir = ROOT / "data" / "test_lab"

test_pairs = []

for image_path in sorted(
    test_image_dir.glob("*.jpg")
):
    mask_path = (
        test_mask_dir
        / f"{image_path.stem}.png"
    )

    if not mask_path.exists():
        raise FileNotFoundError(mask_path)

    test_pairs.append((image_path, mask_path))

if len(test_pairs) != 237:
    raise RuntimeError(
        f"预期237张测试图像，实际{len(test_pairs)}张"
    )


# ===========================================
# 3. 最终测试：仅测试验证集选中模型
# ===========================================

model = create_model(
    CHECKPOINTS[winner_name]
)

total_tp = total_fp = total_fn = 0

per_image_rows = []

for index, (image_path, mask_path) in enumerate(
    test_pairs, start=1
):
    image, gt = load_pair(
        image_path, mask_path
    )

    prob = predict(model, image)
    pred = prob > winner_threshold

    tp = int(np.count_nonzero(pred & gt))
    fp = int(np.count_nonzero(pred & ~gt))
    fn = int(np.count_nonzero(~pred & gt))

    total_tp += tp
    total_fp += fp
    total_fn += fn

    metrics = calculate_metrics(
        tp, fp, fn
    )

    per_image_rows.append({
        "image": image_path.name,
        "gt_pixels": int(gt.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        **metrics
    })

    # 保存237张预测Mask
    mask_image = Image.fromarray(
        pred.astype(np.uint8) * 255
    )
    mask_image.save(
        MASK_OUT / f"{image_path.stem}.png"
    )

    # 绿色=TP，红色=FP，青色=FN
    overlay = image.astype(np.float32)

    tp_region = pred & gt
    fp_region = pred & ~gt
    fn_region = ~pred & gt

    def blend(region, color):
        overlay[region] = (
            0.55 * overlay[region]
            + 0.45 * np.array(color)
        )

    blend(tp_region, [0, 255, 0])
    blend(fp_region, [255, 0, 0])
    blend(fn_region, [0, 255, 255])

    Image.fromarray(
        np.clip(overlay, 0, 255).astype(np.uint8)
    ).save(
        OVERLAY_OUT / f"{image_path.stem}.png"
    )

    if index % 30 == 0:
        print(
            f"已完成: {index}/{len(test_pairs)}",
            flush=True
        )


# ===========================================
# 4. 汇总测试指标
# ===========================================

micro = calculate_metrics(
    total_tp, total_fp, total_fn
)

mean_image_dice = float(np.mean([
    row["dice"] for row in per_image_rows
]))

mean_image_iou = float(np.mean([
    row["iou"] for row in per_image_rows
]))

test_summary = [{
    "model": winner_name,
    "threshold": winner_threshold,
    "test_images": len(test_pairs),
    "precision": micro["precision"],
    "recall": micro["recall"],
    "micro_dice": micro["dice"],
    "micro_iou": micro["iou"],
    "mean_image_dice": mean_image_dice,
    "mean_image_iou": mean_image_iou,
    "tp": total_tp,
    "fp": total_fp,
    "fn": total_fn
}]

save_csv(
    OUT / "test_summary.csv",
    test_summary
)

save_csv(
    OUT / "test_per_image.csv",
    per_image_rows
)


# ===========================================
# 5. 找出典型困难测试样本
# ===========================================

ranked = sorted(
    per_image_rows,
    key=lambda row: row["dice"]
)

print("\n========== 最低Dice的5张图像 ==========")

for row in ranked[:5]:
    print(
        row["image"],
        f"Dice={row['dice']:.4f}",
        f"Recall={row['recall']:.4f}"
    )


# ===========================================
# 6. 自动生成项目收官报告
# ===========================================

report = [
    "# Road Crack Segmentation 项目最终报告",
    "",
    "## 数据与实验设置",
    "",
    "- 数据集：DeepCrack537",
    "- 训练来源：240张",
    "- 验证来源：60张",
    "- 测试集：237张",
    "- 最终训练输入：原始384×544图像",
    "- 最终模型依据完整验证集Dice选择",
    "",
    "## 完整验证集模型比较",
    "",
    "| 模型 | 阈值 | Precision | Recall | Dice | IoU |",
    "|---|---:|---:|---:|---:|---:|"
]

for r in sorted(
    selection_rows,
    key=lambda x: x["dice"],
    reverse=True
):
    report.append(
        f"| {r['model']} | "
        f"{r['threshold']:.2f} | "
        f"{r['precision']:.4f} | "
        f"{r['recall']:.4f} | "
        f"{r['dice']:.4f} | "
        f"{r['iou']:.4f} |"
    )

report += [
    "",
    "## 最终测试",
    "",
    f"- 模型：{winner_name}",
    f"- 阈值：{winner_threshold:.2f}",
    f"- Micro Precision：{micro['precision']:.4f}",
    f"- Micro Recall：{micro['recall']:.4f}",
    f"- Micro Dice/F1：{micro['dice']:.4f}",
    f"- Micro IoU：{micro['iou']:.4f}",
    f"- Mean image Dice：{mean_image_dice:.4f}",
    "",
    "## 最困难的5张测试图像",
    ""
]

for r in ranked[:5]:
    report.append(
        f"- {r['image']}："
        f"Dice={r['dice']:.4f}，"
        f"Recall={r['recall']:.4f}"
    )

report += [
    "",
    "## 项目限制",
    "",
    "- 测试集在早期实验中曾被查看，",
    "  因此本次评价不属于完全未接触的盲测。",
    "- 模型选择与阈值优化均使用60张验证图像。",
    "- 整体Dice并不能完整反映细裂纹连通性。",
    "",
    "## 输出",
    "",
    "- full_validation_comparison.csv",
    "- test_summary.csv",
    "- test_per_image.csv",
    "- test_masks/：237张预测Mask",
    "- test_overlays/：237张错误叠加图"
]

(OUT / "project_final_report.md").write_text(
    "\n".join(report),
    encoding="utf-8"
)


print("\n========== 最终测试完成 ==========")

for key, value in test_summary[0].items():
    print(key, ":", value)

print("\n报告保存位置:", OUT)
