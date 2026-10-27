
from pathlib import Path
import csv

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from models.resnet_unet import ResNetUNet
from datasets.crack_dataset import CrackDataset
from datasets.splits import get_train_val_pairs


# =====================================
# 1. 基本配置
# =====================================

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "lesson41"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

THRESHOLD = 0.73

print("设备:", DEVICE)


# =====================================
# 2. 加载验证集与模型
# =====================================

_, val_pairs = get_train_val_pairs(
    ROOT / "data"
)

dataset = CrackDataset(
    val_pairs,
    training=False,
    augment=False
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

print("验证图像数量:", len(dataset))


# =====================================
# 3. 计算局部暗线对比度
# =====================================

def calculate_contrast(image):
    """
    image: [H, W, 3], 范围0~1

    估计某个像素相对于周围区域有多暗。
    注意：这是一个局部亮度代理指标，
    并非严格的物理对比度测量。
    """

    gray = (
        0.299 * image[:, :, 0]
        + 0.587 * image[:, :, 1]
        + 0.114 * image[:, :, 2]
    )

    gray_t = torch.from_numpy(
        gray.astype(np.float32)
    )[None, None]

    padded = F.pad(
        gray_t,
        (10, 10, 10, 10),
        mode="replicate"
    )

    local_mean = F.avg_pool2d(
        padded,
        kernel_size=21,
        stride=1
    )[0, 0].numpy()

    contrast = np.maximum(
        local_mean - gray,
        0
    )

    return contrast


# =====================================
# 4. 遍历验证集
# =====================================

records = []
rows = []

gt_prob_list = []
gt_contrast_list = []

# 分别统计裂纹与背景的概率分布
bins = np.linspace(0, 1, 51)
positive_hist = np.zeros(50, dtype=np.int64)
negative_hist = np.zeros(50, dtype=np.int64)


with torch.inference_mode():

    for index in range(len(dataset)):

        image_t, mask_t = dataset[index]

        x = image_t.unsqueeze(0).to(DEVICE)

        prob = torch.sigmoid(
            model(x)
        )[0, 0].cpu().numpy()

        image = image_t.permute(
            1, 2, 0
        ).numpy()

        gt = mask_t[0].numpy() > 0.5
        pred = prob > THRESHOLD

        tp = int((pred & gt).sum())
        fp = int((pred & ~gt).sum())
        fn = int((~pred & gt).sum())

        gt_count = int(gt.sum())

        dice_den = 2 * tp + fp + fn

        dice = (
            2 * tp / dice_den
            if dice_den > 0 else 1.0
        )

        recall = (
            tp / gt_count
            if gt_count > 0 else None
        )

        contrast = calculate_contrast(image)

        if gt_count > 0:
            gt_prob_list.append(prob[gt])
            gt_contrast_list.append(contrast[gt])

        positive_hist += np.histogram(
            prob[gt], bins=bins
        )[0]

        negative_hist += np.histogram(
            prob[~gt], bins=bins
        )[0]

        name = val_pairs[index][0].name

        rows.append({
            "image": name,
            "gt_pixels": gt_count,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "recall": recall,
            "dice": dice
        })

        records.append({
            "name": name,
            "image": image,
            "gt": gt,
            "prob": prob,
            "pred": pred,
            "recall": recall,
            "gt_pixels": gt_count
        })


# =====================================
# 5. 保存逐图统计
# =====================================

with open(
    OUT / "validation_diagnosis.csv",
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(rows[0].keys())
    )

    writer.writeheader()
    writer.writerows(rows)


# =====================================
# 6. 分析漏检像素的概率
# =====================================

gt_probs = np.concatenate(gt_prob_list)
gt_contrasts = np.concatenate(gt_contrast_list)

false_negative_probs = gt_probs[
    gt_probs <= THRESHOLD
]

deep_fn = (
    false_negative_probs < 0.30
).sum()

near_fn = (
    false_negative_probs >= 0.30
).sum()

print("\n========== 漏检概率诊断 ==========")

print("真实裂纹像素数:", len(gt_probs))
print("漏检像素数:", len(false_negative_probs))

if len(false_negative_probs) > 0:

    print(
        "低概率漏检占比(P<0.30): "
        f"{deep_fn / len(false_negative_probs):.2%}"
    )

    print(
        "中间概率漏检占比(0.30~0.73): "
        f"{near_fn / len(false_negative_probs):.2%}"
    )


# =====================================
# 7. 局部对比度与召回率
# =====================================

q1, q2 = np.quantile(
    gt_contrasts,
    [1/3, 2/3]
)

groups = {
    "Low contrast": gt_contrasts <= q1,
    "Medium contrast": (
        (gt_contrasts > q1)
        & (gt_contrasts <= q2)
    ),
    "High contrast": gt_contrasts > q2
}

print("\n========== 局部对比度诊断 ==========")
print(f"对比度分界点: {q1:.4f}, {q2:.4f}")

for name, selection in groups.items():

    count = int(selection.sum())

    if count == 0:
        print(name, "无可用像素")
        continue

    recall = (
        gt_probs[selection] > THRESHOLD
    ).mean()

    print(
        f"{name}: "
        f"GT像素={count}, "
        f"Recall={recall:.4f}"
    )


# =====================================
# 8. 概率分布可视化
# =====================================

centers = (
    bins[:-1] + bins[1:]
) / 2

pos_freq = (
    positive_hist / max(positive_hist.sum(), 1)
)

neg_freq = (
    negative_hist / max(negative_hist.sum(), 1)
)

plt.figure(figsize=(10, 5))

plt.plot(
    centers,
    pos_freq,
    label="Ground Truth Crack"
)

plt.plot(
    centers,
    neg_freq,
    label="Ground Truth Background"
)

plt.axvline(
    THRESHOLD,
    color="red",
    linestyle="--",
    label=f"Threshold={THRESHOLD}"
)

plt.xlabel("Predicted Crack Probability")
plt.ylabel("Fraction Within Each Class")
plt.title("Prediction Probability Distribution")

plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()

plt.savefig(
    OUT / "probability_distribution.png",
    dpi=160
)


# =====================================
# 9. 找到漏检最严重的验证图像
# =====================================

# 排除GT裂纹极少的图像，避免排序不稳定
valid_records = [
    r for r in records
    if r["gt_pixels"] >= 50
]

worst = sorted(
    valid_records,
    key=lambda r: r["recall"]
)[:3]

print("\n========== 低召回率样本 ==========")

for r in worst:
    print(
        r["name"],
        f"Recall={r['recall']:.4f}"
    )


# =====================================
# 10. 绘制错误样本
# =====================================

if worst:

    fig, axes = plt.subplots(
        len(worst), 5,
        figsize=(17, 4 * len(worst)),
        squeeze=False,
        constrained_layout=True
    )

    for i, r in enumerate(worst):

        image = r["image"]
        gt = r["gt"]
        prob = r["prob"]
        pred = r["pred"]

        # 红色=误检，青色=漏检
        fp_mask = pred & ~gt
        fn_mask = ~pred & gt

        overlay = image.copy()

        overlay[fp_mask] = (
            0.45 * image[fp_mask]
            + 0.55 * np.array([1.0, 0.0, 0.0])
        )

        overlay[fn_mask] = (
            0.35 * image[fn_mask]
            + 0.65 * np.array([0.0, 1.0, 1.0])
        )

        panels = [
            ("Original", image, None),
            ("GT", gt, "gray"),
            ("Probability", prob, "viridis"),
            ("Prediction", pred, "gray"),
            ("Errors", overlay, None)
        ]

        for j, (title, arr, cmap) in enumerate(panels):

            ax = axes[i, j]

            ax.imshow(
                arr,
                cmap=cmap,
                vmin=0 if cmap else None,
                vmax=1 if cmap else None
            )

            ax.set_title(
                f"{title}\n{r['name']}"
                if j == 0 else title,
                fontsize=10
            )

            ax.axis("off")

    plt.savefig(
        OUT / "worst_validation_examples.png",
        dpi=160
    )

plt.show()

print("\n诊断完成，结果保存在:", OUT)
