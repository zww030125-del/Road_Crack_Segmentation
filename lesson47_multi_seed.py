import argparse
from pathlib import Path
import random
import csv

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from torch.utils.data import Dataset, DataLoader

from models.resnet_unet import ResNetUNet
from datasets.crack_dataset import CrackDataset
from datasets.splits import get_train_val_pairs


# 通过命令行传入训练随机种子
parser = argparse.ArgumentParser()

parser.add_argument(
    "--seed",
    type=int,
    default=43
)

args = parser.parse_args()

ROOT = Path(__file__).resolve().parent

# 数据划分始终固定
SPLIT_SEED = 42

# 训练随机种子可以变化
SEED = args.seed

# 每个种子独立保存，避免覆盖模型
OUT = (
    ROOT / "outputs"
    / "lesson47"
    / f"seed_{SEED}"
)

OUT.mkdir(parents=True, exist_ok=True)

print("数据划分种子:", SPLIT_SEED)
print("训练随机种子:", SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

EPOCHS = 15
BATCH_SIZE = 2
CROP = 256
LR = 0.001
FIXED_THRESHOLD = 0.73
TARGET_PRECISION = 0.845

# 仅用于最后的验证集分组评价
LOW_CUT = 0.0779

NEIGHBORS = np.ones((3, 3), dtype=bool)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def local_contrast(image):
    """RGB图像局部暗线对比度，image范围0~1"""
    gray = (
        0.299 * image[:, :, 0]
        + 0.587 * image[:, :, 1]
        + 0.114 * image[:, :, 2]
    )
    local_mean = ndi.uniform_filter(
        gray, size=21, mode="nearest"
    )
    return np.maximum(local_mean - gray, 0)


def read_pair(pair):
    image_path, mask_path = pair

    with Image.open(image_path) as f:
        image = np.asarray(
            f.convert("RGB"), dtype=np.float32
        ) / 255.0

    with Image.open(mask_path) as f:
        mask = np.asarray(
            f.convert("L")
        ) > 127

    return image, mask


# ===========================================
# 1. 训练数据：普通裁剪或困难裁剪
# ===========================================

class PatchDataset(Dataset):

    def __init__(
        self, pairs, hard=False,
        hard_probability=0.5, seed=42
    ):
        self.pairs = pairs
        self.hard = hard
        self.hard_probability = hard_probability
        self.rng = np.random.default_rng(seed)
        self.anchors = []

        if hard:
            print("正在分析训练集中的困难裂纹...")

            n_available = 0

            for pair in pairs:
                image, mask = read_pair(pair)

                skel = skeletonize(mask)

                if not skel.any():
                    self.anchors.append(
                        np.empty((0, 2), dtype=np.int32)
                    )
                    continue

                width = (
                    2 * ndi.distance_transform_edt(mask)
                )

                contrast = local_contrast(image)

                # 分界点仅从当前训练图像计算
                cut = np.quantile(
                    contrast[skel], 0.67
                )

                # 优先选：细 + 相对低对比度
                candidate = (
                    skel
                    & (width <= 3)
                    & (contrast <= cut)
                )

                # 如候选太少，逐级回退
                if candidate.sum() < 5:
                    candidate = skel & (width <= 3)

                if candidate.sum() < 5:
                    candidate = skel

                coords = np.argwhere(
                    candidate
                ).astype(np.int32)

                self.anchors.append(coords)

                if len(coords) > 0:
                    n_available += 1

            print(
                f"具有可用候选位置的训练图像: "
                f"{n_available}/{len(pairs)}"
            )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):

        image, mask = read_pair(
            self.pairs[index]
        )

        h, w = mask.shape

        # 处理小于256的图像
        ph = max(0, CROP - h)
        pw = max(0, CROP - w)

        if ph or pw:
            image = np.pad(
                image,
                ((0, ph), (0, pw), (0, 0)),
                mode="edge"
            )
            mask = np.pad(
                mask,
                ((0, ph), (0, pw)),
                mode="constant"
            )

        h, w = mask.shape

        use_hard = (
            self.hard
            and len(self.anchors[index]) > 0
            and self.rng.random()
            < self.hard_probability
        )

        if use_hard:

            coords = self.anchors[index]

            selected = coords[
                self.rng.integers(len(coords))
            ]

            cy, cx = map(int, selected)

            # 随机偏移，避免困难像素总在中心
            jy = int(self.rng.integers(-50, 51))
            jx = int(self.rng.integers(-50, 51))

            top = int(np.clip(
                cy - CROP // 2 + jy,
                0, h - CROP
            ))

            left = int(np.clip(
                cx - CROP // 2 + jx,
                0, w - CROP
            ))

        else:

            top = int(
                self.rng.integers(0, h - CROP + 1)
            )

            left = int(
                self.rng.integers(0, w - CROP + 1)
            )

        image = image[
            top:top+CROP,
            left:left+CROP
        ]

        mask = mask[
            top:top+CROP,
            left:left+CROP
        ]

        image = torch.from_numpy(
            np.ascontiguousarray(
                image.transpose(2, 0, 1)
            )
        ).float()

        mask = torch.from_numpy(
            np.ascontiguousarray(
                mask[None].astype(np.float32)
            )
        )

        return image, mask


# ===========================================
# 2. 训练集和验证集
# ===========================================

train_pairs, val_pairs = get_train_val_pairs(
    ROOT / "data",
    seed=SPLIT_SEED
)

val_dataset = CrackDataset(
    val_pairs,
    training=False,
    augment=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ===========================================
# 3. 损失函数
# ===========================================

bce = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor(
        [5.0], device=DEVICE
    )
)


def criterion(logits, mask):
    prob = torch.sigmoid(logits)
    dims = (1, 2, 3)

    inter = (prob * mask).sum(dim=dims)
    denom = (
        prob.sum(dim=dims)
        + mask.sum(dim=dims)
    )

    dice_loss = 1 - (
        (2 * inter + 1) / (denom + 1)
    ).mean()

    return bce(logits, mask) + 0.5 * dice_loss


# ===========================================
# 4. 每轮固定阈值验证
# ===========================================

@torch.inference_mode()
def validate(model):

    model.eval()
    tp = fp = fn = 0

    for images, masks in val_loader:

        images = images.to(DEVICE)
        gt = masks.to(DEVICE) > 0.5

        pred = (
            torch.sigmoid(model(images))
            > FIXED_THRESHOLD
        )

        tp += (pred & gt).sum().item()
        fp += (pred & ~gt).sum().item()
        fn += (~pred & gt).sum().item()

    dice = 2 * tp / max(
        2 * tp + fp + fn, 1
    )

    return dice


# ===========================================
# 5. 训练一个模型
# ===========================================

def train_one(name, hard):

    print(f"\n========== {name} ==========")

    set_seed(SEED)

    dataset = PatchDataset(
        train_pairs,
        hard=hard,
        hard_probability=0.5,
        seed=SEED
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(SEED)
    )

    model = ResNetUNet(
        pretrained=True,
        freeze_encoder=True
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        (
            p for p in model.parameters()
            if p.requires_grad
        ),
        lr=LR
    )

    history = []
    best_dice = -1

    for epoch in range(EPOCHS):

        model.train()
        model.encoder.eval()

        total_loss = 0.0

        for images, masks in loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, masks)

            loss.backward()
            optimizer.step()

            total_loss += (
                loss.item() * images.size(0)
            )

        train_loss = (
            total_loss / len(dataset)
        )

        val_dice = validate(model)

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_dice": val_dice
        })

        print(
            f"{name} | "
            f"Epoch {epoch+1:02d}/{EPOCHS} | "
            f"Loss={train_loss:.4f} | "
            f"Dice={val_dice:.4f}",
            flush=True
        )

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(
                model.state_dict(),
                OUT / f"best_{name}.pth"
            )

    with open(
        OUT / f"{name}_history.csv",
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(history[0].keys())
        )
        writer.writeheader()
        writer.writerows(history)

    print(
        f"{name} 最佳固定阈值Dice: "
        f"{best_dice:.4f}"
    )

    return history


# ===========================================
# 6. 运行两组受控实验
# ===========================================

baseline_history = train_one(
    "baseline", hard=False
)

hard_history = train_one(
    "hard_sampling", hard=True
)


# ===========================================
# 7. 加载最佳权重，收集验证集概率
# ===========================================

@torch.inference_mode()
def collect_probabilities(name):

    model = ResNetUNet(
        pretrained=False,
        freeze_encoder=True
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(
            OUT / f"best_{name}.pth",
            map_location=DEVICE,
            weights_only=True
        )
    )

    model.eval()

    all_prob = []
    all_gt = []

    for images, masks in val_loader:

        prob = torch.sigmoid(
            model(images.to(DEVICE))
        ).cpu().numpy()[:, 0]

        gt = (
            masks[:, 0].numpy() > 0.5
        )

        all_prob.append(prob)
        all_gt.append(gt)

    return (
        np.concatenate(all_prob),
        np.concatenate(all_gt)
    )


# ===========================================
# 8. PR扫描
# ===========================================

THRESHOLDS = np.arange(5, 96) / 100


def scan_thresholds(prob, gt):

    result = []

    for threshold in THRESHOLDS:

        pred = prob > threshold

        tp = np.count_nonzero(pred & gt)
        fp = np.count_nonzero(pred & ~gt)
        fn = np.count_nonzero(~pred & gt)

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        dice = (
            2 * tp / max(2 * tp + fp + fn, 1)
        )

        result.append({
            "threshold": float(threshold),
            "precision": precision,
            "recall": recall,
            "dice": dice
        })

    return result


# ===========================================
# 9. 验证集真实细裂纹骨架
# ===========================================

thin_masks = []
thin_low_masks = []

for index in range(len(val_dataset)):

    image_t, mask_t = val_dataset[index]

    image = image_t.permute(
        1, 2, 0
    ).numpy()

    gt = mask_t[0].numpy() > 0.5

    skel = skeletonize(gt)

    width = (
        2 * ndi.distance_transform_edt(gt)
    )

    contrast = local_contrast(image)

    thin = skel & (width <= 3)

    thin_low = (
        thin & (contrast <= LOW_CUT)
    )

    thin_masks.append(thin)
    thin_low_masks.append(thin_low)


def geometry_metrics(prob, gt, threshold):

    thin_found = thin_total = 0
    low_found = low_total = 0

    far_skel = pred_skel = 0

    for i in range(len(prob)):

        pred = prob[i] > threshold

        dilated = ndi.binary_dilation(
            pred,
            structure=NEIGHBORS,
            iterations=2
        )

        thin = thin_masks[i]
        thin_low = thin_low_masks[i]

        thin_found += np.count_nonzero(
            dilated & thin
        )
        thin_total += np.count_nonzero(thin)

        low_found += np.count_nonzero(
            dilated & thin_low
        )
        low_total += np.count_nonzero(
            thin_low
        )

        # 远离真实标签的预测骨架
        skeleton = skeletonize(pred)

        gt_near = ndi.binary_dilation(
            gt[i],
            structure=NEIGHBORS,
            iterations=3
        )

        far_skel += np.count_nonzero(
            skeleton & ~gt_near
        )
        pred_skel += np.count_nonzero(
            skeleton
        )

    return {
        "thin_recall": (
            thin_found / max(thin_total, 1)
        ),
        "thin_low_recall": (
            low_found / max(low_total, 1)
        ),
        "far_skeleton_ratio": (
            far_skel / max(pred_skel, 1)
        )
    }


# ===========================================
# 10. 最终对比
# ===========================================

results = {}
prob_maps = {}

for name in ["baseline", "hard_sampling"]:

    prob, gt = collect_probabilities(name)

    prob_maps[name] = prob

    metrics = scan_thresholds(prob, gt)

    best = max(
        metrics,
        key=lambda x: x["dice"]
    )

    # 选择达到目标Precision后Recall最高的点
    feasible = [
        m for m in metrics
        if m["precision"] >= TARGET_PRECISION
    ]

    matched = (
        max(feasible, key=lambda x: x["recall"])
        if feasible else
        min(
            metrics,
            key=lambda x: abs(
                x["precision"] - TARGET_PRECISION
            )
        )
    )

    geo = geometry_metrics(
        prob, gt, matched["threshold"]
    )

    results[name] = {
        "curve": metrics,
        "best": best,
        "matched": matched,
        "geometry": geo
    }

    print(f"\n========== {name} ==========")

    print(
        f"最佳Dice={best['dice']:.4f}, "
        f"Threshold={best['threshold']:.2f}"
    )

    print(
        f"匹配点: "
        f"P={matched['precision']:.4f}, "
        f"R={matched['recall']:.4f}, "
        f"Dice={matched['dice']:.4f}, "
        f"Threshold={matched['threshold']:.2f}"
    )

    print(
        f"Thin Recall={geo['thin_recall']:.4f}"
    )

    print(
        f"Thin+Low Recall="
        f"{geo['thin_low_recall']:.4f}"
    )

    print(
        f"Far Skeleton Ratio="
        f"{geo['far_skeleton_ratio']:.4f}"
    )

    if abs(
        matched["precision"] - TARGET_PRECISION
    ) > 0.01:
        print(
            "注意：该模型没有足够接近"
            "目标Precision的阈值。"
        )

summary_rows = []

for name, result in results.items():

    best = result["best"]
    matched = result["matched"]
    geo = result["geometry"]

    summary_rows.append({
        "seed": SEED,
        "model": name,
        "best_dice": best["dice"],
        "best_threshold": best["threshold"],
        "matched_precision": matched["precision"],
        "matched_recall": matched["recall"],
        "matched_dice": matched["dice"],
        "thin_recall": geo["thin_recall"],
        "thin_low_recall": geo["thin_low_recall"],
        "far_skeleton_ratio": geo["far_skeleton_ratio"]
    })

with open(
    OUT / "seed_summary.csv",
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(summary_rows[0].keys())
    )

    writer.writeheader()
    writer.writerows(summary_rows)

print("随机种子", SEED, "的实验结果已保存")
# ===========================================
# 11. 可视化
# ===========================================

fig, axes = plt.subplots(
    1, 3,
    figsize=(16, 4.5),
    constrained_layout=True
)

for name, history in [
    ("baseline", baseline_history),
    ("hard_sampling", hard_history)
]:
    axes[0].plot(
        [r["epoch"] for r in history],
        [r["val_dice"] for r in history],
        label=name
    )

for name, result in results.items():

    curve = result["curve"]

    axes[1].plot(
        [r["recall"] for r in curve],
        [r["precision"] for r in curve],
        label=name
    )

    axes[2].plot(
        [r["threshold"] for r in curve],
        [r["dice"] for r in curve],
        label=name
    )

axes[0].set_title("Validation Dice by Epoch")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Dice")

axes[1].set_title("Precision-Recall")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")

axes[2].set_title("Dice vs Threshold")
axes[2].set_xlabel("Threshold")
axes[2].set_ylabel("Dice")

for ax in axes:
    ax.grid(alpha=0.2)
    ax.legend()

plt.savefig(
    OUT / "training_comparison.png",
    dpi=160
)


# ===========================================
# 12. 相同困难样本的预测比较
# ===========================================

target_names = [
    "IMG11-1.jpg",
    "11150.jpg",
    "11178-3.jpg"
]

selected = [
    i for i, pair in enumerate(val_pairs)
    if pair[0].name in target_names
]

if selected:

    fig, axes = plt.subplots(
        len(selected),
        4,
        figsize=(15, 4 * len(selected)),
        squeeze=False,
        constrained_layout=True
    )

    for row, idx in enumerate(selected):

        image_t, mask_t = val_dataset[idx]

        image = image_t.permute(
            1, 2, 0
        ).numpy()

        axes[row, 0].imshow(image)
        axes[row, 0].set_title(
            val_pairs[idx][0].name
        )

        axes[row, 1].imshow(
            mask_t[0].numpy(),
            cmap="gray"
        )
        axes[row, 1].set_title("Ground Truth")

        for col, name in [
            (2, "baseline"),
            (3, "hard_sampling")
        ]:

            threshold = results[name][
                "matched"
            ]["threshold"]

            pred = (
                prob_maps[name][idx] > threshold
            )

            axes[row, col].imshow(
                pred,
                cmap="gray",
                vmin=0,
                vmax=1
            )

            axes[row, col].set_title(
                f"{name}\nT={threshold:.2f}"
            )

        for ax in axes[row]:
            ax.axis("off")

    plt.savefig(
        OUT / "hard_examples.png",
        dpi=160
    )

plt.show()

print("\n第46课完成，结果目录:", OUT)
