"""Train the frozen ResNet18 U-Net baseline at native image resolution."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.morphology import skeletonize
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from datasets.splits import get_train_val_pairs
from losses.soft_cldice import soft_cldice_loss
from models.resnet_unet import ResNetUNet


ROOT = Path(__file__).resolve().parent
SPLIT_SEED = 42
ACCUMULATION = 2
LEARNING_RATE = 1e-3
POS_WEIGHT = 5.0
DICE_WEIGHT = 0.5
CLDICE_WEIGHT = 0.25
SKELETON_ITERATIONS = 10
AUGMENT_PROBABILITY = 0.7
CONTRAST_RANGE = (0.65, 1.10)
BRIGHTNESS_RANGE = (-0.05, 0.05)
HARD_FRACTION = 0.25
THIN_WIDTH = 3.0
LOW_CONTRAST = 0.0779
THRESHOLDS = np.arange(5, 96) / 100
PRECISION_TARGET = 0.85


class FullImages(Dataset):
    def __init__(self, pairs, augment_rng=None):
        self.pairs = pairs
        self.augment_rng = augment_rng

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        with Image.open(image_path) as source:
            rgb = np.asarray(source.convert("RGB"), dtype=np.float32) / 255.0
        with Image.open(mask_path) as source:
            mask = np.asarray(source.convert("L")) > 127
        if rgb.shape[:2] != mask.shape:
            raise ValueError(f"Image and mask sizes differ: {image_path}")
        image = torch.from_numpy(rgb.transpose(2, 0, 1).copy())
        rng = self.augment_rng
        if rng is not None and rng.random() < AUGMENT_PROBABILITY:
            mean = image.mean(dim=(1, 2), keepdim=True)
            alpha = rng.uniform(*CONTRAST_RANGE)
            beta = rng.uniform(*BRIGHTNESS_RANGE)
            image = ((image - mean) * alpha + mean + beta).clamp(0, 1)
        return image, torch.from_numpy(mask[None])


def thin_low_skeleton(image, mask):
    skeleton = skeletonize(mask)
    gray = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
    contrast = ndimage.uniform_filter(gray, size=21, mode="nearest") - gray
    width = 2 * ndimage.distance_transform_edt(mask)
    thin_low = skeleton & (width <= THIN_WIDTH) & (contrast <= LOW_CONTRAST)
    return thin_low, int(skeleton.sum())


def rank_training_images(dataset):
    rows = []
    for index, (image, mask) in enumerate(dataset):
        thin_low, total = thin_low_skeleton(image.numpy(), mask.numpy()[0])
        points = int(thin_low.sum())
        rows.append({
            "index": index,
            "image": dataset.pairs[index][0].name,
            "thin_low_points": points,
            "skeleton_points": total,
            "difficulty": points / max(total, 1),
        })
    rows.sort(key=lambda row: (
        -row["difficulty"], -row["thin_low_points"], row["image"],
    ))
    limit = int(np.ceil(len(dataset) * HARD_FRACTION))
    hard_indices = [
        row["index"] for row in rows[:limit] if row["thin_low_points"] > 0
    ]
    hard_set = set(hard_indices)
    for row in rows:
        row["extra_draw_per_epoch"] = int(row["index"] in hard_set)
    return rows, hard_indices


def objective(logits, targets, bce):
    probabilities = logits.sigmoid()
    dims = (1, 2, 3)
    intersection = (probabilities * targets).sum(dim=dims)
    denominator = probabilities.sum(dim=dims) + targets.sum(dim=dims)
    dice_loss = 1 - ((2 * intersection + 1) / (denominator + 1)).mean()
    structure_loss = soft_cldice_loss(logits, targets, iterations=SKELETON_ITERATIONS)
    return bce(logits, targets) + DICE_WEIGHT * dice_loss + CLDICE_WEIGHT * structure_loss


def train_epoch(model, loader, optimizer, bce, device):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    for step, (images, masks) in enumerate(loader, start=1):
        logits = model(images.to(device))
        loss = objective(logits, masks.to(device, dtype=logits.dtype), bce)
        value = loss.item()
        if not np.isfinite(value):
            raise FloatingPointError(f"Non-finite loss at training step {step}")
        total_loss += value
        group_start = (step - 1) // ACCUMULATION * ACCUMULATION
        group_size = min(ACCUMULATION, len(loader) - group_start)
        (loss / group_size).backward()
        if step % ACCUMULATION == 0 or step == len(loader):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    return total_loss / len(loader)


def count_above_thresholds(scores):
    # Each score votes for the thresholds strictly below it.
    bins = np.searchsorted(THRESHOLDS, scores, side="left")
    hist = np.bincount(bins.ravel(), minlength=len(THRESHOLDS) + 1)
    return np.cumsum(hist[::-1])[-2::-1]


@torch.inference_mode()
def validate(model, samples, device):
    model.eval()
    true_positive = np.zeros(len(THRESHOLDS), dtype=np.int64)
    predicted_positive = np.zeros_like(true_positive)
    thin_low_hits = np.zeros_like(true_positive)
    positives = thin_low_points = 0
    for image, truth, thin_low in samples:
        scores = model(image[None].to(device)).sigmoid()[0, 0].cpu().numpy()
        predicted_positive += count_above_thresholds(scores)
        true_positive += count_above_thresholds(scores[truth])
        # A point is covered when any prediction in its 5x5 neighborhood is positive.
        near_scores = ndimage.maximum_filter(scores, size=5, mode="nearest")
        thin_low_hits += count_above_thresholds(near_scores[thin_low])
        positives += int(truth.sum())
        thin_low_points += int(thin_low.sum())

    return [
        {
            "threshold": float(threshold),
            "precision": int(tp) / max(int(predicted), 1),
            "recall": int(tp) / max(positives, 1),
            "dice": 2 * int(tp) / max(int(predicted) + positives, 1),
            "iou": int(tp) / max(int(predicted - tp) + positives, 1),
            "tp": int(tp),
            "fp": int(predicted - tp),
            "fn": positives - int(tp),
            "thin_low_covered": int(covered),
            "thin_low_points": thin_low_points,
            "thin_low_recall_2px": int(covered) / thin_low_points if thin_low_points else None,
        }
        for threshold, tp, predicted, covered in zip(
            THRESHOLDS, true_positive, predicted_positive, thin_low_hits,
        )
    ]


def encoder_hash(model):
    digest = hashlib.sha256()
    for value in model.encoder.state_dict().values():
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def save_table(path, rows):
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42, help="Training seed; split stays fixed")
    parser.add_argument("--output", type=Path, help="New run directory")
    args = parser.parse_args()
    if args.epochs < 1 or args.seed < 0:
        parser.error("--epochs must be positive and --seed nonnegative")
    output = args.output or ROOT / "outputs" / "full_image_baseline" / f"seed_{args.seed}"
    if output.exists():
        parser.error(f"Output already exists: {output}. Specify a new --output.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    train_pairs, val_pairs = get_train_val_pairs(ROOT / "data", seed=SPLIT_SEED)
    if not train_pairs or not val_pairs:
        raise ValueError("Both the training and validation split must contain images")
    print(f"Device={device} | train={len(train_pairs)} | validation={len(val_pairs)}", flush=True)
    model = ResNetUNet(pretrained=True, freeze_encoder=True).to(device)
    initial_encoder = encoder_hash(model)

    train = FullImages(train_pairs)
    difficulty, hard_indices = rank_training_images(train)
    # All originals are seen once, and each selected hard image is seen one extra time.
    epoch_pairs = train_pairs + [train_pairs[index] for index in hard_indices]
    loader = DataLoader(
        FullImages(epoch_pairs, augment_rng=np.random.default_rng(args.seed)),
        batch_size=1, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_samples = []
    for image, mask in FullImages(val_pairs):
        truth = mask.numpy()[0]
        thin_low, _ = thin_low_skeleton(image.numpy(), truth)
        val_samples.append((image, truth, thin_low))

    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT], device=device))
    optimizer = torch.optim.Adam(
        (param for param in model.parameters() if param.requires_grad), lr=LEARNING_RATE,
    )
    output.mkdir(parents=True, exist_ok=False)
    save_table(output / "training_image_difficulty.csv", difficulty)
    print(f"Hard images={len(hard_indices)} | training draws/epoch={len(loader)}", flush=True)
    history = []
    best = None
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, loader, optimizer, bce, device)
        curve = validate(model, val_samples, device)
        point = max(curve, key=lambda row: row["dice"])
        row = {"epoch": epoch, "loss": loss, **point}
        history.append(row)
        save_table(output / "training_history.csv", history)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | loss={loss:.4f} | "
            f"full-val Dice={point['dice']:.4f} @ {point['threshold']:.2f}",
            flush=True,
        )
        if best is None or point["dice"] > best["dice"]:
            best = row
            feasible = [row for row in curve if row["precision"] >= PRECISION_TARGET]
            matched = max(feasible, key=lambda row: (row["recall"], row["dice"]), default=None)
            torch.save(model.state_dict(), output / "best_model.pth")
            save_table(output / "validation_thresholds.csv", curve)

    if encoder_hash(model) != initial_encoder:
        raise RuntimeError("Frozen encoder parameters or buffers changed during training")
    summary = {
        "protocol": "native_full_image_validation",
        "architecture": "ResNet18 U-Net",
        "encoder": "ImageNet ResNet18_Weights.DEFAULT; all layers and BN frozen",
        "encoder_sha256": initial_encoder,
        "training_seed": args.seed,
        "split_seed": SPLIT_SEED,
        "epochs": args.epochs,
        "train_images": len(train_pairs),
        "validation_images": len(val_pairs),
        "hard_images": len(hard_indices),
        "draws_per_epoch": len(loader),
        "hard_fraction": HARD_FRACTION,
        "hard_width_max": THIN_WIDTH,
        "hard_contrast_max": LOW_CONTRAST,
        "augmentation": {
            "probability": AUGMENT_PROBABILITY,
            "contrast": CONTRAST_RANGE,
            "brightness": BRIGHTNESS_RANGE,
        },
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "batch_size": 1,
        "gradient_accumulation": ACCUMULATION,
        "loss": {"pos_weight": POS_WEIGHT, "dice_weight": DICE_WEIGHT,
                 "soft_cldice_weight": CLDICE_WEIGHT, "skeleton_iterations": SKELETON_ITERATIONS},
        "device": str(device),
        "peak_cuda_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1)
        if device.type == "cuda" else None,
        "best": best,
        "precision_target": PRECISION_TARGET,
        "precision_target_point": matched,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    print(f"Checkpoint: {output / 'best_model.pth'}", flush=True)


if __name__ == "__main__":
    main()
