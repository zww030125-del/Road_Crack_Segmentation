
from pathlib import Path
import random
import csv

import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from torch.utils.data import Dataset, DataLoader

from models.resnet_unet import ResNetUNet
from datasets.splits import get_train_val_pairs


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "final_full_image"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SPLIT_SEED = 42
TRAIN_SEED = 42

EPOCHS = 15
LR = 0.001
ACCUM_STEPS = 2
VAL_THRESHOLD = 0.73


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FullImageDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]

        with Image.open(image_path) as im:
            image = np.asarray(
                im.convert("RGB"),
                dtype=np.float32
            ) / 255.0

        with Image.open(mask_path) as im:
            mask = (
                np.asarray(im.convert("L")) > 127
            ).astype(np.float32)

        h, w = mask.shape

        assert image.shape[:2] == mask.shape
        assert h % 32 == 0 and w % 32 == 0, (
            f"图像尺寸不满足32的倍数: {h}x{w}"
        )

        image = torch.from_numpy(
            image.transpose(2, 0, 1).copy()
        )

        mask = torch.from_numpy(
            mask[None].copy()
        )

        return image, mask


set_seed(TRAIN_SEED)

train_pairs, val_pairs = get_train_val_pairs(
    ROOT / "data",
    seed=SPLIT_SEED
)

train_loader = DataLoader(
    FullImageDataset(train_pairs),
    batch_size=1,
    shuffle=True,
    num_workers=0,
    generator=torch.Generator().manual_seed(
        TRAIN_SEED
    )
)

val_loader = DataLoader(
    FullImageDataset(val_pairs),
    batch_size=1,
    shuffle=False,
    num_workers=0
)

model = ResNetUNet(
    pretrained=True,
    freeze_encoder=True
).to(DEVICE)

bce = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor(
        [5.0], device=DEVICE
    )
)


def calculate_loss(logits, masks):
    probs = torch.sigmoid(logits)

    dims = (1, 2, 3)

    intersection = (
        probs * masks
    ).sum(dim=dims)

    denominator = (
        probs.sum(dim=dims)
        + masks.sum(dim=dims)
    )

    dice = (
        2 * intersection + 1
    ) / (denominator + 1)

    return (
        bce(logits, masks)
        + 0.5 * (1 - dice.mean())
    )


optimizer = torch.optim.Adam(
    (p for p in model.parameters() if p.requires_grad),
    lr=LR
)


@torch.inference_mode()
def validate():
    model.eval()

    tp = fp = fn = 0

    for images, masks in val_loader:
        images = images.to(DEVICE)
        gt = masks.to(DEVICE) > 0.5

        pred = (
            torch.sigmoid(model(images))
            > VAL_THRESHOLD
        )

        tp += (pred & gt).sum().item()
        fp += (pred & ~gt).sum().item()
        fn += (~pred & gt).sum().item()

    return 2 * tp / max(2 * tp + fp + fn, 1)


print("训练设备:", DEVICE)
print("训练图像:", len(train_pairs))
print("验证图像:", len(val_pairs))

history = []
best_dice = -1.0

for epoch in range(EPOCHS):
    model.train()
    model.encoder.eval()

    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0

    for step, (images, masks) in enumerate(
        train_loader, start=1
    ):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        logits = model(images)
        loss = calculate_loss(logits, masks)

        # 记录未缩放的训练损失
        total_loss += loss.item()

        # 梯度累积
        (loss / ACCUM_STEPS).backward()

        if (
            step % ACCUM_STEPS == 0
            or step == len(train_loader)
        ):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    train_loss = total_loss / len(train_loader)
    val_dice = validate()

    history.append({
        "epoch": epoch + 1,
        "loss": train_loss,
        "val_dice": val_dice
    })

    print(
        f"Epoch {epoch+1:02d}/{EPOCHS} | "
        f"Loss={train_loss:.4f} | "
        f"Full-Val Dice={val_dice:.4f}",
        flush=True
    )

    if val_dice > best_dice:
        best_dice = val_dice

        torch.save(
            model.state_dict(),
            OUT / "best_full_image.pth"
        )

with open(
    OUT / "training_history.csv",
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

print("\n完整图像训练完成")
print(f"最佳完整验证Dice(0.73): {best_dice:.4f}")
print("模型路径:", OUT / "best_full_image.pth")
