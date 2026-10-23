
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from unet_model import MiniUNet
from lesson07_dataset import val_loader

ROOT = Path(__file__).resolve().parent

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODELS = {
    "BCE + Dice": ROOT / "outputs" / "best_unet_bce_dice.pth",
    "BCE + Dice + Aug":
        ROOT / "outputs" / "best_unet_bce_dice_aug.pth"
}

# 自动搜索0.05到0.95，间隔0.01
thresholds = np.round(
    np.arange(0.05, 0.951, 0.01), 2
)


# ==================================
# 1. 只进行一次模型预测
# ==================================

@torch.inference_mode()
def get_predictions(model_path):

    model = MiniUNet().to(device)

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device,
            weights_only=True
        )
    )

    model.eval()

    all_probs = []
    all_labels = []

    for images, masks in val_loader:

        images = images.to(device)

        logits = model(images)
        probs = torch.sigmoid(logits)

        all_probs.append(
            probs.cpu().numpy().ravel()
        )

        all_labels.append(
            (masks.numpy().ravel() > 0.5)
        )

    return (
        np.concatenate(all_probs),
        np.concatenate(all_labels)
    )


# ==================================
# 2. 自动搜索最佳阈值
# ==================================

plt.figure(figsize=(9, 5))

for model_name, model_path in MODELS.items():

    probabilities, labels = get_predictions(
        model_path
    )

    gt_count = np.count_nonzero(labels)

    results = []

    for t in thresholds:

        prediction = probabilities > t

        tp = np.count_nonzero(
            prediction & labels
        )

        pred_count = np.count_nonzero(
            prediction
        )

        dice = (
            2 * tp / (pred_count + gt_count)
            if (pred_count + gt_count) > 0
            else 1.0
        )

        results.append(dice)

    results = np.array(results)

    best_index = np.argmax(results)

    best_threshold = thresholds[best_index]
    best_dice = results[best_index]
    # 使用最佳阈值重新计算Precision和Recall
    best_pred = probabilities > best_threshold

    tp = np.count_nonzero(best_pred & labels)
    fp = np.count_nonzero(best_pred & ~labels)
    fn = np.count_nonzero(~best_pred & labels)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    print(f"最佳阈值下Precision: {precision:.4f}")
    print(f"最佳阈值下Recall: {recall:.4f}")
    print(f"\n模型: {model_name}")
    print(f"最佳阈值: {best_threshold:.2f}")
    print(f"最佳验证Dice: {best_dice:.4f}")

    plt.plot(
        thresholds,
        results,
        label=model_name
    )

    plt.scatter(
        best_threshold,
        best_dice
    )

plt.xlabel("Prediction Threshold")
plt.ylabel("Validation Dice")
plt.title("Automatic Threshold Search")

plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    ROOT / "outputs" / "threshold_search.png",
    dpi=150
)

plt.show()
