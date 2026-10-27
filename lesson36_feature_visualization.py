
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from models.resnet_unet import ResNetUNet
from datasets.crack_dataset import CrackDataset
from datasets.splits import get_train_val_pairs


# =====================================
# 1. 基本配置
# =====================================

ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    ROOT / "outputs" / "best_resnet18_frozen.pth"
)

OUTPUT_DIR = ROOT / "outputs" / "feature_maps"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

THRESHOLD = 0.73

# 修改这个数字，可以查看不同验证集样本
SAMPLE_INDEX = 0

print("运行设备:", DEVICE)


# =====================================
# 2. 读取验证集样本
# =====================================

train_pairs, val_pairs = get_train_val_pairs(
    ROOT / "data"
)

val_dataset = CrackDataset(
    val_pairs,
    training=False,
    augment=False
)

image_tensor, gt_tensor = val_dataset[
    SAMPLE_INDEX
]

image_name = val_pairs[SAMPLE_INDEX][0].name

# 用于绘图的RGB图像
image_np = image_tensor.permute(
    1, 2, 0
).numpy()

gt_np = gt_tensor[0].numpy()

# 添加Batch维度
x = image_tensor.unsqueeze(0).to(DEVICE)

print("样本名称:", image_name)
print("输入尺寸:", x.shape)


# =====================================
# 3. 加载训练好的ResNet18 U-Net
# =====================================

model = ResNetUNet(
    pretrained=False,
    freeze_encoder=True
).to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )
)

model.eval()

print("模型加载完成")


# =====================================
# 4. 注册Forward Hooks
# =====================================

features = {}

layer_specs = [
    ("Conv1", model.encoder.relu),
    ("Layer1", model.encoder.layer1),
    ("Layer2", model.encoder.layer2),
    ("Layer3", model.encoder.layer3),
    ("Layer4", model.encoder.layer4),
    ("Decoder0", model.dec0),
]


def create_hook(name):

    def hook(module, inputs, output):
        # 保存特征图，但不修改网络输出
        features[name] = output.detach().cpu()

    return hook


handles = []

for name, module in layer_specs:

    handle = module.register_forward_hook(
        create_hook(name)
    )

    handles.append(handle)


# =====================================
# 5. 执行一次前向传播
# =====================================

try:
    with torch.inference_mode():
        logits = model(x)
        probabilities = torch.sigmoid(logits)

finally:
    # 及时移除钩子
    for handle in handles:
        handle.remove()


prob_np = probabilities[0, 0].cpu().numpy()

pred_np = prob_np > THRESHOLD

print("\n不同网络层的特征尺寸:")

for name, _ in layer_specs:
    print(name, features[name].shape)


# =====================================
# 6. 保存预测结果概览
# =====================================

overlay = image_np.copy()

overlay[pred_np] = (
    0.55 * image_np[pred_np]
    + 0.45 * np.array([1.0, 0.0, 0.0])
)

fig, axes = plt.subplots(
    1, 5, figsize=(17, 4)
)

axes[0].imshow(image_np)
axes[0].set_title("Original")

axes[1].imshow(
    gt_np, cmap="gray", vmin=0, vmax=1
)
axes[1].set_title("Ground Truth")

axes[2].imshow(
    prob_np, cmap="viridis", vmin=0, vmax=1
)
axes[2].set_title("Probability")

axes[3].imshow(
    pred_np, cmap="gray", vmin=0, vmax=1
)
axes[3].set_title("Prediction")

axes[4].imshow(overlay)
axes[4].set_title("Overlay")

for ax in axes:
    ax.axis("off")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / f"sample_{SAMPLE_INDEX:02d}_overview.png",
    dpi=160
)


# =====================================
# 7. 可视化各层特征图
# =====================================

# 每层展示3个空间响应变化较明显的通道
NUM_CHANNELS = 3

fig, axes = plt.subplots(
    len(layer_specs),
    NUM_CHANNELS,
    figsize=(12, 22)
)

for row, (name, _) in enumerate(layer_specs):

    # [1, C, H, W] -> [C, H, W]
    fmap = features[name][0].float()

    channels, height, width = fmap.shape

    # 根据空间响应的标准差，选择展示通道
    scores = fmap.flatten(1).std(
        dim=1,
        unbiased=False
    )

    selected = torch.topk(
        scores,
        k=NUM_CHANNELS
    ).indices.tolist()

    for col, channel_idx in enumerate(selected):

        activation = fmap[
            channel_idx
        ].numpy()

        # 单独归一化，便于观察当前通道的空间变化
        low, high = np.percentile(
            activation,
            [2, 98]
        )

        if high - low < 1e-8:
            display_map = np.zeros_like(
                activation
            )
        else:
            display_map = np.clip(
                (activation - low) / (high - low),
                0, 1
            )

        ax = axes[row, col]

        ax.imshow(
            display_map,
            cmap="magma",
            interpolation="nearest",
            vmin=0,
            vmax=1
        )

        ax.set_title(
            f"{name} | Ch {channel_idx}\n"
            f"{height} x {width}",
            fontsize=10
        )

        ax.axis("off")

plt.suptitle(
    "Feature Maps at Different Network Stages",
    fontsize=15
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / f"sample_{SAMPLE_INDEX:02d}_features.png",
    dpi=160
)

plt.show()

print("\n特征可视化完成")
print("保存位置:", OUTPUT_DIR)


# =====================================
# 8. 分析Decoder0对最终预测的贡献
# =====================================

# Decoder0输出 [16, 256, 256]
decoder_features = features["Decoder0"][0].numpy()

# 最后1×1卷积的16个通道权重
weights = (
    model.output.weight
    .detach()
    .cpu()
    .numpy()[0, :, 0, 0]
)

bias = model.output.bias.detach().cpu().item()

# 每个通道对Logit的贡献
contributions = (
    decoder_features
    * weights[:, None, None]
)

# 正向贡献
positive = np.maximum(
    contributions, 0
).sum(axis=0)

# 负向贡献的绝对值
negative = np.maximum(
    -contributions, 0
).sum(axis=0)

# 重建最终Logit
reconstructed_logits = (
    positive - negative + bias
)

# 重建概率
reconstructed_prob = 1 / (
    1 + np.exp(
        -np.clip(reconstructed_logits, -40, 40)
    )
)

difference = np.abs(
    reconstructed_prob - prob_np
).max()

print("重建概率最大误差:", difference)
print("最终层16个权重:", weights)

fig, axes = plt.subplots(
    1, 4, figsize=(16, 4),
    constrained_layout=True
)

axes[0].imshow(image_np)
axes[0].set_title("Original")

axes[1].imshow(positive, cmap="magma")
axes[1].set_title("Positive Logit Contributions")

axes[2].imshow(negative, cmap="magma")
axes[2].set_title("Negative Logit Contributions")

axes[3].imshow(
    reconstructed_prob,
    cmap="viridis",
    vmin=0,
    vmax=1
)
axes[3].set_title("Reconstructed Probability")

for ax in axes:
    ax.axis("off")

plt.savefig(
    OUTPUT_DIR / "decoder_contributions.png",
    dpi=160
)

plt.show()
