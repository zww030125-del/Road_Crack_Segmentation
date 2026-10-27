
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "dilation_demo"
OUT.mkdir(parents=True, exist_ok=True)

# 1. 构建一张只有中心像素为1的图像
x = torch.zeros(1, 1, 21, 21)
x[0, 0, 10, 10] = 1.0

# 2. 使用相同的3×3卷积核
kernel = torch.ones(1, 1, 3, 3)

dilations = [1, 2, 3, 4]

fig, axes = plt.subplots(
    1, 4,
    figsize=(14, 4),
    constrained_layout=True
)

for ax, d in zip(axes, dilations):

    # 不同空洞率，相同卷积权重
    output = F.conv2d(
        x,
        kernel,
        padding=d,
        dilation=d
    )

    result = output[0, 0].numpy()

    ax.imshow(
        result,
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="nearest"
    )

    ax.set_title(
        f"Dilation={d}\n"
        f"Effective kernel={2*d+1}x{2*d+1}"
    )

    ax.axis("off")

    print(
        f"Dilation={d}, "
        f"Output={tuple(output.shape)}, "
        f"Active pixels={(result > 0).sum()}"
    )

plt.savefig(
    OUT / "dilation_comparison.png",
    dpi=160
)

plt.show()
