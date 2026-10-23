
from pathlib import Path
import gc
import time

import numpy as np
import torch

from unet_model import MiniUNet
from resnet_unet import ResNetUNet


# =====================================
# 1. 基本配置
# =====================================

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 1
IMAGE_SIZE = 256

WARMUP = 20
RUNS = 100

print("设备:", device)

if device.type == "cuda":
    print("显卡:", torch.cuda.get_device_name(0))


# =====================================
# 2. 配置待比较的模型
# =====================================

MODELS = [
    {
        "name": "Mini U-Net",
        "create": lambda: MiniUNet(),
        "path": OUTPUT_DIR / "best_unet_bce_dice.pth",
        "threshold": 0.76
    },
    {
        "name": "ResNet18 Frozen",
        "create": lambda: ResNetUNet(
            pretrained=False,
            freeze_encoder=True
        ),
        "path": OUTPUT_DIR / "best_resnet18_frozen.pth",
        "threshold": 0.73
    }
]


# =====================================
# 3. 单张图像推理
# =====================================

def predict(model, x, threshold):
    logits = model(x)
    probabilities = torch.sigmoid(logits)
    return probabilities > threshold


# =====================================
# 4. 模型性能测试
# =====================================

def benchmark(config):

    name = config["name"]
    path = config["path"]
    threshold = config["threshold"]

    if not path.exists():
        print("权重文件不存在:", path)
        return

    # 创建并加载模型
    model = config["create"]().to(device)

    state_dict = torch.load(
        path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.eval()

    # 参数量
    num_params = sum(
        p.numel() for p in model.parameters()
    )

    # 权重文件大小
    file_mb = path.stat().st_size / (1024 ** 2)

    # 固定尺寸的模拟输入
    x = torch.rand(
        BATCH_SIZE, 3,
        IMAGE_SIZE, IMAGE_SIZE,
        device=device
    )

    # 预热
    with torch.inference_mode():
        for _ in range(WARMUP):
            pred = predict(model, x, threshold)

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    times = []

    # 正式计时
    with torch.inference_mode():

        for _ in range(RUNS):

            if device.type == "cuda":

                start = torch.cuda.Event(
                    enable_timing=True
                )

                end = torch.cuda.Event(
                    enable_timing=True
                )

                start.record()

                pred = predict(
                    model, x, threshold
                )

                end.record()

                torch.cuda.synchronize()

                elapsed_ms = start.elapsed_time(end)

            else:
                start = time.perf_counter()

                pred = predict(
                    model, x, threshold
                )

                elapsed_ms = (
                    time.perf_counter() - start
                ) * 1000

            times.append(elapsed_ms)

    # 统计结果
    avg_ms = float(np.mean(times))
    median_ms = float(np.median(times))
    p95_ms = float(np.percentile(times, 95))

    fps = (
        1000.0 / avg_ms
        if avg_ms > 0 else 0
    )

    print("\n" + "=" * 40)
    print("模型:", name)
    print("=" * 40)

    print(f"参数量: {num_params:,}")
    print(f"权重大小: {file_mb:.2f} MB")

    print(f"平均推理延迟: {avg_ms:.3f} ms")
    print(f"中位数延迟: {median_ms:.3f} ms")
    print(f"P95延迟: {p95_ms:.3f} ms")
    print(f"理论FPS: {fps:.1f}")

    if device.type == "cuda":

        peak_mb = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 2)
        )

        print(f"PyTorch显存峰值: {peak_mb:.2f} MiB")

    # 清理，避免两个模型互相占用显存
    del pred, x, model, state_dict
    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()


# =====================================
# 5. 依次测试
# =====================================

for config in MODELS:
    benchmark(config)

print("\n全部测试完成")
