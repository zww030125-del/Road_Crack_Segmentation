
from pathlib import Path
import time

import numpy as np
import torch
import onnxruntime as ort

from resnet_unet import ResNetUNet


# =====================================
# 1. 配置
# =====================================

ROOT = Path(__file__).resolve().parent

PTH_PATH = ROOT / "outputs" / "best_resnet18_frozen.pth"
ONNX_PATH = ROOT / "outputs" / "resnet18_unet.onnx"

WARMUP = 20
RUNS = 100
CPU_THREADS = 4

# 设置PyTorch CPU线程数
torch.set_num_threads(CPU_THREADS)

# =====================================
# 2. 加载PyTorch模型
# =====================================

model = ResNetUNet(
    pretrained=False,
    freeze_encoder=True
)

model.load_state_dict(
    torch.load(
        PTH_PATH,
        map_location="cpu",
        weights_only=True
    )
)

model.eval()

# =====================================
# 3. 加载ONNX Runtime模型
# =====================================

options = ort.SessionOptions()

options.intra_op_num_threads = CPU_THREADS
options.graph_optimization_level = (
    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
)

session = ort.InferenceSession(
    str(ONNX_PATH),
    sess_options=options,
    providers=["CPUExecutionProvider"]
)

print("ONNX运行设备:", session.get_providers())

# =====================================
# 4. 创建相同输入
# =====================================

rng = np.random.default_rng(42)

x_np = rng.random(
    (1, 3, 256, 256),
    dtype=np.float32
)

x_torch = torch.from_numpy(x_np)

# =====================================
# 5. 定义两种推理方式
# =====================================

@torch.inference_mode()
def pytorch_predict():
    return model(x_torch).numpy()


def onnx_predict():
    return session.run(
        ["logits"],
        {"image": x_np}
    )[0]


# =====================================
# 6. 再次检查结果一致性
# =====================================

torch_out = pytorch_predict()
onnx_out = onnx_predict()

max_diff = np.max(
    np.abs(torch_out - onnx_out)
)

print(f"最大绝对误差: {max_diff:.8f}")

np.testing.assert_allclose(
    torch_out,
    onnx_out,
    rtol=1e-3,
    atol=1e-4
)

print("输出一致性检查通过")


# =====================================
# 7. 性能测试
# =====================================

def benchmark(fn):
    for _ in range(WARMUP):
        fn()

    times = []

    for _ in range(RUNS):
        start = time.perf_counter()

        fn()

        elapsed = (
            time.perf_counter() - start
        ) * 1000

        times.append(elapsed)

    return {
        "mean": np.mean(times),
        "median": np.median(times),
        "p95": np.percentile(times, 95)
    }


torch_result = benchmark(pytorch_predict)
onnx_result = benchmark(onnx_predict)

# =====================================
# 8. 输出对比结果
# =====================================

print("\n========== PyTorch CPU ==========")

for key, value in torch_result.items():
    print(f"{key}: {value:.3f} ms")

print("\n========== ONNX Runtime CPU ==========")

for key, value in onnx_result.items():
    print(f"{key}: {value:.3f} ms")

speedup = (
    torch_result["mean"]
    / onnx_result["mean"]
)

print(f"\nONNX相对PyTorch加速比: {speedup:.2f}x")
