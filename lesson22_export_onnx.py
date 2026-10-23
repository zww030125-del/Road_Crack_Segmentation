
from pathlib import Path

import numpy as np
import torch
import onnx
import onnxruntime as ort

from resnet_unet import ResNetUNet


# =====================================
# 1. 路径配置
# =====================================

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"

MODEL_PATH = (
    OUTPUT_DIR / "best_resnet18_frozen.pth"
)

ONNX_PATH = (
    OUTPUT_DIR / "resnet18_unet.onnx"
)

# =====================================
# 2. 加载PyTorch模型
# =====================================

# 导出阶段使用CPU，减少环境依赖
device = torch.device("cpu")

model = ResNetUNet(
    pretrained=False,
    freeze_encoder=True
).to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )
)

model.eval()

print("PyTorch模型加载成功")


# =====================================
# 3. 创建示例输入
# =====================================

# Batch=1, RGB, 256×256
torch.manual_seed(42)

dummy_input = torch.rand(
    1, 3, 256, 256,
    dtype=torch.float32
)


# =====================================
# 4. 导出ONNX
# =====================================

torch.onnx.export(
    model,
    (dummy_input,),
    str(ONNX_PATH),
    input_names=["image"],
    output_names=["logits"],
    opset_version=18,
    dynamo=True,
    external_data=False
)

print("ONNX导出成功:", ONNX_PATH)


# =====================================
# 5. 检查ONNX模型是否合法
# =====================================

onnx_model = onnx.load(str(ONNX_PATH))

onnx.checker.check_model(onnx_model)

print("ONNX模型结构检查通过")


# =====================================
# 6. PyTorch预测
# =====================================

with torch.inference_mode():
    torch_output = model(
        dummy_input
    ).numpy()


# =====================================
# 7. ONNX Runtime预测
# =====================================

session = ort.InferenceSession(
    str(ONNX_PATH),
    providers=["CPUExecutionProvider"]
)

onnx_output = session.run(
    ["logits"],
    {
        "image": dummy_input.numpy()
    }
)[0]


# =====================================
# 8. 比较输出是否一致
# =====================================

max_diff = np.max(
    np.abs(torch_output - onnx_output)
)

print("PyTorch输出尺寸:", torch_output.shape)
print("ONNX输出尺寸:", onnx_output.shape)

print("最大绝对误差:", max_diff)

np.testing.assert_allclose(
    torch_output,
    onnx_output,
    rtol=1e-3,
    atol=1e-4
)

print("PyTorch与ONNX输出一致性检查通过")

file_size = (
    ONNX_PATH.stat().st_size / (1024 ** 2)
)

print(f"ONNX文件大小: {file_size:.2f} MB")
