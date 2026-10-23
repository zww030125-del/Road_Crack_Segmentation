
from pathlib import Path

import torch

# 原始模型
from unet_model import MiniUNet as OldMiniUNet
from resnet_unet import ResNetUNet as OldResNetUNet

# 模块化后的新模型
from models.mini_unet import MiniUNet as NewMiniUNet
from models.resnet_unet import ResNetUNet as NewResNetUNet


ROOT = Path(__file__).resolve().parent

torch.manual_seed(42)

device = torch.device("cpu")

# 使用同一张模拟RGB图像
x = torch.rand(1, 3, 256, 256)


def check_model(name, old_model, new_model, weight_path):

    print(f"\n开始检查: {name}")

    weights = torch.load(
        weight_path,
        map_location=device,
        weights_only=True
    )

    # 分别加载相同权重
    old_model.load_state_dict(weights)
    new_model.load_state_dict(weights)

    old_model.eval()
    new_model.eval()

    with torch.inference_mode():
        old_output = old_model(x)
        new_output = new_model(x)

    max_diff = (
        old_output - new_output
    ).abs().max().item()

    print("旧模型输出尺寸:", old_output.shape)
    print("新模型输出尺寸:", new_output.shape)
    print("最大绝对误差:", max_diff)

    torch.testing.assert_close(
        old_output,
        new_output,
        rtol=0,
        atol=0
    )

    print("模型一致性检查通过")


# ====================================
# 检查Mini U-Net
# ====================================

check_model(
    name="Mini U-Net",
    old_model=OldMiniUNet(),
    new_model=NewMiniUNet(),
    weight_path=ROOT / "outputs" / "best_unet_bce_dice.pth"
)


# ====================================
# 检查ResNet18 U-Net
# ====================================

check_model(
    name="ResNet18 U-Net",
    old_model=OldResNetUNet(
        pretrained=False,
        freeze_encoder=True
    ),
    new_model=NewResNetUNet(
        pretrained=False,
        freeze_encoder=True
    ),
    weight_path=ROOT / "outputs" / "best_resnet18_frozen.pth"
)

print("\n全部模型检查通过")
