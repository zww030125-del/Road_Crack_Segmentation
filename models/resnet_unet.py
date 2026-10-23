
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18, ResNet18_Weights


# 解码器的基础卷积模块
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class ResNetUNet(nn.Module):

    def __init__(self, pretrained=True, freeze_encoder=True):
        super().__init__()

        self.freeze_encoder = freeze_encoder

        # 1. 加载ResNet18编码器
        weights = (
            ResNet18_Weights.DEFAULT
            if pretrained else None
        )

        self.encoder = resnet18(weights=weights)

        # 2. 冻结编码器
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # 3. 解码器
        self.dec4 = DoubleConv(512 + 256, 128)
        self.dec3 = DoubleConv(128 + 128, 64)
        self.dec2 = DoubleConv(64 + 64, 32)
        self.dec1 = DoubleConv(32 + 64, 16)

        # 恢复到原始图像分辨率
        self.dec0 = DoubleConv(16 + 3, 16)

        # 二分类输出
        self.output = nn.Conv2d(16, 1, 1)

        # ImageNet预训练权重对应的归一化参数
        self.register_buffer(
            "mean",
            torch.tensor(
                [0.485, 0.456, 0.406]
            ).view(1, 3, 1, 1)
        )

        self.register_buffer(
            "std",
            torch.tensor(
                [0.229, 0.224, 0.225]
            ).view(1, 3, 1, 1)
        )

    def train(self, mode=True):
        super().train(mode)

        # 冻结时保持ResNet18的BatchNorm统计值不变
        if self.freeze_encoder:
            self.encoder.eval()

        return self

    def up_and_concat(self, x, skip):
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        return torch.cat([x, skip], dim=1)

    def forward(self, x):

        original = x

        # ImageNet归一化
        x = (x - self.mean) / self.std

        # ========= 编码器 =========

        # 256 -> 128
        e0 = self.encoder.relu(
            self.encoder.bn1(
                self.encoder.conv1(x)
            )
        )

        # 128 -> 64
        e1 = self.encoder.layer1(
            self.encoder.maxpool(e0)
        )

        # 64 -> 32
        e2 = self.encoder.layer2(e1)

        # 32 -> 16
        e3 = self.encoder.layer3(e2)

        # 16 -> 8
        e4 = self.encoder.layer4(e3)

        # ========= 解码器 =========

        d4 = self.dec4(
            self.up_and_concat(e4, e3)
        )

        d3 = self.dec3(
            self.up_and_concat(d4, e2)
        )

        d2 = self.dec2(
            self.up_and_concat(d3, e1)
        )

        d1 = self.dec1(
            self.up_and_concat(d2, e0)
        )

        d0 = self.dec0(
            self.up_and_concat(d1, original)
        )

        return self.output(d0)


# ==================================
# 测试网络结构
# ==================================

if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = ResNetUNet(
        pretrained=True,
        freeze_encoder=True
    ).to(device)

    model.eval()

    # 模拟2张RGB图像
    x = torch.rand(
        2, 3, 256, 256,
        device=device
    )

    with torch.inference_mode():
        output = model(x)

    print("设备:", device)
    print("输入尺寸:", x.shape)
    print("输出尺寸:", output.shape)

    total_params = sum(
        p.numel() for p in model.parameters()
    )

    trainable_params = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    print("总参数量:", total_params)
    print("可训练参数量:", trainable_params)
