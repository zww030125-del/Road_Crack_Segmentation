
import torch
import torch.nn as nn


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


class MiniUNet(nn.Module):
    def __init__(self):
        super().__init__()

        # 编码器
        self.enc1 = DoubleConv(3, 8)
        self.enc2 = DoubleConv(8, 16)
        self.pool = nn.MaxPool2d(2)

        # 瓶颈层
        self.bottleneck = DoubleConv(16, 32)

        # 解码器
        self.up1 = nn.ConvTranspose2d(
            32, 16, 2, stride=2
        )
        self.dec1 = DoubleConv(32, 16)

        self.up2 = nn.ConvTranspose2d(
            16, 8, 2, stride=2
        )
        self.dec2 = DoubleConv(16, 8)

        # 输出裂纹Logits
        self.output = nn.Conv2d(8, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))

        b = self.bottleneck(self.pool(e2))

        d1 = self.up1(b)
        d1 = torch.cat([d1, e2], dim=1)
        d1 = self.dec1(d1)

        d2 = self.up2(d1)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)

        return self.output(d2)
