
import torch
import torch.nn as nn

from .mini_unet import DoubleConv


class MiniUNetPP(nn.Module):
    """
    三层尺度的简化U-Net++。

    用于学习嵌套密集跳跃连接。
    当前不启用深度监督。
    """

    def __init__(self):
        super().__init__()

        self.pool = nn.MaxPool2d(2)

        # 1. 编码器
        self.enc0 = DoubleConv(3, 8)
        self.enc1 = DoubleConv(8, 16)
        self.enc2 = DoubleConv(16, 32)

        # 2. 上采样模块
        self.up10 = nn.ConvTranspose2d(
            16, 8, 2, stride=2
        )

        self.up20 = nn.ConvTranspose2d(
            32, 16, 2, stride=2
        )

        self.up11 = nn.ConvTranspose2d(
            16, 8, 2, stride=2
        )

        # 3. 嵌套卷积节点
        self.conv01 = DoubleConv(8 + 8, 8)

        self.conv11 = DoubleConv(16 + 16, 16)

        # 融合三个来源的特征
        self.conv02 = DoubleConv(
            8 + 8 + 8, 8
        )

        # 4. 最终分割输出
        self.output = nn.Conv2d(8, 1, 1)

    def forward(self, x):

        # ===== 编码器 =====
        x00 = self.enc0(x)
        x10 = self.enc1(self.pool(x00))
        x20 = self.enc2(self.pool(x10))

        # ===== 第一组嵌套节点 =====

        # X(0,1)：浅层 + 中层上采样
        x01 = self.conv01(
            torch.cat(
                [x00, self.up10(x10)],
                dim=1
            )
        )

        # X(1,1)：中层 + 深层上采样
        x11 = self.conv11(
            torch.cat(
                [x10, self.up20(x20)],
                dim=1
            )
        )

        # ===== 第二组嵌套节点 =====

        # X(0,2)：原始浅层 + 中间节点 + 深层融合
        x02 = self.conv02(
            torch.cat(
                [
                    x00,
                    x01,
                    self.up11(x11)
                ],
                dim=1
            )
        )

        return self.output(x02)


if __name__ == "__main__":

    model = MiniUNetPP()

    x = torch.randn(1, 3, 256, 256)

    with torch.inference_mode():
        y = model(x)

    params = sum(
        p.numel() for p in model.parameters()
    )

    print("输入:", x.shape)
    print("输出:", y.shape)
    print("参数量:", params)
