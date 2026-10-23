
import torch
import torch.nn as nn

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 1. 连续两次卷积
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


# 2. 构建简化U-Net
class MiniUNet(nn.Module):
    def __init__(self):
        super().__init__()

        # 编码器
        self.enc1 = DoubleConv(1, 16)
        self.enc2 = DoubleConv(16, 32)
        self.pool = nn.MaxPool2d(2)

        # 瓶颈层
        self.bottleneck = DoubleConv(32, 64)

        # 解码器
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = DoubleConv(64, 32)

        self.up2 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec2 = DoubleConv(32, 16)

        # 输出1通道裂纹分数
        self.out = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        # 编码
        e1 = self.enc1(x)                  # 128×128
        e2 = self.enc2(self.pool(e1))     # 64×64

        # 瓶颈
        b = self.bottleneck(self.pool(e2)) # 32×32

        # 解码 + 跳跃连接
        d1 = self.up1(b)                  # 64×64
        d1 = torch.cat([d1, e2], dim=1)
        d1 = self.dec1(d1)

        d2 = self.up2(d1)                 # 128×128
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)

        return self.out(d2)


# 3. 生成8张模拟裂纹图像及Mask
images = 0.75 + torch.rand(8, 1, 128, 128) * 0.15
masks = torch.zeros_like(images)

for i in range(8):
    pos = 20 + i * 10
    if i % 2 == 0:
        masks[i, 0, 15:115, pos:pos+4] = 1
    else:
        masks[i, 0, pos:pos+4, 15:115] = 1

# 用深色像素表示裂纹
images[masks.bool()] = 0.08

images = images.to(device)
masks = masks.to(device)


# 4. 创建模型
model = MiniUNet().to(device)

# 裂纹占比较少，适当增加正类权重
loss_fn = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([12.0], device=device)
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# 5. 训练30次
for step in range(1, 31):
    model.train()

    logits = model(images)
    loss = loss_fn(logits, masks)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step == 1 or step % 5 == 0:
        print(f"Step {step:02d} | Loss: {loss.item():.4f}")


# 6. 查看像素级预测结果
model.eval()

with torch.no_grad():
    logits = model(images)
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities > 0.5).float()

print("\n输入图像尺寸：", images.shape)
print("预测概率尺寸：", probabilities.shape)
print("二值掩膜尺寸：", predictions.shape)
print("实际裂纹像素数：", int(masks.sum().item()))
print("预测裂纹像素数：", int(predictions.sum().item()))
