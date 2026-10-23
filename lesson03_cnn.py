import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================
# 第一部分：使用固定卷积核
# =============================

# 构造5×5图像
image = torch.tensor([
    [0, 0, 0, 1, 1],
    [0, 0, 0, 1, 1],
    [0, 0, 0, 1, 1],
    [0, 0, 0, 1, 1],
    [0, 0, 0, 1, 1]
], dtype=torch.float32)

# 增加Batch和Channel维度
image = image.unsqueeze(0).unsqueeze(0)

# 定义竖直边缘检测核
kernel = torch.tensor([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
], dtype=torch.float32)

kernel = kernel.view(1, 1, 3, 3)

# 进行卷积
output = F.conv2d(image, kernel)

print("输入尺寸：", image.shape)
print("输出尺寸：", output.shape)
print("卷积结果：")
print(output[0, 0])


# =============================
# 第二部分：使用可学习的卷积层
# =============================

device = torch.device("cuda")

# 创建两张32×32的RGB模拟图像
x = torch.rand(2, 3, 32, 32, device=device)

# 创建CNN卷积层
conv = nn.Conv2d(
    in_channels=3,
    out_channels=16,
    kernel_size=3,
    padding=0
).to(device)

# 前向传播
y = conv(x)

print("\nCNN卷积层：")
print("输入尺寸：", x.shape)
print("输出尺寸：", y.shape)
print("卷积核权重尺寸：", conv.weight.shape)
print("权重是否可以训练：", conv.weight.requires_grad)