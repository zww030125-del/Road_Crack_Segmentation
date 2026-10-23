import torch
import torch.nn as nn

# 1. 设置随机种子
torch.manual_seed(42)

device = torch.device("cuda")

# 2. 准备训练数据
x = torch.tensor([
    [0.2, 0.2],
    [0.2, 0.8],
    [0.8, 0.2],
    [0.8, 0.8]
], dtype=torch.float32, device=device)

# 每个样本的类别标签
y = torch.tensor(
    [0, 1, 1, 0],
    dtype=torch.long,
    device=device
)

# 3. 构建神经网络
model = nn.Sequential(
    nn.Linear(2, 8),
    
    nn.Linear(8, 2)
).to(device)

# 4. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=0.02
)

# 5. 模型训练
for epoch in range(1000):

    outputs = model(x)

    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 200 == 0:
        print(
            f"Epoch {epoch+1}, "
            f"Loss: {loss.item():.4f}"
        )

# 6. 查看训练结果
model.eval()

with torch.no_grad():
    predictions = model(x).argmax(dim=1)

print("真实标签：", y.cpu().tolist())
print("预测标签：", predictions.cpu().tolist())