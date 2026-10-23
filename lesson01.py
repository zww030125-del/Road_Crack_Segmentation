import torch

# 使用GPU
device = torch.device("cuda")

# 定义需要学习的参数，初始值为1
w = torch.tensor(1.0, device=device, requires_grad=True)

# 定义优化器，学习率为0.1
optimizer = torch.optim.SGD([w], lr=0.1)

# 重复训练10次
for epoch in range(10):

    # 1. 前向计算：计算损失
    loss = (w - 3) ** 2

    # 2. 清除上一次的梯度
    optimizer.zero_grad()

    # 3. 反向传播：自动计算梯度
    loss.backward()

    # 4. 更新参数
    optimizer.step()

    print(
        f"第{epoch + 1}次："
        f"w={w.item():.4f}"
    )