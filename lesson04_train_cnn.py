
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================================
# 1. 生成模拟数据
# 0：横向裂纹，1：纵向裂纹
# ======================================

def create_data(n, seed):
    g = torch.Generator().manual_seed(seed)

    # 随机灰度背景
    images = 0.65 + 0.35 * torch.rand(
        n, 1, 32, 32, generator=g
    )

    labels = torch.randint(0, 2, (n,), generator=g)

    for i in range(n):
        pos = int(torch.randint(8, 24, (1,), generator=g))

        if labels[i] == 0:
            # 横向黑线
            images[i, 0, pos:pos+2, 4:28] = 0.08
        else:
            # 纵向黑线
            images[i, 0, 4:28, pos:pos+2] = 0.08

    return images, labels


train_x, train_y = create_data(400, 100)
val_x, val_y = create_data(100, 200)
test_x, test_y = create_data(100, 300)

train_loader = DataLoader(
    TensorDataset(train_x, train_y),
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    TensorDataset(val_x, val_y),
    batch_size=32
)

test_loader = DataLoader(
    TensorDataset(test_x, test_y),
    batch_size=32
)

# ======================================
# 2. 建立CNN
# ======================================

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = SimpleCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ======================================
# 3. 定义评价函数
# ======================================

def evaluate(loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total


# ======================================
# 4. 训练CNN
# ======================================

for epoch in range(15):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

    avg_loss = total_loss / len(train_x)
    val_acc = evaluate(val_loader)

    print(
        f"Epoch {epoch+1:02d} | "
        f"Loss: {avg_loss:.4f} | "
        f"Val Accuracy: {val_acc:.2%}"
    )

# ======================================
# 5. 独立测试
# ======================================

test_acc = evaluate(test_loader)
print(f"\n最终测试集准确率: {test_acc:.2%}")

torch.save(model.state_dict(), "outputs/simple_cnn.pth")
print("模型已保存为 simple_cnn.pth")
