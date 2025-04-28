import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 三层卷积 + 池化
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  # 输入 3 通道，输出 16 通道
        self.pool  = nn.MaxPool2d(2)                 # 池化：宽高各 /2
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        # 全连接层，输出 3 类
        self.fc = nn.Linear(64 * 28 * 28, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # → 16×112×112
        x = self.pool(F.relu(self.conv2(x)))  # → 32×56×56
        x = self.pool(F.relu(self.conv3(x)))  # → 64×28×28
        x = x.view(x.size(0), -1)             # 展平
        x = self.fc(x)                        # → [batch, 3]
        return x

model = SimpleCNN()