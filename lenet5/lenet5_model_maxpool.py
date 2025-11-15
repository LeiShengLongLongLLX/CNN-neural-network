import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        # C1: 1x32x32 -> 6x28x28
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
        # S2: 6x28x28 -> 6x14x14，改为最大池化
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # C3: 6x14x14 -> 16x10x10
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        # S4: 16x10x10 -> 16x5x5，改为最大池化
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # C5: 16x5x5 -> 120x1x1 (kernel=5)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1)
        # 全连接层
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))   # C1
        x = self.pool1(x)           # S2最大池化
        x = F.relu(self.conv2(x))   # C3
        x = self.pool2(x)           # S4最大池化
        x = F.relu(self.conv3(x))   # C5
        x = x.view(x.size(0), -1)   # 展平
        x = F.relu(self.fc1(x))     # FC1
        x = self.fc2(x)             # FC2
        return x
