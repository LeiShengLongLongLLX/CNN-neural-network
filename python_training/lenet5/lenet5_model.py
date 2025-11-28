#define lenet5
import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义lenet5神经网络模型
class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        # C1: 1x32x32 -> 6x28x28
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)  
        # S2: 6x28x28 -> 6x14x14
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        # C3: 6x14x14 -> 16x10x10
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        # S4: 16x10x10 -> 16x5x5
        # C5: 16x5x5 -> 120x1x1  (kernel=5)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1)
        # Fully connected layers
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        # x: (N,1,32,32)
        x = F.relu(self.conv1(x))   # -> (N,6,28,28)
        x = self.pool(x)            # -> (N,6,14,14)
        x = F.relu(self.conv2(x))   # -> (N,16,10,10)
        x = self.pool(x)            # -> (N,16,5,5)
        x = F.relu(self.conv3(x))   # -> (N,120,1,1)
        x = x.view(x.size(0), -1)   # -> (N,120)
        x = F.relu(self.fc1(x))     # -> (N,84)
        x = self.fc2(x)             # -> (N,num_classes)
        return x
