# 保存为 lenet5_pytorch.py 并运行
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import time

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

# ----------------------------
# 训练 / 评估 工具函数
# ----------------------------
def get_dataloaders(batch_size=64, root='./data'):
    transform = transforms.Compose([
        transforms.Pad(2),                  # MNIST: 28->32
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST 常用均值/方差
    ])
    train_set = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader

def train_one_epoch(model, device, dataloader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for imgs, targets in dataloader:
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += imgs.size(0)
    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc

def evaluate(model, device, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, targets in dataloader:
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += imgs.size(0)
    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc

# ----------------------------
# 训练示例 -- 小规模演示
# ----------------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # 超参数（可修改）
    epochs = 5
    batch_size = 128
    lr = 0.01
    momentum = 0.9

    train_loader, test_loader = get_dataloaders(batch_size=batch_size)
    model = LeNet5(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    for epoch in range(1, epochs+1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, device, train_loader, optimizer, criterion)
        test_loss, test_acc = evaluate(model, device, test_loader, criterion)
        t1 = time.time()
        print(f'Epoch {epoch:02d} | Time: {t1-t0:.1f}s | Train loss: {train_loss:.4f}, Train acc: {train_acc*100:.2f}% | '
              f'Test loss: {test_loss:.4f}, Test acc: {test_acc*100:.2f}%')

    # 保存模型
    torch.save(model.state_dict(), 'lenet5_mnist.pth')
    print('Model saved to lenet5_mnist.pth')

if __name__ == '__main__':
    main()
