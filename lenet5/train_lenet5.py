# train_lenet5.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
from lenet5_model import LeNet5  # 导入模型结构

def get_dataloaders(batch_size=128, root='./data'):
    transform = transforms.Compose([
        transforms.Pad(2),  # 28->32
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    test_set  = datasets.MNIST(root=root, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def train_one_epoch(model, device, dataloader, optimizer, criterion):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
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
    return running_loss / total, correct / total

def evaluate(model, device, dataloader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, targets in dataloader:
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += imgs.size(0)
    return running_loss / total, correct / total

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    epochs = 5
    lr = 0.01
    batch_size = 128

    train_loader, test_loader = get_dataloaders(batch_size)
    model = LeNet5().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for epoch in range(1, epochs+1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, device, train_loader, optimizer, criterion)
        test_loss, test_acc = evaluate(model, device, test_loader, criterion)
        end = time.time()
        print(f"Epoch {epoch:02d} | 用时: {end-start:.1f}s | 训练: loss={train_loss:.4f}, acc={train_acc*100:.2f}% | 测试: loss={test_loss:.4f}, acc={test_acc*100:.2f}%")

    torch.save(model.state_dict(), 'lenet5_mnist.pth')
    print("模型权重已保存到 lenet5_mnist.pth")

if __name__ == "__main__":
    main()
