# train_lenet5.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
from lenet5_model_maxpool import LeNet5  # 导入lenet5(maxpool版本)
import platform

# 打印设备信息
def print_device_info(device):
    print("===== Device Info =====")

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Device: CUDA")
        print(f"GPU: {gpu_name}")
        print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("Device: CPU")

    print(f"Python Platform: {platform.platform()}")
    print("========================\n")


# 加载数据集
def get_dataloaders(batch_size=128, root='G:\Project\CNN_accerator_basedonRISC-V\data'):
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

# 训练模型
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

# 测试模型
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
    # 打印设备信息
    print_device_info(device)

    # 训练轮数
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
        print(f"Epoch {epoch:02d} | Time: {end-start:.1f}s | Train: loss={train_loss:.4f}, acc={train_acc*100:.2f}% | Test: loss={test_loss:.4f}, acc={test_acc*100:.2f}%")

    torch.save(model.state_dict(), 'G:\Project\CNN_accerator_basedonRISC-V\ModelWeigh\lenet5_mnist.pth')
    print("Model have been saved to lenet5_mnist.pth")

if __name__ == "__main__":
    main()
