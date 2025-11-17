import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import platform

from MobileV1 import MobileNetV1  # 你自己的模型文件


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


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    start_time = time.time()

    total_batches = len(train_loader)

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 100 == 0:
            progress = 100.0 * batch_idx / total_batches
            print(f"Train Epoch: {epoch} "
                  f"[{batch_idx * len(data)}/{len(train_loader.dataset)} "
                  f"({progress:.1f}%)]  Loss: {loss.item():.6f}")

    epoch_time = time.time() - start_time
    avg_loss = running_loss / total_batches
    print(f"====> Epoch {epoch} Finished | Avg Loss: {avg_loss:.6f} | Time: {epoch_time:.2f} sec\n")

    return avg_loss, epoch_time


def test(model, device, test_loader, criterion, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total_batches = len(test_loader)
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            if batch_idx % 50 == 0:
                progress = 100.0 * batch_idx / total_batches
                print(f"Test Epoch: {epoch} "
                      f"[{batch_idx}/{total_batches} ({progress:.1f}%)]  "
                      f"Batch Loss: {loss.item():.6f}")

    avg_loss = test_loss / total_batches
    accuracy = 100.0 * correct / len(test_loader.dataset)
    elapsed = time.time() - start_time

    print(f"====> Test Epoch {epoch} | Avg Loss: {avg_loss:.6f} | "
          f"Accuracy: {accuracy:.2f}% | Time: {elapsed:.2f} sec\n")

    return avg_loss, accuracy


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 打印设备信息
    print_device_info(device)

    #输入尺寸
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.SVHN(root="./data", split="train",
                                  download=True, transform=transform)
    test_dataset = datasets.SVHN(root="./data", split="test",
                                 download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    model = MobileNetV1().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    epochs = 5 #训练次数

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, criterion, epoch)
        test(model, device, test_loader, criterion, epoch)

    torch.save(model.state_dict(), "MobileNetV1_svhn.pth")
    print("Model saved to MobileNetV1_svhn.pth")


if __name__ == "__main__":
    main()
