import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# 定义数据转换（保持原始32x32大小）
transform = transforms.Compose([
    transforms.ToTensor()
])

# 加载SVHN训练集和测试集
train_dataset = datasets.SVHN(root='G:\Project\CNN_accerator_basedonRISC-V\data\SVHN', split='train', download=True, transform=transform)
test_dataset = datasets.SVHN(root='G:\Project\CNN_accerator_basedonRISC-V\data\SVHN', split='test', download=True, transform=transform)

print(f"train dataset size: {len(train_dataset)}")
print(f"test dataset size: {len(test_dataset)}")

# 随机显示训练集中几张图片和标签
def imshow(img, label):
    img = img.numpy().transpose((1, 2, 0))  # CHW -> HWC
    plt.imshow(img)
    plt.title(f"Label: {label}")
    plt.axis('off')
    plt.show()

# 随机显示5张图片
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i in range(5):
    image, label = train_dataset[i]
    label = label % 10  # 把10转成0
    img = image.numpy().transpose((1, 2, 0))  # CHW转HWC，matplotlib显示需要
    axes[i].imshow(img)
    axes[i].set_title(f"label: {label}")
    axes[i].axis('off')

plt.show()