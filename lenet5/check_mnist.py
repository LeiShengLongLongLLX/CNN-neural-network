import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 定义数据预处理
transform = transforms.Compose([transforms.ToTensor()])

# 加载训练集和测试集
train_dataset = datasets.MNIST(root='G:\Project\CNN_accerator_basedonRISC-V\data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='G:\Project\CNN_accerator_basedonRISC-V\data', train=False, download=True, transform=transform)

# 打印数量
print(f"训练集图片数量: {len(train_dataset)}")
print(f"测试集图片数量: {len(test_dataset)}")

# 随机查看几张图片
fig, axes = plt.subplots(1, 5, figsize=(10, 2))
for i in range(5):
    image, label = train_dataset[i]
    axes[i].imshow(image.squeeze(), cmap='gray')
    axes[i].set_title(f"Label: {label}")
    axes[i].axis('off')

plt.show()
