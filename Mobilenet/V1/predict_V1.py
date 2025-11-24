# predict_mobilenetv1.py

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

from MobileV1 import MobileNetV1    # 导入MobileNetV1模型 


# 1. 加载模型
def load_model(model_path, num_classes=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = MobileNetV1(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model, device


# 2. 图像预处理（支持 JPG）
def preprocess_image(img_path, device):
    img = Image.open(img_path).convert('RGB')  # SVHN 是 RGB，所以保持 3 通道

    # Resize 到 MobileNet 输入尺寸
    img = img.resize((32, 32))

    # 可选：如果是黑底白字的手写图像，可以反色
    # img = ImageOps.invert(img)

    # 显示预处理后的图片
    plt.imshow(img)
    plt.title("预处理后的图像")
    plt.axis('off')
    plt.show()

    # MobileNetV1 的预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4377, 0.4438, 0.4728],    # SVHN 官方统计
            std=[0.1980, 0.2010, 0.1970]
        )
    ])

    img_tensor = transform(img).unsqueeze(0).to(device)  # shape: (1,3,32,32)
    return img_tensor


# 3. 预测
def predict_image(model, img_tensor):
    with torch.no_grad():
        output = model(img_tensor)
        pred = output.argmax(dim=1).item()
    return pred


# 4. 主程序
if __name__ == '__main__':
    model_path = 'G:\\Project\\CNN_accerator_basedonRISC-V\\ModelWeigh\\mobilenetv1_svhn.pth'

    model, device = load_model(model_path)

    img_path = input("Please enter the path of the JPG image to recognize: ")

    img_tensor = preprocess_image(img_path, device)

    pred = predict_image(model, img_tensor)

    print(f'Prediction result: {pred}')