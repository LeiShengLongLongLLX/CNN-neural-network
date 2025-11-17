# predict_lenet5.py

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from lenet5_model import LeNet5
from PIL import ImageOps


# 加载模型权重
def load_model(model_path='G:\Project\CNN_accerator_basedonRISC-V\ModelWeigh\lenet5_mnist.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = LeNet5()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model

# 预处理输入照片
def preprocess_image(img_path):
    img = Image.open(img_path).convert('L') #转成灰度
    img = img.resize((32, 32)) #调整到32*32大小

    img = ImageOps.invert(img) # 自动反转为黑底白字

    # 显示预处理的照片
    # plt.ion()
    plt.imshow(img, cmap='gray')
    plt.title("预处理后的灰度图（已反色）")
    plt.axis('off')
    plt.show() 
    # plt.draw()
    # plt.pause(10)

    transform = transforms.Compose([
        # transforms.Grayscale(num_output_channels=1),  # 转灰度
        # transforms.Resize((32, 32)),                  # 调整到LeNet输入尺寸
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)  # (1,1,32,32) #转换成张量，添加batch维度
    return img_tensor

# 预测图片里面的数字
def predict_image(model, img_tensor):
    with torch.no_grad():
        output = model(img_tensor)
        pred = output.argmax(dim=1).item()
    return pred

if __name__ == '__main__':
    model = load_model('G:\Project\CNN_accerator_basedonRISC-V\ModelWeigh\lenet5_mnist.pth')
    img_path = input("Please enter the path of the image to recognize: ")
    img_tensor = preprocess_image(img_path)
    pred = predict_image(model, img_tensor)
    print(f'Predict Result: Number {pred}')