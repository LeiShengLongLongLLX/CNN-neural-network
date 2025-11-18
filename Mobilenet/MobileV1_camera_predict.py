import cv2
import torch
from torchvision import transforms
from PIL import Image
from MobileV1 import MobileNetV1   # 导入MobileNetV1模型


# 1. 加载 MobileNetV1 模型
def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = MobileNetV1(num_classes=10)        # 数字分类 0~9
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model, device


# 2. 摄像头帧预处理（SVHN 格式）
def preprocess_frame(frame):
    # frame 是 BGR → 转成 RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize 到 MobileNetV1 的输入 32x32
    img = cv2.resize(img, (32, 32))

    # PIL 图像格式
    pil_img = Image.fromarray(img)

    # SVHN 的标准 Normalize（非常重要）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4377, 0.4438, 0.4728],
            std=[0.1980, 0.2010, 0.1970]
        )
    ])

    img_tensor = transform(pil_img).unsqueeze(0)   # shape: (1,3,32,32)
    return img_tensor


# 3. 主程序：摄像头实时预测
def main():
    model_path = 'G:\\Project\\CNN_accerator_basedonRISC-V\\ModelWeigh\\mobilenetv1_svhn.pth'
    model, device = load_model(model_path)

    # 修复黑屏：使用 DirectShow 后端
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # 启动 MJPG 编码，避免 USB 摄像头黑屏
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Cannot open camera")
        return

    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot read camera frame")
            break

        img_tensor = preprocess_frame(frame).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            pred = output.argmax(dim=1).item()

        # 显示到画面上
        cv2.putText(frame, f'Predict Number: {pred}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('MobileNetV1 Camera Demo', frame)

        print(f'Predict Number: {pred}', end='\r')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
