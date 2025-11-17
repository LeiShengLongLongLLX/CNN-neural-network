import cv2
import torch
from torchvision import transforms
from PIL import Image, ImageOps
from lenet5_model import LeNet5  # 确保你的模型定义文件在同目录下

def load_model(model_path='G:\Project\CNN_accerator_basedonRISC-V\ModelWeigh\lenet5_mnist.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = LeNet5()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model, device

def preprocess_frame(frame):
    # frame 是 BGR 格式
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转灰度
    img = cv2.resize(img, (32, 32))  # 调整大小

    pil_img = Image.fromarray(img)
    pil_img = ImageOps.invert(pil_img)  # 反色（黑底白字）

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    img_tensor = transform(pil_img).unsqueeze(0)  # (1,1,32,32)
    return img_tensor

def main():
    model, device = load_model('lenet5_mnist.pth')

    # -------------------------------
    # 使用 DirectShow 后端修复黑屏
    # -------------------------------
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # 尝试启用 MJPG（很多 USB 摄像头必须用它才不黑屏）
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    print("按 'q' 退出")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头画面")
            break

        img_tensor = preprocess_frame(frame).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            pred = output.argmax(dim=1).item()

        # 在摄像头画面上显示预测结果
        cv2.putText(frame, f'Predict Number : {pred}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

        cv2.imshow('Camera show', frame)

        # 在终端实时输出预测
        print(f'Predict Number: {pred}', end='\r')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
