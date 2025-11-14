import cv2

# 使用 DirectShow 后端
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


if not cap.isOpened():
    print("无法打开摄像头")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取摄像头画面")
        break

    cv2.imshow('camera show', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
