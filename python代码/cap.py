from ultralytics import YOLO
import cv2

# 1. 加载模型
model = YOLO(r"C:\Users\86158\Desktop\pcb缺陷\license_plate_exp4\weights\best.pt")

# 2. 打开摄像头（Windows 下强烈建议加上 cv2.CAP_DSHOW，防止黑屏）
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("无法打开摄像头，请把 0 改成 1 试试")
    exit()

print("摄像头已打开，按 'q' 键退出")

while True:
    ret, frame = cap.read()
    if not ret:
        print("读取失败")
        break

    # 3. 对每一帧进行预测
    results = model.predict(
        source=frame,
        imgsz=416,        # 👈 摄像头实时检测建议用 416
        conf=0.4,
        device=0,
        verbose=False
    )

    # 4. 画框并显示
    annotated_frame = results[0].plot()
    cv2.imshow("YOLO PCB Camera", annotated_frame)

    # 5. 按 'q' 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()