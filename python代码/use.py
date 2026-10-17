from ultralytics import YOLO

# 1. 加载你训练好的模型
model = YOLO(r"C:\Users\86158\Desktop\pcb缺陷\license_plate_exp4\weights\best.pt")

# 2. 预测视频
results = model.predict(
    source=r"C:\Users\86158\Desktop\pcb缺陷\yolo\images\val\44000054.jpg",
    stream=True,                      # 流式处理，防止内存溢出
    save=True,                        # 保存带检测框的视频
    imgsz=530,                        # 👈 降回 416，降低显存占用
    batch=1,                          # 👈 明确指定一次只处理一帧
    conf=0.4,
    project=r"C:\Users\86158\Desktop\pcb缺陷\use_result",
    name="my_test_final"
)

print("开始处理...")
for r in results:
    pass
print("处理完成")