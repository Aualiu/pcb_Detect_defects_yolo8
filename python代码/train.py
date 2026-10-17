from ultralytics import YOLO

# 1. 加载预训练模型（用绝对路径，防止去 GitHub 下载报错）
model = YOLO(r"D:\yolo\ultralytics-8.4.18\yolov8n.pt")

# 2. 开始训练
results = model.train(
    data=r"C:\Users\86158\Desktop\pcb缺陷\yolo\data.yaml",  # ⚠️ 这里填你 car.yaml 的真实路径，建议用绝对路径加 r
    epochs=100,           # 训练 100 轮
    imgsz=416,            # 图片尺寸，4GB显存建议用 416，比 640 省显存
    batch=2,              # ⚠️ 你的显卡只能用 2，千万不要改大！
    device=0,
    workers=0,            # Windows 下防止多线程报错
    project=r"C:\Users\86158\Desktop\pcb缺陷",  # 训练结果保存的根目录
    amp=False,
    name="license_plate_exp"               # 本次训练保存的子文件夹名
)