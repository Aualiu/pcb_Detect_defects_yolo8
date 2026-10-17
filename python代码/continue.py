from ultralytics import YOLO

model = YOLO(r"C:\Users\86158\Desktop\pcb缺陷\license_plate_exp4\weights\last.pt")

# 开始续训
results = model.train(
    resume=True,  # 开启断点续训，它会自动找回之前的参数
    device=0,
    workers=0
)