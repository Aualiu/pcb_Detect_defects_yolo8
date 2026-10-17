# PCB 缺陷检测系统 — YOLOv8 + PyQt5

基于 **YOLOv8** 训练的印制电路板（PCB）表面缺陷检测模型，并使用 **PyQt5** 封装了一个可视化操作界面，支持单张图片、文件夹批量、视频文件三种检测方式，并提供缺陷统计、结果保存与 CSV 导出功能。

---

## ✨ 项目亮点

- **6 类 PCB 缺陷检测**：`open`（开路）、`short`（短路）、`mousebite`（鼠咬）、`spur`（毛刺）、`copper`（铜渣）、`pin-hole`（针孔）
- **可视化 GUI**：基于 PyQt5，左侧参数控制 + 右侧图像显示 + 缺陷统计表/明细表
- **多种检测方式**：单张图片 / 文件夹批量 / 视频文件
- **文件夹逐张手动检测**：可选逐张浏览 + 「检测当前张」，适合人工复核
- **参数实时可调**：置信度 conf、IoU 阈值、输入尺寸 imgsz、推理设备（GPU/CPU）
- **结果保存与导出**：自动保存带框图片/视频，可一键导出统计 CSV
- **中文路径兼容**：通过 `np.fromfile + cv2.imdecode` 读写中文路径图片，避免 OpenCV 编码问题

---

## 📁 项目结构

```
pcb缺陷/
├── python代码/
│   ├── pcb_detect_gui.py     # PyQt5 可视化界面（主程序）
│   ├── train.py              # 模型训练脚本
│   ├── continue.py           # 断点续训脚本
│   ├── use.py                # 命令行单图检测示例
│   └── cap.py                # 摄像头实时检测示例
├── yolo/
│   ├── data.yaml             # 数据集配置（类别、路径）
│   ├── images/               # 训练/验证图片
│   └── labels/               # 训练/验证标签
├── license_plate_exp4/
│   ├── weights/
│   │   ├── best.pt           # 最佳权重（默认加载）
│   │   └── last.pt           # 最后一轮权重
│   ├── args.yaml             # 训练参数记录
│   └── results.png           # 训练曲线
├── use_result/               # 检测结果输出目录
└── tool/
    └── X-AnyLabeling-CPU.exe # 数据标注工具
```

> **说明**：模型目录命名为 `license_plate_exp4` 是训练初期的历史命名，实际训练内容为 PCB 缺陷检测模型，不影响使用。

---

## 🔧 环境配置

### 1. Python 环境

项目使用 **Python 3.10** 虚拟环境，核心依赖：

| 包 | 版本 | 说明 |
| --- | --- | --- |
| ultralytics | 8.4.18 | YOLOv8 框架 |
| PyQt5 | — | 可视化界面 |
| opencv-python | 5.0.0 | 图像读写 |
| torch | 2.7.1+cu118 | PyTorch（CUDA 11.8） |

```bash
# 安装依赖
pip install ultralytics==8.4.18 PyQt5 opencv-python torch==2.7.1+cu118
```

### 2. 验证 CUDA

```python
import torch
print(torch.cuda.is_available())  # True 表示 GPU 可用
```

---

## 📊 数据集说明

数据集为 DeepPCB 系列 PCB 缺陷图像，共 **6 类缺陷**：

| 类别 ID | 名称 | 中文 | 说明 |
| --- | --- | --- | --- |
| 0 | open | 开路 | 导线断路 |
| 1 | short | 短路 | 不该相连的导线相连 |
| 2 | mousebite | 鼠咬 | 导线边缘不规则缺失 |
| 3 | spur | 毛刺 | 导线边缘突出的小尖刺 |
| 4 | copper | 铜渣 | 多余的铜斑块 |
| 5 | pin-hole | 针孔 | 铜面上的小圆孔 |

数据集按 YOLO 格式组织：

```
yolo/
├── data.yaml
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

`data.yaml` 内容：

```yaml
path: D:/project1/DeepPCB-master/yolo   # 数据集根目录
train: images/train
val: images/val
nc: 6
names: ['open', 'short', 'mousebite', 'spur', 'copper', 'pin-hole']
```

> 首次训练请修改 `path` 为你本地 `yolo/` 目录的绝对路径。

数据标注可使用项目自带的 `tool/X-AnyLabeling-CPU.exe`。

---

## 🏋️ 模型训练

### 首次训练

运行 [python代码/train.py](python代码/train.py)：

```python
from ultralytics import YOLO

model = YOLO(r"D:\yolo\ultralytics-8.4.18\yolov8n.pt")  # 预训练权重

results = model.train(
    data=r"yolo/data.yaml",
    epochs=100,
    imgsz=416,
    batch=2,
    device=0,
    workers=0,
    project=r"pcb缺陷",
    amp=False,
    name="license_plate_exp4"
)
```

**关键训练参数**（来自 [license_plate_exp4/args.yaml](license_plate_exp4/args.yaml)）：

- `imgsz=416`：输入分辨率 416×416（4GB 显存推荐）
- `batch=2`：显存受限下的批大小
- `amp=False`：关闭混合精度（避免部分显卡报错）
- `workers=0`：Windows 下防止多线程卡死

### 断点续训

运行 [python代码/continue.py](python代码/continue.py)：

```python
model = YOLO(r"license_plate_exp4/weights/last.pt")
model.train(resume=True, device=0, workers=0)
```

训练完成后，最佳权重保存在 `license_plate_exp4/weights/best.pt`。

---

## 🖥️ 可视化界面使用

### 启动

使用配置好的 Python 环境运行：

```bash
python python代码/pcb_detect_gui.py
```

### 操作流程

1. **加载模型**：点击「加载模型」，默认指向 `license_plate_exp4/weights/best.pt`
2. **选择输入源**：
   - 单张图片：选择一张图片 → 点「开始检测」
   - 文件夹：选择图片文件夹 → 逐张检测 或 批量检测
   - 视频文件：选择视频 → 点「开始检测」
3. **调整参数**：拖动 conf/IoU 滑块，选择 imgsz 与推理设备
4. **查看结果**：右侧显示带框图像，底部标签页查看统计与明细
5. **导出**：点「导出统计 CSV」保存检测结果

### 界面布局

```
┌─────────────────────┬──────────────────────────────────────┐
│  模型（加载权重）      │                                      │
│  输入源（单选）         │                                      │
│  ──────────────────   │         图像显示区（带检测框）          │
│  逐张检测              │                                      │
│   ◀ 上一张 / 下一张 ▶  ├──────────────────────────────────────┤
│   检测当前张            │  缺陷统计 | 检测明细 | 运行日志          │
│  ──────────────────   │                                      │
│  检测参数              │                                      │
│   conf / IoU / imgsz  │                                      │
│   推理设备             │                                      │
│  ──────────────────   │                                      │
│  输出目录              │                                      │
│  ──────────────────   │                                      │
│  开始检测 / 停止         │                                      │
│  导出统计 CSV          │                                      │
└─────────────────────┴──────────────────────────────────────┘
```

### 文件夹逐张检测

文件夹模式下，输入源下方会出现「逐张检测」面板：

- 选中文件夹后自动加载图片列表，显示第 1 张**原图**（不自动检测）
- **◀ 上一张 / 下一张 ▶**：切换预览图，不触发推理
- **检测当前张**：只对当前显示的图片跑模型
- 统计与明细**逐张累加**，适合人工复核与抽检

点「开始检测」仍可对整个文件夹**批量检测**，结果输出到 `use_result/detect_时间戳/` 子目录。

### 结果保存

- 勾选「保存检测结果」后，带框图片/视频自动保存到输出目录
- 逐张模式直接写入输出目录；批量/单图/视频模式写入带时间戳的子目录，避免覆盖
- CSV 导出包含：缺陷统计表 + 检测明细表（来源文件、类别、置信度、坐标框）

---

## 📈 检测示例

以一张 PCB 图像为例，模型检测到 `short`（短路）缺陷并绘制边界框。

（此处可替换为你自己的检测结果截图，建议放在 `use_result/` 目录下并在此引用）

```
![检测示例](use_result/my_test_final6/44000054.jpg)
```

---

## ⚙️ 训练参数调优建议

| 显存 | imgsz | batch | 说明 |
| --- | --- | --- | --- |
| 4 GB | 416 | 2 | 本项目使用配置 |
| 6 GB | 530 | 4 | 可适当增大 |
| 8 GB+ | 640 | 8 | 推荐默认配置 |

- 训练出现 CUDA out of memory：降低 `imgsz` 或 `batch`
- Windows 下训练务必设 `workers=0`，否则 DataLoader 可能卡死
- `amp=False` 可在部分显卡上避免混合精度报错

---

## ❓ 常见问题

**Q：界面启动提示模型加载失败？**
A：检查 `license_plate_exp4/weights/best.pt` 是否存在，或点「…」手动选择权重文件。

**Q：中文路径下保存的图片打不开？**
A：界面已通过 `np.fromfile + cv2.imdecode` 处理中文路径，确保使用的是项目内置的保存逻辑而非直接调用 `cv2.imwrite`。

**Q：GPU 不可用？**
A：在「推理设备」中切换为 CPU，或检查 PyTorch 的 CUDA 版本是否与驱动匹配。

**Q：批量检测很慢？**
A：适当降低 `imgsz`（如 416），或在 GPU 模式下运行。

**Q：检测框太多/太少？**
A：调高 `conf` 置信度可减少误检；调低可提高召回率。

---

## 📄 License

本项目仅供学习与研究使用。数据集版权归原作者所有。
