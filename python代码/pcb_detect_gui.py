# -*- coding: utf-8 -*-
"""
PCB 缺陷检测可视化界面
基于 YOLOv8 + PyQt5，支持单张图片 / 文件夹批量 / 视频文件检测

运行环境（虚拟环境）：
    C:\\Users\\86158\\.conda\\envs\\yolov8\\python.exe
运行方式：
    python pcb_detect_gui.py
"""
import os
import sys
import csv
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QLineEdit,
    QGroupBox, QVBoxLayout, QHBoxLayout, QFormLayout, QRadioButton,
    QButtonGroup, QSlider, QComboBox, QCheckBox, QSplitter, QTabWidget,
    QTableWidget, QTableWidgetItem, QTextEdit, QFileDialog, QProgressBar,
    QMessageBox, QHeaderView, QAbstractItemView,
)

from ultralytics import YOLO

# ---------------------------------------------------------------- 常量配置
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = BASE_DIR / "license_plate_exp4" / "weights" / "best.pt"
DEFAULT_OUTPUT = BASE_DIR / "use_result"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".m4v"}


# ---------------------------------------------------------------- 图像显示控件
class ImageView(QLabel):
    """自适应缩放的图像显示标签"""

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(480, 360)
        self.setStyleSheet("background:#20232a; color:#888; font-size:14px;")
        self.setText("暂无图像")
        self._pix = None

    def set_image(self, bgr):
        """传入 OpenCV 的 BGR 图像并显示"""
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        self._pix = QPixmap.fromImage(qimg)
        self._refresh()

    def _refresh(self):
        if self._pix is None:
            return
        self.setPixmap(self._pix.scaled(self.size(), Qt.KeepAspectRatio,
                                        Qt.SmoothTransformation))

    def resizeEvent(self, event):
        self._refresh()
        super().resizeEvent(event)

    def clear(self):
        self._pix = None
        self.setText("暂无图像")


# ---------------------------------------------------------------- 推理线程
class InferWorker(QThread):
    """后台推理线程：负责跑模型，通过信号把结果送回主线程"""

    sig_frame = pyqtSignal(object)            # 标注后的 BGR 图像
    sig_detail = pyqtSignal(dict)             # 单张图片的检测明细
    sig_stats = pyqtSignal(dict, float)       # 累计分类计数, 当前FPS
    sig_progress = pyqtSignal(int, int, str)  # 当前, 总数, 描述
    sig_log = pyqtSignal(str)
    sig_done = pyqtSignal(dict)               # 汇总结果
    sig_error = pyqtSignal(str)

    def __init__(self, model, mode, source, params, out_dir, save):
        super().__init__()
        self.model = model
        self.mode = mode          # 'image' | 'folder' | 'video'
        self.source = source
        self.params = params      # {conf, iou, imgsz, device}
        self.out_dir = Path(out_dir)
        self.save = save
        self._stop = False

        self.class_names = {int(k): str(v) for k, v in model.names.items()}
        self.counts = {name: 0 for name in self.class_names.values()}
        self.records = []         # 每张图片的明细
        self.frame_idx = 0
        self.saved_files = []

    def stop(self):
        self._stop = True

    # ------------------------------------------------------------ 主流程
    def run(self):
        try:
            if self.mode == "video":
                self._run_video()
            else:
                self._run_images()
        except Exception as exc:  # noqa: BLE001
            self.sig_error.emit(f"{exc}\n{traceback.format_exc()}")
        finally:
            summary = {
                "mode": self.mode,
                "counts": dict(self.counts),
                "frames": self.frame_idx,
                "records": self.records,
                "saved_files": self.saved_files,
                "out_dir": str(self.out_dir),
                "stopped": self._stop,
            }
            self.sig_done.emit(summary)

    # ------------------------------------------------------------ 图片 / 文件夹
    def _collect_images(self):
        if self.mode == "image":
            return [Path(self.source)]
        files = []
        for p in sorted(Path(self.source).iterdir()):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                files.append(p)
        return files

    def _run_images(self):
        files = self._collect_images()
        total = len(files)
        if total == 0:
            self.sig_log.emit("未找到可处理的图片")
            return

        out_dir = self.out_dir
        if self.save:
            out_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        for i, f in enumerate(files):
            if self._stop:
                self.sig_log.emit("已手动停止")
                break
            res = self.model.predict(
                source=str(f),
                conf=self.params["conf"],
                iou=self.params["iou"],
                imgsz=self.params["imgsz"],
                device=self.params["device"],
                verbose=False,
            )[0]

            annotated = res.plot()
            record = self._parse_result(res, f)
            self.records.append(record)
            for name, c in record["counts"].items():
                self.counts[name] += c

            if self.save:
                out_path = out_dir / f.name
                cv2.imwrite(str(out_path), annotated)
                self.saved_files.append(str(out_path))

            elapsed = max(time.time() - t0, 1e-6)
            self.sig_frame.emit(annotated)
            self.sig_detail.emit(record)
            self.sig_stats.emit(dict(self.counts), (i + 1) / elapsed)
            self.sig_progress.emit(i + 1, total, f.name)

    # ------------------------------------------------------------ 视频
    def _run_video(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.sig_log.emit("无法打开视频文件")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 0 else 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if self.save:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            video_name = Path(self.source).stem + "_result.mp4"
            out_path = self.out_dir / video_name
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
            self.saved_files.append(str(out_path))

        last_t = time.time()
        self.frame_idx = 0
        while not self._stop:
            ret, frame = cap.read()
            if not ret:
                break

            res = self.model.predict(
                source=frame,
                conf=self.params["conf"],
                iou=self.params["iou"],
                imgsz=self.params["imgsz"],
                device=self.params["device"],
                verbose=False,
            )[0]

            annotated = res.plot()
            for name, c in self._parse_result(res, None, frame_only=True)["counts"].items():
                self.counts[name] += c

            if writer is not None:
                writer.write(annotated)

            now = time.time()
            cur_fps = 1.0 / max(now - last_t, 1e-6)
            last_t = now

            self.frame_idx += 1
            self.sig_frame.emit(annotated)
            self.sig_stats.emit(dict(self.counts), cur_fps)
            self.sig_progress.emit(self.frame_idx, total, "视频检测中")

        cap.release()
        if writer is not None:
            writer.release()
        if self._stop:
            self.sig_log.emit("已手动停止")

    # ------------------------------------------------------------ 结果解析
    def _parse_result(self, res, file_path, frame_only=False):
        boxes = []
        counts = {name: 0 for name in self.class_names.values()}
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), cf, cid in zip(xyxy, confs, clss):
                name = self.class_names.get(int(cid), str(cid))
                counts[name] = counts.get(name, 0) + 1
                if not frame_only:
                    boxes.append({
                        "cls": name,
                        "conf": float(cf),
                        "x1": float(x1), "y1": float(y1),
                        "x2": float(x2), "y2": float(y2),
                    })
        return {
            "file": str(file_path) if file_path else "",
            "boxes": boxes,
            "counts": counts,
        }


# ---------------------------------------------------------------- 主界面
class PcbMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PCB 缺陷检测系统 — YOLOv8")
        self.resize(1280, 800)

        self.model = None
        self.worker = None
        self.counts_total = {}
        self.records = []
        self.summary = {}

        # 文件夹逐张检测状态
        self.folder_files = []
        self.folder_idx = -1
        self._manual_run = False   # 当前这次推理是否为手动逐张检测

        self._build_ui()
        self._connect()

    # ------------------------------------------------------------ 界面搭建
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([330, 950])

        # 底部状态栏
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(240)
        self.progress.setValue(0)
        self.status_text = QLabel("就绪")
        self.statusBar().addWidget(self.status_text, 1)
        self.statusBar().addPermanentWidget(self.progress)
        self.fps_label = QLabel("FPS: -")
        self.statusBar().addPermanentWidget(self.fps_label)

    def _build_left_panel(self):
        panel = QWidget()
        panel.setMinimumWidth(320)
        panel.setMaximumWidth(400)
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        # ---- 模型
        gb_model = QGroupBox("模型")
        m_lay = QVBoxLayout(gb_model)
        self.model_path_edit = QLineEdit(str(DEFAULT_MODEL))
        self.model_path_edit.setReadOnly(True)
        row = QHBoxLayout()
        row.addWidget(self.model_path_edit)
        self.btn_browse_model = QPushButton("…")
        self.btn_browse_model.setFixedWidth(32)
        row.addWidget(self.btn_browse_model)
        m_lay.addLayout(row)
        self.btn_load_model = QPushButton("加载模型")
        m_lay.addWidget(self.btn_load_model)
        lay.addWidget(gb_model)

        # ---- 输入源
        gb_src = QGroupBox("输入源")
        s_lay = QVBoxLayout(gb_src)
        r_row = QHBoxLayout()
        self.rb_image = QRadioButton("单张图片")
        self.rb_folder = QRadioButton("文件夹")
        self.rb_video = QRadioButton("视频文件")
        self.rb_image.setChecked(True)
        self.mode_group = QButtonGroup(self)
        for rb in (self.rb_image, self.rb_folder, self.rb_video):
            self.mode_group.addButton(rb)
            r_row.addWidget(rb)
        s_lay.addLayout(r_row)

        row2 = QHBoxLayout()
        self.src_path_edit = QLineEdit()
        self.src_path_edit.setPlaceholderText("请选择图片 / 文件夹 / 视频")
        row2.addWidget(self.src_path_edit)
        self.btn_browse_src = QPushButton("浏览")
        self.btn_browse_src.setFixedWidth(56)
        row2.addWidget(self.btn_browse_src)
        s_lay.addLayout(row2)
        lay.addWidget(gb_src)

        # ---- 逐张检测（仅文件夹模式显示）
        self.gb_step = QGroupBox("逐张检测")
        st_lay = QVBoxLayout(self.gb_step)
        self.step_label = QLabel("未加载文件夹")
        self.step_label.setWordWrap(True)
        st_lay.addWidget(self.step_label)
        st_row = QHBoxLayout()
        self.btn_prev = QPushButton("◀ 上一张")
        self.btn_detect_one = QPushButton("检测当前张")
        self.btn_next = QPushButton("下一张 ▶")
        st_row.addWidget(self.btn_prev)
        st_row.addWidget(self.btn_detect_one)
        st_row.addWidget(self.btn_next)
        st_lay.addLayout(st_row)
        lay.addWidget(self.gb_step)
        self.gb_step.setVisible(False)

        # ---- 参数
        gb_param = QGroupBox("检测参数")
        p_lay = QFormLayout(gb_param)
        p_lay.setLabelAlignment(Qt.AlignRight)

        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(1, 100)
        self.conf_slider.setValue(40)
        self.conf_label = QLabel("0.40")
        p_lay.addRow("置信度 conf", self._row(self.conf_slider, self.conf_label))

        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setRange(1, 100)
        self.iou_slider.setValue(45)
        self.iou_label = QLabel("0.45")
        p_lay.addRow("IoU 阈值", self._row(self.iou_slider, self.iou_label))

        self.imgsz_combo = QComboBox()
        self.imgsz_combo.addItems(["416", "320", "530", "640"])
        p_lay.addRow("输入尺寸 imgsz", self.imgsz_combo)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["GPU (cuda:0)", "CPU"])
        p_lay.addRow("推理设备", self.device_combo)

        self.save_check = QCheckBox("保存检测结果")
        self.save_check.setChecked(True)
        p_lay.addRow("", self.save_check)
        lay.addWidget(gb_param)

        # ---- 输出
        gb_out = QGroupBox("输出目录")
        o_lay = QVBoxLayout(gb_out)
        row3 = QHBoxLayout()
        self.out_path_edit = QLineEdit(str(DEFAULT_OUTPUT))
        row3.addWidget(self.out_path_edit)
        self.btn_browse_out = QPushButton("…")
        self.btn_browse_out.setFixedWidth(32)
        row3.addWidget(self.btn_browse_out)
        o_lay.addLayout(row3)
        self.btn_open_out = QPushButton("打开结果目录")
        o_lay.addWidget(self.btn_open_out)
        lay.addWidget(gb_out)

        # ---- 操作按钮
        op_row = QHBoxLayout()
        self.btn_start = QPushButton("开始检测")
        self.btn_start.setMinimumHeight(38)
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setMinimumHeight(38)
        self.btn_stop.setEnabled(False)
        op_row.addWidget(self.btn_start)
        op_row.addWidget(self.btn_stop)
        lay.addLayout(op_row)

        self.btn_export = QPushButton("导出统计 CSV")
        lay.addWidget(self.btn_export)

        lay.addStretch(1)
        return panel

    @staticmethod
    def _row(widget, label, width=44):
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(widget)
        label.setFixedWidth(width)
        label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        h.addWidget(label)
        return w

    def _build_right_panel(self):
        vsplit = QSplitter(Qt.Vertical)

        self.image_view = ImageView()
        vsplit.addWidget(self.image_view)

        self.tabs = QTabWidget()
        # 统计表
        self.table_stats = QTableWidget(0, 2)
        self.table_stats.setHorizontalHeaderLabels(["缺陷类别", "数量"])
        self._setup_table(self.table_stats)
        self.tabs.addTab(self.table_stats, "缺陷统计")

        # 明细表
        self.table_detail = QTableWidget(0, 7)
        self.table_detail.setHorizontalHeaderLabels(
            ["来源文件", "类别", "置信度", "x1", "y1", "x2", "y2"])
        self._setup_table(self.table_detail)
        self.tabs.addTab(self.table_detail, "检测明细")

        # 日志
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setStyleSheet("font-family: Consolas, monospace;")
        self.tabs.addTab(self.log_edit, "运行日志")

        vsplit.addWidget(self.tabs)
        vsplit.setSizes([560, 260])
        return vsplit

    @staticmethod
    def _setup_table(table):
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.horizontalHeader().setStretchLastSection(True)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

    # ------------------------------------------------------------ 信号连接
    def _connect(self):
        self.btn_browse_model.clicked.connect(self.on_browse_model)
        self.btn_browse_src.clicked.connect(self.on_browse_src)
        self.btn_browse_out.clicked.connect(self.on_browse_out)
        self.btn_open_out.clicked.connect(self.on_open_out)
        self.btn_load_model.clicked.connect(self.on_load_model)
        self.btn_start.clicked.connect(self.on_start)
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_export.clicked.connect(self.on_export)
        self.btn_prev.clicked.connect(self.on_prev)
        self.btn_next.clicked.connect(self.on_next)
        self.btn_detect_one.clicked.connect(self.on_detect_current)

        self.conf_slider.valueChanged.connect(
            lambda v: self.conf_label.setText(f"{v / 100:.2f}"))
        self.iou_slider.valueChanged.connect(
            lambda v: self.iou_label.setText(f"{v / 100:.2f}"))
        self.mode_group.buttonClicked.connect(self.on_mode_changed)

    # ------------------------------------------------------------ 界面事件
    def on_browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择模型权重", str(BASE_DIR), "PyTorch 模型 (*.pt *.onnx);;所有文件 (*)")
        if path:
            self.model_path_edit.setText(path)

    def on_browse_src(self):
        if self.rb_image.isChecked():
            path, _ = QFileDialog.getOpenFileName(
                self, "选择图片", str(BASE_DIR), "图片 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp)")
        elif self.rb_folder.isChecked():
            path = QFileDialog.getExistingDirectory(self, "选择文件夹", str(BASE_DIR))
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, "选择视频", str(BASE_DIR), "视频 (*.mp4 *.avi *.mov *.mkv *.wmv *.flv)")
        if path:
            self.src_path_edit.setText(path)
            if self.rb_folder.isChecked():
                self._load_folder(path)

    def on_browse_out(self):
        path = QFileDialog.getExistingDirectory(self, "选择输出目录", self.out_path_edit.text())
        if path:
            self.out_path_edit.setText(path)

    def on_open_out(self):
        out = Path(self.out_path_edit.text())
        if out.exists():
            os.startfile(str(out))
        else:
            QMessageBox.information(self, "提示", "输出目录不存在，检测后会自动创建。")

    def on_mode_changed(self):
        if self.rb_image.isChecked():
            self.src_path_edit.setPlaceholderText("请选择图片文件")
        elif self.rb_folder.isChecked():
            self.src_path_edit.setPlaceholderText("请选择图片文件夹")
        else:
            self.src_path_edit.setPlaceholderText("请选择视频文件")

        # 逐张检测面板只在文件夹模式显示
        is_folder = self.rb_folder.isChecked()
        self.gb_step.setVisible(is_folder)
        if not is_folder:
            self.folder_files = []
            self.folder_idx = -1
            self.step_label.setText("未加载文件夹")
        self._sync_nav()

    # ------------------------------------------------------------ 文件夹逐张检测
    @staticmethod
    def _imread_unicode(path):
        """读取中文路径图片（避开 OpenCV 非 ASCII 路径问题）"""
        try:
            data = np.fromfile(str(path), dtype=np.uint8)
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception:  # noqa: BLE001
            return None

    def _load_folder(self, path):
        """载入文件夹中的图片列表，默认停在第 1 张（不自动检测）"""
        files = [p for p in sorted(Path(path).iterdir())
                 if p.is_file() and p.suffix.lower() in IMG_EXTS]
        if not files:
            QMessageBox.warning(self, "提示", "该文件夹下没有可识别的图片。")
            self.folder_files = []
            self.folder_idx = -1
            self.step_label.setText("未加载文件夹")
            self._sync_nav()
            return

        self.folder_files = files
        self.folder_idx = 0
        # 新会话：清空上一轮的统计与明细
        self.counts_total = ({str(v): 0 for v in self.model.names.values()}
                             if self.model is not None else {})
        self.records = []
        self.summary = {}
        self.table_detail.setRowCount(0)
        self.progress.setValue(0)
        self._refresh_stats_table()
        self._show_folder_current()
        self.log(f"已加载文件夹，共 {len(files)} 张图片，请用「上一张 / 下一张」选择后点「检测当前张」")
        self._sync_nav()

    def _show_folder_current(self):
        """显示当前选中图片（原图，不含检测框）"""
        if not self.folder_files or self.folder_idx < 0:
            return
        p = self.folder_files[self.folder_idx]
        img = self._imread_unicode(p)
        if img is not None:
            self.image_view.set_image(img)
        else:
            self.image_view.clear()
        self.step_label.setText(
            f"当前：{p.name}\n({self.folder_idx + 1} / {len(self.folder_files)})")
        self.status_text.setText(
            f"当前图片：{p.name}  ({self.folder_idx + 1}/{len(self.folder_files)})")

    def on_prev(self):
        if not self.folder_files:
            return
        self.folder_idx = (self.folder_idx - 1) % len(self.folder_files)
        self._show_folder_current()

    def on_next(self):
        if not self.folder_files:
            return
        self.folder_idx = (self.folder_idx + 1) % len(self.folder_files)
        self._show_folder_current()

    def on_detect_current(self):
        """只检测当前显示的这一张图片"""
        if self.model is None:
            QMessageBox.warning(self, "提示", "请先加载模型。")
            return
        if not self.folder_files or self.folder_idx < 0:
            QMessageBox.warning(self, "提示", "请先加载图片文件夹。")
            return
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(self, "提示", "请等待当前检测结束。")
            return

        src = self.folder_files[self.folder_idx]
        self._manual_run = True
        self.log(f"逐张检测：{src.name}")
        # 逐张模式直接写入输出目录，不再每次新建时间戳子目录
        worker = InferWorker(self.model, "image", str(src), self._current_params(),
                             Path(self.out_path_edit.text()), self.save_check.isChecked())
        self._start_worker(worker)

    def _sync_nav(self):
        running = self.worker is not None and self.worker.isRunning()
        ok = (not running) and bool(self.folder_files)
        for btn in (self.btn_prev, self.btn_next, self.btn_detect_one):
            btn.setEnabled(ok)

    def _current_params(self):
        return {
            "conf": self.conf_slider.value() / 100.0,
            "iou": self.iou_slider.value() / 100.0,
            "imgsz": int(self.imgsz_combo.currentText()),
            "device": 0 if self.device_combo.currentIndex() == 0 else "cpu",
        }

    def _start_worker(self, worker):
        worker.sig_frame.connect(self.on_frame)
        worker.sig_detail.connect(self.on_detail)
        worker.sig_stats.connect(self.on_stats)
        worker.sig_progress.connect(self.on_progress)
        worker.sig_log.connect(self.log)
        worker.sig_done.connect(self.on_done)
        worker.sig_error.connect(self.on_error)
        self.worker = worker
        worker.start()
        self._set_running(True)

    def on_load_model(self):
        path = self.model_path_edit.text().strip()
        if not path or not Path(path).exists():
            QMessageBox.warning(self, "错误", "模型文件不存在，请重新选择。")
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.model = YOLO(path)
            names = ", ".join(str(v) for v in self.model.names.values())
            self.log(f"模型加载成功：{path}")
            self.log(f"类别（{len(self.model.names)}）：{names}")
            self.status_text.setText(f"模型已加载 — {Path(path).name}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "错误", f"模型加载失败：\n{exc}")
        finally:
            QApplication.restoreOverrideCursor()

    def current_mode(self):
        if self.rb_folder.isChecked():
            return "folder"
        if self.rb_video.isChecked():
            return "video"
        return "image"

    def on_start(self):
        if self.model is None:
            QMessageBox.warning(self, "提示", "请先加载模型。")
            return
        src = self.src_path_edit.text().strip()
        if not src or not Path(src).exists():
            QMessageBox.warning(self, "提示", "请先选择有效的输入源。")
            return

        mode = self.current_mode()
        if mode == "folder" and not self.folder_files:
            QMessageBox.warning(self, "提示", "请先加载图片文件夹（点「浏览」选择）。")
            return
        if mode == "image" and Path(src).suffix.lower() not in IMG_EXTS:
            QMessageBox.warning(self, "提示", "所选文件不是受支持的图片格式。")
            return
        if mode == "video" and Path(src).suffix.lower() not in VIDEO_EXTS:
            QMessageBox.warning(self, "提示", "所选文件不是受支持的视频格式。")
            return

        # 每次检测输出到带时间戳的子目录，避免互相覆盖
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(self.out_path_edit.text()) / f"detect_{stamp}"

        params = self._current_params()

        # 批量检测：重置界面数据
        self._manual_run = False
        self.counts_total = {str(v): 0 for v in self.model.names.values()}
        self.records = []
        self.table_detail.setRowCount(0)
        self.summary = {}
        self.progress.setValue(0)
        self.fps_label.setText("FPS: -")
        if mode == "folder":
            self.log(f"开始批量检测文件夹（{len(self.folder_files)} 张）| 参数={params}")
        else:
            self.log(f"开始检测 | 模式={mode} | 参数={params}")

        worker = InferWorker(
            self.model, mode, src, params, out_dir, self.save_check.isChecked())
        self._start_worker(worker)

    def on_stop(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.log("正在停止…")

    def _set_running(self, running):
        self.btn_start.setEnabled(not running)
        self.btn_stop.setEnabled(running)
        self.btn_load_model.setEnabled(not running)
        self.btn_export.setEnabled(not running)
        self._sync_nav()

    # ------------------------------------------------------------ 线程回调
    def on_frame(self, bgr):
        self.image_view.set_image(bgr)

    def on_detail(self, record):
        self.records.append(record)
        for box in record["boxes"]:
            r = self.table_detail.rowCount()
            self.table_detail.insertRow(r)
            filename = Path(record["file"]).name if record["file"] else "-"
            cells = [filename, box["cls"], f"{box['conf']:.3f}",
                     f"{box['x1']:.0f}", f"{box['y1']:.0f}",
                     f"{box['x2']:.0f}", f"{box['y2']:.0f}"]
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if c not in (0, 1):
                    item.setTextAlignment(Qt.AlignCenter)
                self.table_detail.setItem(r, c, item)
        self.table_detail.scrollToBottom()

    def on_stats(self, counts, fps):
        if self._manual_run:
            # 逐张手动检测：把当前这单张的计数累加到总数上
            for name, c in counts.items():
                self.counts_total[name] = self.counts_total.get(name, 0) + c
        else:
            self.counts_total = counts
        self._refresh_stats_table()
        self.fps_label.setText(f"FPS: {fps:.1f}")

    def _refresh_stats_table(self):
        self.table_stats.setRowCount(0)
        total = 0
        for name, cnt in self.counts_total.items():
            r = self.table_stats.rowCount()
            self.table_stats.insertRow(r)
            self.table_stats.setItem(r, 0, QTableWidgetItem(name))
            item = QTableWidgetItem(str(cnt))
            item.setTextAlignment(Qt.AlignCenter)
            self.table_stats.setItem(r, 1, item)
            total += cnt
        # 合计行
        r = self.table_stats.rowCount()
        self.table_stats.insertRow(r)
        self.table_stats.setItem(r, 0, QTableWidgetItem("合计"))
        item = QTableWidgetItem(str(total))
        item.setTextAlignment(Qt.AlignCenter)
        self.table_stats.setItem(r, 1, item)

    def on_progress(self, cur, total, text):
        if total > 0:
            self.progress.setMaximum(total)
            self.progress.setValue(min(cur, total))
            self.status_text.setText(f"{text}  ({cur}/{total})")
        else:
            self.progress.setMaximum(0)  # 忙碌状态
            self.status_text.setText(f"{text}  已处理 {cur} 帧")

    def on_done(self, summary):
        self.summary = summary
        self._set_running(False)
        total = sum(summary.get("counts", {}).values())
        if summary.get("stopped"):
            self.status_text.setText("已停止")
        elif self._manual_run:
            # 逐张模式：回到当前图片浏览状态，方便继续切下一张
            self.progress.setValue(self.progress.maximum())
            name = self.folder_files[self.folder_idx].name if self.folder_files else ""
            self.status_text.setText(
                f"当前图片已检测：{name}  ({self.folder_idx + 1}/{len(self.folder_files)})")
            self.log(f"本张检测完成 | 缺陷数 {total}")
        else:
            self.status_text.setText("检测完成")
            self.progress.setValue(self.progress.maximum())
            self.log(f"检测结束 | 共 {summary.get('frames', 0)} 帧/张 | 缺陷总数 {total}")
        self._manual_run = False
        if summary.get("saved_files"):
            self.log(f"结果已保存至：{summary.get('out_dir')}")

    def on_error(self, msg):
        self._manual_run = False
        self._set_running(False)
        self.log(f"[错误] {msg}")
        QMessageBox.critical(self, "运行错误", msg)

    # ------------------------------------------------------------ 导出
    def on_export(self):
        if not self.records and not self.summary:
            QMessageBox.information(self, "提示", "暂无检测结果可导出。")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出 CSV", str(Path(self.out_path_edit.text()) / "detect_result.csv"),
            "CSV 文件 (*.csv)")
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as fp:
                writer = csv.writer(fp)
                writer.writerow(["=== 缺陷统计 ==="])
                writer.writerow(["类别", "数量"])
                for name, cnt in self.counts_total.items():
                    writer.writerow([name, cnt])
                writer.writerow([])
                writer.writerow(["=== 检测明细 ==="])
                writer.writerow(["来源文件", "类别", "置信度", "x1", "y1", "x2", "y2"])
                for rec in self.records:
                    for b in rec["boxes"]:
                        writer.writerow([Path(rec["file"]).name, b["cls"],
                                         f"{b['conf']:.4f}", f"{b['x1']:.1f}",
                                         f"{b['y1']:.1f}", f"{b['x2']:.1f}", f"{b['y2']:.1f}"])
                if self.summary.get("mode") == "video":
                    writer.writerow([])
                    writer.writerow(["=== 视频累计统计 ==="])
                    for name, cnt in self.summary.get("counts", {}).items():
                        writer.writerow([name, cnt])
            self.log(f"CSV 已导出：{path}")
            QMessageBox.information(self, "成功", f"已导出：\n{path}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "错误", f"导出失败：\n{exc}")

    # ------------------------------------------------------------ 工具
    def log(self, text):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_edit.append(f"[{ts}] {text}")

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(3000)
        event.accept()


# ---------------------------------------------------------------- 入口
def main():
    app = QApplication(sys.argv)
    win = PcbMainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
