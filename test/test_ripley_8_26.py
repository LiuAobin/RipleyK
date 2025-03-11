#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import math
from datetime import datetime
import numpy as np
import pandas as pd

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import (QMainWindow, QWidget, QLabel, QPushButton, QFileDialog,
                             QVBoxLayout, QHBoxLayout, QGridLayout, QLineEdit, QComboBox,
                             QProgressBar, QRadioButton, QGroupBox, QGraphicsView, QGraphicsScene,
                             QGraphicsPixmapItem, QGraphicsRectItem)
from PyQt5.QtCore import Qt, QRectF, pyqtSignal, QThread

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree
from shapely.geometry import Point, box
from pyproj import Transformer

########################################################################
# 定义辅助函数：计算圆与矩形区域交集面积所占比例（用于周长法边界修正）
########################################################################
def circle_region_intersection_fraction(p, r, region):
    """
    计算以点 p 为圆心，半径为 r 的圆与给定矩形区域的交集面积占圆面积的比例
    参数:
        p: (x, y) 坐标
        r: 圆半径
        region: (xmin, ymin, xmax, ymax) 矩形区域
    返回:
        交集面积比例（0~1）
    """
    circle = Point(p[0], p[1]).buffer(r, resolution=16)  # 利用 shapely 创建圆形多边形
    region_poly = box(region[0], region[1], region[2], region[3])
    inter_area = circle.intersection(region_poly).area
    return inter_area / (math.pi * r * r)

########################################################################
# 定义 Ripley 函数计算的工作线程类（继承 QThread）
########################################################################
class RipleyWorker(QThread):
    # 定义信号：progress_signal 用于更新进度条；result_signal 用于传递计算结果（K函数和L函数列表）
    progress_signal = pyqtSignal(int)
    result_signal = pyqtSignal(list, list)

    def __init__(self, points, times, region, r_min, r_max, r_steps,
                 t_min, t_max, correction_method, analysis_type):
        """
        初始化计算线程
        参数:
            points: np.array，形状 (n,2)，存储空间坐标数据
            times: list，存储每个点对应的时间（字符串形式），若仅进行空间分析可传入 None
            region: (xmin, ymin, xmax, ymax) 计算区域
            r_min: Ripley 函数半径最小值
            r_max: Ripley 函数半径最大值
            r_steps: 半径分成的步数
            t_min: 时空分析时的时间最小差值（秒）
            t_max: 时空分析时的时间最大差值（秒）
            correction_method: 边界修正方法，选项："超环面法"、"缓冲区法"、"周长法"
            analysis_type: 分析类型，"空间" 或 "时空"
        """
        super(RipleyWorker, self).__init__()
        self.points = points
        self.times = times
        self.region = region
        self.r_min = r_min
        self.r_max = r_max
        self.r_steps = r_steps
        self.t_min = t_min
        self.t_max = t_max
        self.correction_method = correction_method
        self.analysis_type = analysis_type

    def run(self):
        """
        在单独线程中计算 Ripley 函数，并通过信号发送计算进度和结果
        """
        xmin, ymin, xmax, ymax = self.region
        area = (xmax - xmin) * (ymax - ymin)
        lambda_est = len(self.points) / area  # 点密度估计

        # 根据用户输入的半径范围生成 r 值列表
        r_values = np.linspace(self.r_min, self.r_max, self.r_steps)
        K_values = []

        # 如果采用超环面法，则对数据进行周期性扩展
        if self.correction_method == "超环面法":
            extended_points = []
            index_mapping = []  # 记录扩展后每个点对应原始数据的索引
            width = xmax - xmin
            height = ymax - ymin
            # 对每个点检查是否靠近边界，若是则添加对应副本
            for i, p in enumerate(self.points):
                extended_points.append(p)
                index_mapping.append(i)
                if p[0] - xmin < self.r_max:
                    extended_points.append(np.array([p[0] + width, p[1]]))
                    index_mapping.append(i)
                if xmax - p[0] < self.r_max:
                    extended_points.append(np.array([p[0] - width, p[1]]))
                    index_mapping.append(i)
                if p[1] - ymin < self.r_max:
                    extended_points.append(np.array([p[0], p[1] + height]))
                    index_mapping.append(i)
                if ymax - p[1] < self.r_max:
                    extended_points.append(np.array([p[0], p[1] - height]))
                    index_mapping.append(i)
                # 处理角落：同时靠近水平和垂直边界的情况
                if (p[0] - xmin < self.r_max) and (p[1] - ymin < self.r_max):
                    extended_points.append(np.array([p[0] + width, p[1] + height]))
                    index_mapping.append(i)
                if (p[0] - xmin < self.r_max) and (ymax - p[1] < self.r_max):
                    extended_points.append(np.array([p[0] + width, p[1] - height]))
                    index_mapping.append(i)
                if (xmax - p[0] < self.r_max) and (p[1] - ymin < self.r_max):
                    extended_points.append(np.array([p[0] - width, p[1] + height]))
                    index_mapping.append(i)
                if (xmax - p[0] < self.r_max) and (ymax - p[1] < self.r_max):
                    extended_points.append(np.array([p[0] - width, p[1] - height]))
                    index_mapping.append(i)
            extended_points = np.array(extended_points)
            # 构造 kd-tree（加速邻域查询）
            tree = cKDTree(extended_points)
            # 针对每个 r 值计算 Ripley K 函数
            for idx, r in enumerate(r_values):
                total_count = 0.0
                valid_points = 0
                # 对于每个原始数据点 p，查询扩展点集中距离 p 在 r 范围内的所有点
                for i, p in enumerate(self.points):
                    indices = tree.query_ball_point(p, r)
                    count = 0
                    for j in indices:
                        # 如果扩展点 j 对应的原始点为 p 自身，则要排除自计
                        if index_mapping[j] == i:
                            # 如果扩展点的坐标与 p 完全相同，则为 p 自身
                            if np.allclose(extended_points[j], p):
                                continue
                            else:
                                count += 1
                        else:
                            count += 1
                    total_count += count
                    valid_points += 1
                if valid_points == 0:
                    K = 0
                else:
                    K = total_count / (lambda_est * valid_points)
                K_values.append(K)
                progress = int((idx + 1) / len(r_values) * 100)
                self.progress_signal.emit(progress)
            L_values = [math.sqrt(K / math.pi) for K in K_values]
            self.result_signal.emit(K_values, L_values)
            return  # 结束超环面法分支

        # 对于其他方法：缓冲区法、周长法或无修正（时空或空间分析）
        # 若为时空分析，则将时间转换为数值（这里采用时间戳，单位：秒）
        if self.analysis_type == "时空":
            times_numeric = []
            for t in self.times:
                try:
                    dt = datetime.strptime(t, "%Y/%m/%d")
                except Exception as e:
                    dt = datetime.strptime(t, "%Y-%m-%d")
                times_numeric.append(dt.timestamp())
            times_numeric = np.array(times_numeric)
        else:
            times_numeric = None

        # 构造 kd-tree（仅针对原始数据点）
        tree = cKDTree(self.points)
        for idx, r in enumerate(r_values):
            total_count = 0.0
            valid_points = 0
            # 对所有点同时查询邻域，返回列表，其中每个元素为该点在 r 范围内的邻居索引列表
            neighbors_list = tree.query_ball_point(self.points, r)
            for i, neighbors in enumerate(neighbors_list):
                p = self.points[i]
                # 若采用缓冲区法，则若 p 离边界不足 r，则跳过该点（不计入计算）
                if self.correction_method == "缓冲区法":
                    if (p[0] - xmin < r) or (xmax - p[0] < r) or (p[1] - ymin < r) or (ymax - p[1] < r):
                        continue
                count = 0.0
                # 遍历 p 的邻居点
                for j in neighbors:
                    if j == i:
                        continue  # 排除自身
                    # 若为时空分析，则只计入时间差在 [t_min, t_max] 范围内的邻居
                    if self.analysis_type == "时空":
                        time_diff = abs(times_numeric[j] - times_numeric[i])
                        if time_diff < self.t_min or time_diff > self.t_max:
                            continue
                    count += 1
                # 若采用周长法，则对计数进行权重修正（利用圆与区域的交集比例）
                if self.correction_method == "周长法":
                    fraction = circle_region_intersection_fraction(p, r, self.region)
                    if fraction == 0:
                        continue
                    count = count / fraction
                total_count += count
                valid_points += 1
            if valid_points == 0:
                K = 0
            else:
                K = total_count / (lambda_est * valid_points)
            K_values.append(K)
            progress = int((idx + 1) / len(r_values) * 100)
            self.progress_signal.emit(progress)
        L_values = [math.sqrt(K / math.pi) for K in K_values]
        self.result_signal.emit(K_values, L_values)

########################################################################
# 自定义地图显示控件，继承 QGraphicsView，实现地图的缩放、拖动以及区域选择功能
########################################################################
class MapGraphicsView(QGraphicsView):
    # 定义信号：当用户在地图上框选计算区域后，将区域参数传递给主窗口
    regionSelected = pyqtSignal(float, float, float, float)

    def __init__(self, parent=None):
        super(MapGraphicsView, self).__init__(parent)
        # 创建场景用于显示地图和数据点
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        # 地图图像项（QGraphicsPixmapItem）
        self.map_pixmap_item = None
        # 用于显示用户拖拽选择的矩形区域（rubber band 效果）
        self.rubberBand = None
        self.origin = QtCore.QPoint()
        self.setMouseTracking(True)  # 开启鼠标跟踪
        self.scaleFactor = 1.0  # 当前缩放因子

    def loadMap(self, map_path):
        """
        加载地图图像文件，并添加到场景中显示
        参数:
            map_path: 地图图像文件路径
        """
        pixmap = QtGui.QPixmap(map_path)
        self.map_pixmap_item = self.scene.addPixmap(pixmap)
        self.setSceneRect(QRectF(pixmap.rect()))

    def wheelEvent(self, event):
        """
        重写鼠标滚轮事件，实现地图缩放
        """
        if event.angleDelta().y() > 0:
            factor = 1.25
        else:
            factor = 0.8
        self.scale(factor, factor)
        self.scaleFactor *= factor

    def mousePressEvent(self, event):
        """
        重写鼠标按下事件，开始区域选择（绘制 rubber band）
        """
        if event.button() == Qt.LeftButton:
            self.origin = event.pos()
            if not self.rubberBand:
                self.rubberBand = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Rectangle, self)
            self.rubberBand.setGeometry(QtCore.QRect(self.origin, QtCore.QSize()))
            self.rubberBand.show()
        super(MapGraphicsView, self).mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """
        重写鼠标移动事件，更新 rubber band 的大小
        """
        if self.rubberBand:
            rect = QtCore.QRect(self.origin, event.pos()).normalized()
            self.rubberBand.setGeometry(rect)
        super(MapGraphicsView, self).mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """
        重写鼠标释放事件，确定选取区域，并发射 regionSelected 信号
        """
        if self.rubberBand:
            self.rubberBand.hide()
            # 将 rubber band 的视图坐标转换为场景坐标
            rect = self.rubberBand.geometry()
            topLeft = self.mapToScene(rect.topLeft())
            bottomRight = self.mapToScene(rect.bottomRight())
            xmin = topLeft.x()
            ymin = topLeft.y()
            xmax = bottomRight.x()
            ymax = bottomRight.y()
            self.regionSelected.emit(xmin, ymin, xmax, ymax)
        super(MapGraphicsView, self).mouseReleaseEvent(event)

    def addPoint(self, x, y):
        """
        在地图上添加一个表示数据点的小红点
        参数:
            x, y: 数据点的坐标
        返回:
            QGraphicsEllipseItem 对象
        """
        radius = 3
        ellipse = self.scene.addEllipse(x - radius, y - radius, radius*2, radius*2,
                                        QtGui.QPen(Qt.red), QtGui.QBrush(Qt.red))
        return ellipse

########################################################################
# 主窗口类，构建整个用户界面
########################################################################
class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        # 设置窗口标题和初始大小
        self.setWindowTitle("Ripley 函数空间和时空分析程序")
        self.resize(1200, 800)

        # 主部件和布局
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QHBoxLayout(self.main_widget)

        # 左侧控制面板
        self.control_panel = QWidget()
        self.control_layout = QVBoxLayout(self.control_panel)
        self.main_layout.addWidget(self.control_panel, 0)

        # 添加数据文件读取按钮
        self.load_button = QPushButton("读取数据文件")
        self.load_button.clicked.connect(self.load_data)
        self.control_layout.addWidget(self.load_button)

        # 显示文件路径标签
        self.file_label = QLabel("未加载数据文件")
        self.control_layout.addWidget(self.file_label)

        # 参数设置区域（半径、时间等）
        self.param_group = QGroupBox("参数设置")
        self.param_layout = QGridLayout(self.param_group)
        self.control_layout.addWidget(self.param_group)

        # 半径参数：最小值、最大值和步数
        self.label_r_min = QLabel("半径 r 最小值:")
        self.edit_r_min = QLineEdit("0")
        self.param_layout.addWidget(self.label_r_min, 0, 0)
        self.param_layout.addWidget(self.edit_r_min, 0, 1)
        self.label_r_max = QLabel("半径 r 最大值:")
        self.edit_r_max = QLineEdit("1000")
        self.param_layout.addWidget(self.label_r_max, 0, 2)
        self.param_layout.addWidget(self.edit_r_max, 0, 3)
        self.label_r_steps = QLabel("半径步数:")
        self.edit_r_steps = QLineEdit("50")
        self.param_layout.addWidget(self.label_r_steps, 0, 4)

        # 时间参数（仅用于时空分析）：最小差值和最大差值（单位：秒）
        self.label_t_min = QLabel("时间 t 最小差值:")
        self.edit_t_min = QLineEdit("0")
        self.param_layout.addWidget(self.label_t_min, 1, 0)
        self.label_t_max = QLabel("时间 t 最大差值:")
        self.edit_t_max = QLineEdit("86400")  # 默认 1 天 = 86400 秒
        self.param_layout.addWidget(self.label_t_max, 1, 1)

        # 计算区域设置（手动输入）
        self.region_group = QGroupBox("计算区域设置")
        self.region_layout = QGridLayout(self.region_group)
        self.control_layout.addWidget(self.region_group)

        self.label_xmin = QLabel("区域 xmin:")
        self.edit_xmin = QLineEdit("300000")
        self.region_layout.addWidget(self.label_xmin, 0, 0)
        self.region_layout.addWidget(self.edit_xmin, 0, 1)
        self.label_ymin = QLabel("区域 ymin:")
        self.edit_ymin = QLineEdit("3400000")
        self.region_layout.addWidget(self.label_ymin, 0, 2)
        self.region_layout.addWidget(self.edit_ymin, 0, 3)
        self.label_xmax = QLabel("区域 xmax:")
        self.edit_xmax = QLineEdit("400000")
        self.region_layout.addWidget(self.label_xmax, 1, 0)
        self.label_ymax = QLabel("区域 ymax:")
        self.edit_ymax = QLineEdit("3500000")
        self.region_layout.addWidget(self.label_ymax, 1, 1)

        # 选择边界修正方法下拉框
        self.label_correction = QLabel("边界修正方法:")
        self.combo_correction = QComboBox()
        self.combo_correction.addItems(["超环面法", "缓冲区法", "周长法"])
        self.control_layout.addWidget(self.label_correction)
        self.control_layout.addWidget(self.combo_correction)

        # 选择分析类型（空间或时空）
        self.analysis_group = QGroupBox("分析类型")
        self.analysis_layout = QHBoxLayout(self.analysis_group)
        self.radio_spatial = QRadioButton("空间")
        self.radio_spatial.setChecked(True)
        self.radio_spatiotemporal = QRadioButton("时空")
        self.analysis_layout.addWidget(self.radio_spatial)
        self.analysis_layout.addWidget(self.radio_spatiotemporal)
        self.control_layout.addWidget(self.analysis_group)

        # 开始计算按钮
        self.start_button = QPushButton("开始计算")
        self.start_button.clicked.connect(self.start_calculation)
        self.control_layout.addWidget(self.start_button)

        # 进度条显示计算进度
        self.progress_bar = QProgressBar()
        self.control_layout.addWidget(self.progress_bar)

        # 右侧显示区域：上部为地图，下部为 Matplotlib 绘图区域（显示 K 函数和 L 函数曲线）
        self.display_panel = QWidget()
        self.display_layout = QVBoxLayout(self.display_panel)
        self.main_layout.addWidget(self.display_panel, 1)

        # 地图显示控件
        self.map_view = MapGraphicsView()
        self.map_view.setMinimumHeight(300)
        self.display_layout.addWidget(self.map_view)
        # 当用户在地图上框选区域时，更新手动输入的区域参数
        self.map_view.regionSelected.connect(self.update_region_from_map)

        # Matplotlib 绘图区域
        self.figure, self.axs = plt.subplots(1, 2, figsize=(10, 4))
        self.canvas = FigureCanvas(self.figure)
        self.display_layout.addWidget(self.canvas)

        # 数据存储变量
        self.data_points = None    # np.array 存储 (n,2) 空间数据
        self.data_times = None     # list 存储时间字符串
        self.data_loaded = False

        # 加载上海市地图文件，假设文件名为 "shanghai_map.png" 位于当前目录
        map_path = "shanghai_map.png"
        if os.path.exists(map_path):
            self.map_view.loadMap(map_path)
        else:
            QtWidgets.QMessageBox.warning(self, "提示", "未找到上海市地图文件：shanghai_map.png")

        # 构造坐标转换器，将地图坐标（EPSG:4326）转换为数据坐标系（这里假设数据坐标系为 EPSG:3857，
        # 如有需要请修改为实际坐标系），保证数据点与地图显示一致
        self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    def load_data(self):
        """
        打开文件对话框，读取 CSV 数据文件，并在地图上显示数据点
        """
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "选择数据文件", "",
                                                  "CSV Files (*.csv);;All Files (*)", options=options)
        if fileName:
            try:
                # 利用 pandas 读取 CSV 文件
                df = pd.read_csv(fileName)
                # 检查必须的列
                if not set(["longitude", "latitude", "Date"]).issubset(df.columns):
                    QtWidgets.QMessageBox.warning(self, "错误", "数据文件中缺少必要的列：longitude, latitude, Date")
                    return
                # 提取数据点和时间
                self.data_points = df[["longitude", "latitude"]].to_numpy()
                self.data_times = df["Date"].astype(str).tolist()
                self.data_loaded = True
                self.file_label.setText(f"已加载数据文件: {fileName}")

                # 在地图上显示每个数据点
                # 如有需要，可进行坐标转换（例如：若数据为 EPSG:3857，而地图为 EPSG:4326）
                for point in self.data_points:
                    # 此处默认数据点已在地图坐标系中，若不一致请调用 transformer.transform(...)
                    self.map_view.addPoint(point[0], point[1])
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "错误", f"读取数据文件失败：{e}")

    def update_region_from_map(self, xmin, ymin, xmax, ymax):
        """
        当用户在地图上框选计算区域后，将选取的区域更新到手动输入区域中
        参数:
            xmin, ymin, xmax, ymax: 框选区域的坐标
        """
        self.edit_xmin.setText(str(int(xmin)))
        self.edit_ymin.setText(str(int(ymin)))
        self.edit_xmax.setText(str(int(xmax)))
        self.edit_ymax.setText(str(int(ymax)))

    def start_calculation(self):
        """
        点击“开始计算”按钮后，读取用户输入参数，创建工作线程计算 Ripley 函数
        """
        if not self.data_loaded:
            QtWidgets.QMessageBox.warning(self, "错误", "请先加载数据文件")
            return

        try:
            r_min = float(self.edit_r_min.text())
            r_max = float(self.edit_r_max.text())
            r_steps = int(self.edit_r_steps.text())
            t_min = float(self.edit_t_min.text())
            t_max = float(self.edit_t_max.text())
            xmin = float(self.edit_xmin.text())
            ymin = float(self.edit_ymin.text())
            xmax = float(self.edit_xmax.text())
            ymax = float(self.edit_ymax.text())
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "错误", "参数输入有误，请检查")
            return

        region = (xmin, ymin, xmax, ymax)

        # 判断分析类型：空间或时空
        if self.radio_spatial.isChecked():
            analysis_type = "空间"
        else:
            analysis_type = "时空"

        # 获取边界修正方法
        correction_method = self.combo_correction.currentText()

        # 创建工作线程并传递参数，连接进度和结果信号
        self.worker = RipleyWorker(self.data_points, self.data_times, region,
                                   r_min, r_max, r_steps, t_min, t_max,
                                   correction_method, analysis_type)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.result_signal.connect(self.handle_result)
        self.worker.start()

    def update_progress(self, value):
        """
        更新进度条显示
        参数:
            value: 进度百分比
        """
        self.progress_bar.setValue(value)

    def handle_result(self, K_values, L_values):
        """
        当计算完成后，在绘图区域中显示 K 函数和 L 函数曲线
        参数:
            K_values: K 函数值列表
            L_values: L 函数值列表
        """
        r_min = float(self.edit_r_min.text())
        r_max = float(self.edit_r_max.text())
        r_steps = int(self.edit_r_steps.text())
        r_values = np.linspace(r_min, r_max, r_steps)

        # 清空之前的图像
        self.axs[0].clear()
        self.axs[1].clear()

        # 绘制 K 函数曲线
        self.axs[0].plot(r_values, K_values, marker='o', linestyle='-')
        self.axs[0].set_title("K 函数曲线")
        self.axs[0].set_xlabel("r")
        self.axs[0].set_ylabel("K(r)")

        # 绘制 L 函数曲线
        self.axs[1].plot(r_values, L_values, marker='o', linestyle='-')
        self.axs[1].set_title("L 函数曲线")
        self.axs[1].set_xlabel("r")
        self.axs[1].set_ylabel("L(r)")

        self.figure.tight_layout()
        self.canvas.draw()

########################################################################
# 主函数入口
########################################################################
def main():
    app = QtWidgets.QApplication(sys.argv)
    # 为避免中文乱码，设置程序使用中文字体（例如：宋体）
    font = QtGui.QFont()
    font.setFamily("SimSun")
    app.setFont(font)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
