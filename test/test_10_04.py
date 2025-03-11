#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
示例：利用 PyQt5 实现上海市数据的 Ripley 函数分析及地图显示
    - 结合上海市地图文件（shp、dbf、cpg、prj），显示上海市地图
    - 在文件读取过程中，将 CSV 数据中的点显示在地图上，并支持缩放、平移
    - 支持在地图上通过鼠标拖拽选择区域，选择后更新参数文本框，并自动缩放到对应区域
    - 用户可设置 Ripley 函数分析参数（区域、空间半径、时间范围等），并分别绘制 K 函数与 L 函数
    - 分析计算在独立线程中进行，并通过进度条显示进度
"""
import math
import sys
import os
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
from shapely.geometry import Polygon
from pyproj import Transformer
from scipy.spatial import KDTree, cKDTree
import matplotlib

# 设置 Matplotlib 使用支持中文的字体，避免中文乱码
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 如有需要请换成系统中存在的中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 指定后端为 Qt5Agg
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

# 导入 PyQt5 相关模块
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, QProgressBar, QPushButton, QLabel,
                             QVBoxLayout, QHBoxLayout, QWidget, QLineEdit, QComboBox, QDoubleSpinBox,
                             QGroupBox, QGridLayout, QTabWidget)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# 导入 matplotlib 与 PyQt5 结合的类
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# 导入 RectangleSelector 用于区域选择
from matplotlib.widgets import RectangleSelector

##############################################
# 计算线程：在独立线程中进行 Ripley 函数计算
##############################################

class ComputationThread(QThread):
    progress_signal = pyqtSignal(int)   # 发射进度百分比
    result_signal = pyqtSignal(object)    # 发射计算结果

    def __init__(self, points, times, region, r_min, r_max, r_step,
                 t_min, t_max, t_step, analysis_type, correction_method, parent=None):
        """
        :param points: numpy 数组，形状 (N,2) 存储经度、纬度
        :param times: numpy 数组，形状 (N,) 存储时间（转换为 float，例如使用 datetime.toordinal()），空间分析可为 None
        :param region: 元组 (xmin, xmax, ymin, ymax)，指定研究区域
        :param r_min, r_max, r_step: 空间半径的最小、最大值及步长
        :param t_min, t_max, t_step: 时间范围的最小、最大值及步长（时空分析使用）
        :param analysis_type: "spatial" 或 "spatiotemporal"
        :param correction_method: "torus"（超环面）、"buffer"（缓冲区）或 "perimeter"（周长法）
        """
        super(ComputationThread, self).__init__(parent)
        self.points = points
        self.times = times
        self.region = region
        self.r_min = r_min
        self.r_max = r_max
        self.r_step = r_step
        self.t_min = t_min
        self.t_max = t_max
        self.t_step = t_step
        self.analysis_type = analysis_type
        self.correction_method = correction_method

    def run(self):
        if self.analysis_type == "spatial":
            r_vals, K_vals = self.compute_spatial_ripley()
            results = {
                "analysis_type": "spatial",
                "r": r_vals,
                "K": K_vals
            }
        else:
            r_vals, t_vals, K_st = self.compute_spatiotemporal_ripley()
            results = {
                "analysis_type": "spatiotemporal",
                "r": r_vals,
                "t": t_vals,
                "K": K_st
            }
        self.result_signal.emit(results)

    def compute_spatial_ripley(self):
        points = self.points
        n = len(points)
        A = (self.region[1] - self.region[0]) * (self.region[3] - self.region[2])
        r_values = np.arange(self.r_min, self.r_max + self.r_step, self.r_step)
        K_values = np.zeros_like(r_values, dtype=float)
        num_r = len(r_values)

        if self.correction_method == "torus":
            Lx = self.region[1] - self.region[0]
            Ly = self.region[3] - self.region[2]
            offsets = np.array([[dx, dy] for dx in (-Lx, 0, Lx) for dy in (-Ly, 0, Ly)])
            replicated_points = np.concatenate([points + offset for offset in offsets], axis=0)
            tree = cKDTree(replicated_points)
            n_block = n
            for idx, r in enumerate(r_values):
                total = 0
                for i, p in enumerate(points):
                    neighbors = tree.query_ball_point(p, r)
                    self_index = i + 4 * n_block  # 第5块对应(0,0)偏移
                    if self_index in neighbors:
                        count = len(neighbors) - 1
                    else:
                        count = len(neighbors)
                    total += count
                K_values[idx] = A / (n * (n - 1)) * total
                self.progress_signal.emit(int(100 * (idx + 1) / num_r))
        elif self.correction_method == "buffer":
            tree = cKDTree(points)
            for idx, r in enumerate(r_values):
                mask = ((points[:, 0] - self.region[0] >= r) &
                        (self.region[1] - points[:, 0] >= r) &
                        (points[:, 1] - self.region[2] >= r) &
                        (self.region[3] - points[:, 1] >= r))
                interior_points = points[mask]
                n_interior = len(interior_points)
                if n_interior == 0:
                    K_values[idx] = np.nan
                else:
                    total = 0
                    for p in interior_points:
                        count = len(tree.query_ball_point(p, r)) - 1
                        total += count
                    K_values[idx] = A / (n_interior * (n - 1)) * total
                self.progress_signal.emit(int(100 * (idx + 1) / num_r))
        elif self.correction_method == "perimeter":
            tree = cKDTree(points)
            for idx, r in enumerate(r_values):
                total = 0
                for i, p in enumerate(points):
                    count = len(tree.query_ball_point(p, r)) - 1
                    circle = Point(p[0], p[1]).buffer(r)
                    study_area = box(self.region[0], self.region[2], self.region[1], self.region[3])
                    intersection_area = circle.intersection(study_area).area
                    e = intersection_area / (math.pi * r ** 2) if r > 0 else 1
                    weight = 1 / e if e > 0 else 0
                    total += weight * count
                K_values[idx] = A / (n * (n - 1)) * total
                self.progress_signal.emit(int(100 * (idx + 1) / num_r))
        else:
            K_values = np.zeros_like(r_values)
            self.progress_signal.emit(100)
        return r_values, K_values

    def compute_spatiotemporal_ripley(self):
        points = self.points
        times = self.times
        n = len(points)
        A = (self.region[1] - self.region[0]) * (self.region[3] - self.region[2])
        T_total = np.max(times) - np.min(times)
        V = A * T_total
        r_values = np.arange(self.r_min, self.r_max + self.r_step, self.r_step)
        t_values = np.arange(self.t_min, self.t_max + self.t_step, self.t_step)
        K_st = np.zeros((len(r_values), len(t_values)), dtype=float)
        num_total = len(r_values) * len(t_values)
        count_calculations = 0
        tree = cKDTree(points)
        for i_r, r in enumerate(r_values):
            for i_t, t_lim in enumerate(t_values):
                total = 0
                for i, p in enumerate(points):
                    if self.correction_method == "torus":
                        Lx = self.region[1] - self.region[0]
                        Ly = self.region[3] - self.region[2]
                        count = 0
                        for j, q in enumerate(points):
                            if i == j:
                                continue
                            dx = abs(p[0] - q[0])
                            dx = min(dx, Lx - dx)
                            dy = abs(p[1] - q[1])
                            dy = min(dy, Ly - dy)
                            d = math.hypot(dx, dy)
                            if d <= r and abs(times[i] - times[j]) <= t_lim:
                                count += 1
                        total += count
                    else:
                        indices = tree.query_ball_point(p, r)
                        indices = [j for j in indices if j != i]
                        if self.correction_method == "buffer":
                            if (p[0] - self.region[0] < r or self.region[1] - p[0] < r or
                                p[1] - self.region[2] < r or self.region[3] - p[1] < r):
                                continue
                            count = sum(1 for j in indices if abs(times[i] - times[j]) <= t_lim)
                            total += count
                        elif self.correction_method == "perimeter":
                            count = len([j for j in indices if abs(times[i] - times[j]) <= t_lim])
                            circle = Point(p[0], p[1]).buffer(r)
                            study_area = box(self.region[0], self.region[2], self.region[1], self.region[3])
                            intersection_area = circle.intersection(study_area).area
                            e = intersection_area / (math.pi * r ** 2) if r > 0 else 1
                            weight = 1 / e if e > 0 else 0
                            total += weight * count
                K_st[i_r, i_t] = V / (n * (n - 1)) * total
                count_calculations += 1
                self.progress_signal.emit(int(100 * count_calculations / num_total))
        return r_values, t_values, K_st

########################################################################
# 定义主窗口类 MainWindow，构建 UI 界面、交互逻辑及响应函数
########################################################################
class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("空间/时空 Ripley 函数分析")
        self.resize(1200, 800)

        # 初始化全局变量
        self.data = None              # 数据点数组（Nx2）
        self.time_data = None         # 时间数据（数值型数组，单位秒）
        self.shanghai_map = None      # 上海市地图数据（GeoDataFrame）
        self.region_polygon = None    # 计算区域，shapely Polygon 对象
        self._isPanning = False  # 标记是否处于平移状态
        self._panStart = None  # 记录鼠标按下时的起始点
        self._xlim_start = None  # 平移开始时的 x 轴范围
        self._ylim_start = None  # 平移开始时的 y 轴范围

        self.computation_thread = None

        # 初始化 UI
        self.initUI()

    def initUI(self):
        """
        构造主界面，左侧为参数控制面板，右侧为 Tab 页（地图显示、结果显示）
        """
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)

        # ------------------------------
        # 左侧控制面板
        # ------------------------------
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)

        # 【1】读取数据文件按钮
        self.btn_load_data = QPushButton("读取数据文件")
        self.btn_load_data.clicked.connect(self.load_data)
        control_layout.addWidget(self.btn_load_data)

        # 显示已加载数据文件名称的标签
        self.label_data_path = QLabel("未选择数据文件")
        control_layout.addWidget(self.label_data_path)

        # 【2】加载上海市地图按钮（shp文件）
        self.btn_load_map = QPushButton("加载上海市地图")
        self.btn_load_map.clicked.connect(self.load_shanghai_map)
        control_layout.addWidget(self.btn_load_map)

        # 【3】Ripley 函数参数设置区域
        param_group = QGroupBox("Ripley 函数参数设置")
        param_layout = QGridLayout(param_group)

        # r 最小值
        label_r_min = QLabel("r 最小值:")
        self.spin_r_min = QDoubleSpinBox()
        self.spin_r_min.setRange(0, 1e6)
        self.spin_r_min.setValue(50)
        param_layout.addWidget(label_r_min, 0, 0)
        param_layout.addWidget(self.spin_r_min, 0, 1)

        # r 最大值
        label_r_max = QLabel("r 最大值:")
        self.spin_r_max = QDoubleSpinBox()
        self.spin_r_max.setRange(0, 1e6)
        self.spin_r_max.setValue(1000)
        param_layout.addWidget(label_r_max, 1, 0)
        param_layout.addWidget(self.spin_r_max, 1, 1)

        # r 步长
        label_r_spin = QLabel("r 步长:")
        self.spin_r_step = QDoubleSpinBox()
        self.spin_r_step.setRange(1, 1e5)
        self.spin_r_step.setValue(50)
        param_layout.addWidget(label_r_spin, 2, 0)
        param_layout.addWidget(self.spin_r_step, 2, 1)

        # t 最小值
        label_t_min = QLabel("t 最小值:")
        self.spin_t_min = QDoubleSpinBox()
        self.spin_t_min.setRange(0, 1e6)
        self.spin_t_min.setValue(1)
        param_layout.addWidget(label_t_min, 3, 0)
        param_layout.addWidget(self.spin_t_min, 3, 1)

        # t 最大值
        label_t_max = QLabel("t 最大值:")
        self.spin_t_max = QDoubleSpinBox()
        self.spin_t_max.setRange(0, 1e6)
        self.spin_t_max.setValue(10)
        param_layout.addWidget(label_t_max, 4, 0)
        param_layout.addWidget(self.spin_t_max, 4, 1)

        # t 步长
        label_t_spin = QLabel("t 步长:")
        self.spin_t_step = QDoubleSpinBox()
        self.spin_t_step.setRange(1, 1e5)
        self.spin_t_step.setValue(1)
        param_layout.addWidget(label_t_spin, 5, 0)
        param_layout.addWidget(self.spin_t_step, 5, 1)

        # 计算区域输入（xmin, ymin, xmax, ymax），用户可手动输入
        label_region = QLabel("计算区域 (xmin,ymin,xmax,ymax):")
        param_layout.addWidget(label_region, 6, 0, 1, 2)

        # 研究区域输入：xmin, xmax, ymin, ymax
        label_x_min = QLabel("区域 xmin:")
        self.xmin_edit = QLineEdit()
        param_layout.addWidget(label_x_min, 7, 0)
        param_layout.addWidget(self.xmin_edit, 7, 1)
        label_x_max = QLabel("区域 xmax:")
        self.xmax_edit = QLineEdit()
        param_layout.addWidget(label_x_max, 8, 0)
        param_layout.addWidget(self.xmax_edit, 8, 1)

        label_y_min = QLabel("区域 ymin:")
        self.ymin_edit = QLineEdit()
        param_layout.addWidget(label_y_min, 9, 0)
        param_layout.addWidget(self.ymin_edit, 9, 1)
        label_y_max = QLabel("区域 ymax:")
        self.ymax_edit = QLineEdit()
        param_layout.addWidget(label_y_max, 10, 0)
        param_layout.addWidget(self.ymax_edit, 10, 1)
        # # 当用户手动修改区域时，同步更新显示区域的最值信息
        # self.edit_region.textChanged.connect(self.update_region_from_text)

        # 显示当前区域的最大最小值
        # self.label_region_info = QLabel("当前区域：未选择")
        # param_layout.addWidget(self.label_region_info, 5, 0, 1, 2)

        # 边界修正方法选择
        label_boundary = QLabel("边界修正方法:")
        self.combo_boundary = QComboBox()
        self.combo_boundary.addItems(["超环面法", "缓冲区法", "周长法"])
        param_layout.addWidget(label_boundary, 11, 0)
        param_layout.addWidget(self.combo_boundary, 11, 1)

        control_layout.addWidget(param_group)

        # 计算类型选择：空间或时空 Ripley 函数
        self.combo_compute_type = QComboBox()
        self.combo_compute_type.addItems(["空间 Ripley 函数", "时空 Ripley 函数"])
        control_layout.addWidget(self.combo_compute_type)

        # 【4】开始计算按钮
        self.btn_start_compute = QPushButton("开始计算")
        self.btn_start_compute.clicked.connect(self.start_computation)
        control_layout.addWidget(self.btn_start_compute)

        # 【5】进度条，用于显示数据读取和计算进度
        self.progress_bar = QProgressBar()
        control_layout.addWidget(self.progress_bar)

        # 添加一个弹性间隔，使控件靠上显示
        control_layout.addStretch()

        # ------------------------------
        # 右侧显示面板（Tab页：地图显示、结果显示）
        # ------------------------------
        self.tab_widget = QTabWidget()

        # 地图显示 Tab 页
        self.map_tab = QWidget()
        map_layout = QVBoxLayout(self.map_tab)
        # 创建 Matplotlib 图形，显示上海市地图和数据点
        self.fig_map = plt.Figure(figsize=(5, 4))
        self.canvas_map = FigureCanvas(self.fig_map)
        map_layout.addWidget(self.canvas_map)
        self.ax_map = self.fig_map.add_subplot(111)
        self.ax_map.set_title("上海市地图及数据点")
        # 绑定鼠标滚轮事件，实现缩放
        self.canvas_map.mpl_connect("scroll_event", self.on_scroll)
        self.canvas_map.mpl_connect("button_press_event", self.on_press)
        self.canvas_map.mpl_connect("motion_notify_event", self.on_motion)
        self.canvas_map.mpl_connect("button_release_event", self.on_release)
        # 使用 RectangleSelector 实现区域框选（鼠标左键拖拽）
        self.RS = RectangleSelector(self.ax_map, self.on_select_region,
                                    useblit=True,
                                    button=[1],  # 仅响应鼠标左键
                                    minspanx=5, minspany=5,
                                    spancoords='pixels',
                                    interactive=True)
        self.tab_widget.addTab(self.map_tab, "地图显示")

        # 结果显示 Tab 页（显示 K 函数与 L 函数曲线）
        self.result_tab = QWidget()
        result_layout = QVBoxLayout(self.result_tab)
        self.fig_result = plt.Figure(figsize=(5, 4))
        self.canvas_result = FigureCanvas(self.fig_result)
        result_layout.addWidget(self.canvas_result)
        # 创建两个子图，分别显示 K 函数和 L 函数曲线
        self.ax_K = self.fig_result.add_subplot(211)
        self.ax_L = self.fig_result.add_subplot(212)
        self.ax_K.set_title("K 函数曲线")
        self.ax_L.set_title("L 函数曲线")
        self.tab_widget.addTab(self.result_tab, "结果显示")

        # 将左侧控制面板和右侧 Tab 页加入主布局
        main_layout.addWidget(control_widget, 2)
        main_layout.addWidget(self.tab_widget, 5)

        self.setCentralWidget(main_widget)

    def on_press(self, event):
        """鼠标按下事件：当右键按下时启动平移功能"""
        if event.button == 3 and event.inaxes == self.ax_map:  # 右键（button==3）用于平移
            self._isPanning = True
            self._panStart = (event.xdata, event.ydata)
            self._xlim_start = self.ax_map.get_xlim()
            self._ylim_start = self.ax_map.get_ylim()

    def on_motion(self, event):
        """鼠标移动事件：如果处于平移状态，则根据鼠标移动更新图像范围"""
        if self._isPanning and event.inaxes == self.ax_map and event.xdata is not None and event.ydata is not None:
            dx = self._panStart[0] - event.xdata
            dy = self._panStart[1] - event.ydata
            new_xlim = (self._xlim_start[0] + dx, self._xlim_start[1] + dx)
            new_ylim = (self._ylim_start[0] + dy, self._ylim_start[1] + dy)
            self.ax_map.set_xlim(new_xlim)
            self.ax_map.set_ylim(new_ylim)
            self.canvas_map.draw()

    def on_release(self, event):
        """鼠标释放事件：停止平移"""
        self._isPanning = False

    def load_data(self):
        """
        通过文件对话框选择 CSV 数据文件并读取，
        数据文件要求包含：longitude, latitude, Date 三列；
        Date 列转换为 datetime 类型后，再转换为时间戳（单位秒）。
        """
        file_path, _ = QFileDialog.getOpenFileName(self, "选择数据文件", "", "CSV Files (*.csv);;All Files (*)")
        if file_path:
            try:
                # 读取 CSV 文件，指定编码为 utf-8 避免中文乱码
                df = pd.read_csv(file_path, encoding='utf-8')
                # 检查必要的列是否存在
                if not set(['longitude', 'latitude', 'Date']).issubset(df.columns):
                    self.label_data_path.setText("数据文件格式错误，缺少必要列")
                    return
                # 将经纬度转换为浮点数
                df['longitude'] = df['longitude'].astype(float)
                df['latitude'] = df['latitude'].astype(float)
                # 将日期转换为 datetime 类型，并转换为时间戳（单位：秒）
                df['Date'] = pd.to_datetime(df['Date'])
                df['timestamp'] = df['Date'].astype(np.int64) // 10**9
                # 将数据存储为 numpy 数组（假设数据中的坐标单位为米，与地图坐标一致）
                self.data = df[['longitude', 'latitude']].to_numpy()
                self.time_data = df['timestamp'].to_numpy()
                self.label_data_path.setText("已加载数据文件: " + os.path.basename(file_path))
                # 更新地图显示，将数据点绘制到地图上
                self.update_map()
            except Exception as e:
                self.label_data_path.setText("读取数据文件失败: " + str(e))

    def load_shanghai_map(self):
        """
        通过文件对话框选择上海市地图 shp 文件，
        地图原始坐标系为 EPSG:4326，将其转换为与数据相同的投影（单位米，例如 EPSG:3857），
        并更新地图显示。
        """
        file_path, _ = QFileDialog.getOpenFileName(self, "选择上海市地图 Shapefile", "", "Shapefile (*.shp);;All Files (*)")
        if file_path:
            try:
                self.shanghai_map = gpd.read_file(file_path)
                # 将地图从 EPSG:4326 转换为 EPSG:3857（单位：米），请确保数据与转换后的坐标系一致
                self.shanghai_map = self.shanghai_map.to_crs(epsg=3097)
                self.label_data_path.setText(self.label_data_path.text() + " | 已加载上海市地图: " + os.path.basename(file_path))
                self.update_map()
            except Exception as e:
                self.label_data_path.setText("加载上海市地图失败: " + str(e))

    def update_map(self):
        """
        更新地图显示：
        - 绘制上海市地图（若已加载）；
        - 绘制数据点（若已加载）；
        - 绘制用户选择或输入的计算区域矩形；
        - 同时在区域信息标签中显示区域的最小值和最大值。
        """
        self.ax_map.clear()
        self.ax_map.set_title("上海市地图及数据点")
        # 绘制上海市地图
        if self.shanghai_map is not None:
            self.shanghai_map.plot(ax=self.ax_map, color='lightgray', edgecolor='black')
        # 绘制数据点（假设数据坐标与地图坐标一致）
        if self.data is not None:
            self.ax_map.scatter(self.data[:, 0], self.data[:, 1], c='red', s=10, label='数据点')
            self.ax_map.legend()
        # 绘制计算区域：优先使用 region_polygon（通过鼠标选择或输入）
        if self.region_polygon is not None:
            xmin, ymin, xmax, ymax = self.region_polygon.bounds
            rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 edgecolor='blue', facecolor='none', lw=2)
            self.ax_map.add_patch(rect)
            # 更新区域信息标签，显示当前区域的最小值和最大值
            self.update_region_from_map([xmin,ymin,xmax,ymax])
            # self.label_region_info.setText(f"当前区域：xmin={xmin:.2f}, ymin={ymin:.2f}, xmax={xmax:.2f}, ymax={ymax:.2f}")
        self.canvas_map.draw()

    def on_scroll(self, event):
        """
        处理鼠标滚轮事件，实现地图的缩放功能
        """
        base_scale = 1.2  # 缩放因子
        cur_xlim = self.ax_map.get_xlim()
        cur_ylim = self.ax_map.get_ylim()
        xdata = event.xdata
        ydata = event.ydata
        if xdata is None or ydata is None:
            return
        # 根据滚轮方向确定缩放比例
        if event.button == 'up':  # 放大
            scale_factor = 1 / base_scale
        elif event.button == 'down':  # 缩小
            scale_factor = base_scale
        else:
            scale_factor = 1
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])
        self.ax_map.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * relx])
        self.ax_map.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * rely])
        self.canvas_map.draw()

    def update_region_from_map(self, region):
        """
        :param region:
        :type region:
        :return:
        :rtype:
        """
        self.xmin_edit.setText(str(region[0]))
        self.ymin_edit.setText(str(region[1]))
        self.xmax_edit.setText(str(region[2]))
        self.ymax_edit.setText(str(region[3]))


    def on_select_region(self, eclick, erelease):
        """
        当用户在地图上用鼠标拖拽选定一个矩形区域时触发，
        更新区域输入框、内部 region_polygon 变量，并同步更新区域信息显示
        """
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        xmin, xmax = sorted([x1, x2])
        ymin, ymax = sorted([y1, y2])
        # 将区域坐标更新到输入框中
        # region_str = f"{xmin:.2f},{ymin:.2f},{xmax:.2f},{ymax:.2f}"
        # self.edit_region.setText(region_str)
        self.update_region_from_map([xmin,ymin,xmax,ymax])
        # 更新内部计算区域
        self.region_polygon = box(xmin, ymin, xmax, ymax)
        # 更新地图显示
        self.update_map()

    def update_region_from_text(self):
        """
        当用户手动在区域输入框中修改区域时，
        解析输入内容，并更新内部的 region_polygon 以及区域信息显示；
        格式要求：xmin,ymin,xmax,ymax
        """
        region_text = self.edit_region.text().strip()
        if region_text:
            try:
                coords = [float(x) for x in region_text.split(',')]
                if len(coords) == 4:
                    xmin, ymin, xmax, ymax = coords
                    self.region_polygon = box(xmin, ymin, xmax, ymax)
                    # 更新区域信息标签显示
                    self.update_region_from_map(coords)
                    # self.label_region_info.setText(f"当前区域：xmin={xmin:.2f}, ymin={ymin:.2f}, xmax={xmax:.2f}, ymax={ymax:.2f}")
                    self.update_map()
            except Exception as e:
                print("区域输入错误:", e)

    def start_computation(self):
        """
        当用户点击“开始计算”时，检查数据和计算区域是否已加载，
        读取用户参数，创建并启动计算线程进行 Ripley 函数计算，
        计算过程中更新进度条，计算完成后显示结果。
        """
        if self.data is None:
            self.label_data_path.setText("请先加载数据文件")
            return
        self.label_data_path.setText("计算开始")
        # 如果用户未选择计算区域，则默认使用数据的外包矩形
        if self.region_polygon is None:
            xmin, ymin = np.min(self.data, axis=0)
            xmax, ymax = np.max(self.data, axis=0)
            self.region_polygon = box(xmin, ymin, xmax, ymax)

            self.update_region_from_map([xmin,ymin,xmax,ymax])
        xmin, ymin, xmax, ymax = self.region_polygon.bounds
        self.region = (xmin,xmax,ymin,ymax)
        r_min = self.spin_r_min.value()
        r_max = self.spin_r_max.value()
        r_step = self.spin_r_step.value()
        t_min = self.spin_t_min.value()
        t_max = self.spin_t_max.value()
        t_step = self.spin_t_step.value()

        # 根据用户选择的边界修正方法转换为内部标识
        boundary_text = self.combo_boundary.currentText()
        if boundary_text == "超环面法":
            boundary_method = 'torus'
        elif boundary_text == "缓冲区法":
            boundary_method = 'buffer'
        elif boundary_text == "周长法":
            boundary_method = 'perimeter'
        else:
            boundary_method = 'toroidal'

        # 根据用户选择的计算类型
        compute_type_text = self.combo_compute_type.currentText()
        if compute_type_text == "空间 Ripley 函数":
            compute_type = 'spatial'
        else:
            compute_type = 'spatiotemporal'

        # 创建计算线程对象，并连接进度和结果信号
        self.progress_bar.setValue(0)
        self.computation_thread = ComputationThread(self.data,
                                                    self.time_data,
                                                    self.region,
                                                    r_min, r_max,r_step,
                                                    t_min, t_max,t_step,
                                                    compute_type,boundary_method, )
        #
        # self.computation_thread = ComputationThread(self.data, self.time_data, self.region_polygon,
        #                                             r_min, r_max, t_min, t_max, boundary_method, compute_type)
        self.computation_thread.progress_signal.connect(self.update_progress)
        self.computation_thread.result_signal.connect(self.handle_results)
        self.computation_thread.start()

    def update_progress(self, value):
        """
        根据线程发来的进度值，更新进度条显示
        """
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}收到进度更新：{value}")
        self.progress_bar.setValue(value)

    def handle_results(self, results):
        """
        接收到计算线程返回的结果后，在结果 Tab 中绘制 K 函数和 L 函数曲线；
        如果计算时空 Ripley 函数，则在控制台打印相应信息（可进一步扩展显示）。
        """
        if not results:
            print("计算结果为空")
            return
        if results['analysis_type'] =='spatial':
            print("计算空间 Ripley 函数，结果如下：")
            r_values = results['r']
            K_values = results['K']
            L_values = np.sqrt(K_values/np.pi) - r_values
            self.ax_K.clear()
            self.ax_K.plot(r_values, K_values, 'b-', label='K函数')
            self.ax_K.set_title("K 函数")
            self.ax_K.set_xlabel("r")
            self.ax_K.set_ylabel("K(r)")
            self.ax_K.legend()

            self.ax_L.clear()
            self.ax_L.plot(r_values, L_values, 'r-', label='L函数')
            self.ax_L.set_title("L 函数 (sqrt(K/π)-r)")
            self.ax_L.set_xlabel("r")
            self.ax_L.set_ylabel("L(r)")
            self.ax_L.legend()
        else:
            print("计算时空 Ripley 函数，结果如下：")
            r_values = results["r"]
            t_values = results["t"]
            K_st = results["K"]
            L_st = np.sqrt(K_st/(np.pi*t_values)) - r_values
            self.ax_K.clear()
            im_K = self.ax_K.imshow(K_st,extent=[t_values[0],t_values[-1],r_values[0],r_values[-1]],
                             origin='lower',aspect='auto')
            self.ax_K.set_title("K 函数")
            self.ax_K.set_xlabel("t")
            self.ax_K.set_ylabel("r")
            self.ax_K.legend()

            self.ax_L.clear()
            im_L = self.ax_L.imshow(L_st, extent=[t_values[0],t_values[-1],r_values[0],r_values[-1]],
                             origin='lower',aspect='auto')
            self.ax_L.set_title("L 函数 (sqrt(K/πt)-r)")
            self.ax_L.set_xlabel("t")
            self.ax_L.set_ylabel("r")
            self.ax_L.legend()

        self.canvas_result.draw()


########################################################################
# 主程序入口
########################################################################
if __name__ == '__main__':
    # 创建 QApplication 对象
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())