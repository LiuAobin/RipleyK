#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PyQt程序实现空间及时空Ripley函数分析
- 支持数据文件读取（CSV格式，包含longitude, latitude, Date三列）
- 加载上海市地图（shp文件，原始坐标系EPSG:4326，将转换为大地坐标系，例如EPSG:3857，单位为米）
- 在地图上显示数据点，并支持鼠标滚轮缩放、以及拖拽画框选择计算区域
- 用户可设置Ripley函数的参数（r、t的最小/最大值）及边界修正方法（超环面法、缓冲区法、周长法）
- 计算过程在独立线程中执行，并通过进度条显示进度
- 采用KDTree数据结构加速邻域查询
- 结果同时显示K函数曲线和L函数曲线
"""

import sys
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
from shapely.geometry import Polygon
from pyproj import Transformer
from scipy.spatial import KDTree
import matplotlib

# 使用 Qt5Agg 后端（确保与 PyQt5 兼容）
matplotlib.use("Qt5Agg")
# 设置中文字体（如系统中无 SimHei，请替换为可用的中文字体）
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

# 导入 PyQt5 相关模块
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, QProgressBar, QPushButton, QLabel,
                             QVBoxLayout, QHBoxLayout, QWidget, QLineEdit, QComboBox, QDoubleSpinBox,
                             QGroupBox, QGridLayout, QTabWidget)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# 导入 matplotlib 与 PyQt5 结合的类
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# 导入 RectangleSelector 用于区域选择
from matplotlib.widgets import RectangleSelector


########################################################################
# 定义计算线程类，利用 QThread 在后台执行 Ripley 函数计算
########################################################################
class ComputationThread(QThread):
    # 定义信号：progress_signal 用于更新进度（整数百分比）；result_signal 用于传递计算结果（字典）
    progress_signal = pyqtSignal(int)
    result_signal = pyqtSignal(dict)

    def __init__(self, data, time_data, region_polygon, r_min, r_max, t_min, t_max,
                 boundary_method, compute_type, num_r=50, num_t=10, parent=None):
        """
        初始化计算线程

        参数说明：
        - data：二维数组，每一行为一个数据点 [x, y]
        - time_data：时间数据（数值型，例如时间戳数组）
        - region_polygon：计算区域，shapely Polygon 对象
        - r_min, r_max：Ripley函数中半径r的最小值和最大值
        - t_min, t_max：时空Ripley函数中时间t的最小值和最大值
        - boundary_method：边界修正方法，取值 'toroidal'（超环面法）、'buffer'（缓冲区法）、'perimeter'（周长法）
        - compute_type：计算类型，'spatial' 表示空间Ripley函数，'spatio-temporal' 表示时空Ripley函数
        - num_r：r的离散取值个数（默认50）
        - num_t：t的离散取值个数（默认10，仅在时空计算中使用）
        """
        super(ComputationThread, self).__init__(parent)
        self.data = data
        self.time_data = time_data
        self.region_polygon = region_polygon
        self.r_min = r_min
        self.r_max = r_max
        self.t_min = t_min
        self.t_max = t_max
        self.boundary_method = boundary_method
        self.compute_type = compute_type
        self.num_r = num_r
        self.num_t = num_t

    def run(self):
        """
        重写 run() 方法，在独立线程中执行计算过程，
        并通过 progress_signal 更新进度，通过 result_signal 返回计算结果。
        """
        results = {}
        try:
            # 获取计算区域的边界及面积
            minx, miny, maxx, maxy = self.region_polygon.bounds
            area_region = self.region_polygon.area

            # 构造 r 值数组（在 [r_min, r_max] 区间均匀取值）
            r_values = np.linspace(self.r_min, self.r_max, self.num_r)
            results['r_values'] = r_values

            # 构建 KDTree 加速空间查询
            tree = KDTree(self.data)

            n = len(self.data)
            lambda_density = n / area_region  # 点密度

            # 对于缓冲区法，剔除离边界过近的点（距离小于 r_max 的点不计入计算）
            if self.boundary_method == 'buffer':
                interior_indices = []
                for i, point in enumerate(self.data):
                    if self.region_polygon.contains(Point(point)):
                        # 计算该点到区域边界的距离
                        if Point(point).distance(self.region_polygon.boundary) >= self.r_max:
                            interior_indices.append(i)
                data_indices = np.array(interior_indices)
            else:
                data_indices = np.arange(n)

            # 初始化 K 函数数组（空间函数）
            K_values = np.zeros_like(r_values, dtype=float)

            # 若计算时空Ripley函数，则初始化 t 值数组及 K(r, t) 二维数组
            if self.compute_type == 'spatio-temporal':
                t_values = np.linspace(self.t_min, self.t_max, self.num_t)
                results['t_values'] = t_values
                K_st = np.zeros((self.num_r, self.num_t), dtype=float)

            # -------------------------------
            # 计算空间 Ripley 函数
            # -------------------------------
            for idx, r in enumerate(r_values):
                sum_val = 0.0

                # 若使用周长法，则对每个数据点预先计算校正因子（即圆上在区域内的比例的倒数）
                if self.boundary_method == 'perimeter':
                    num_samples = 100  # 采样点数，用于近似计算圆周在区域内的比例
                    angles = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
                    unit_offsets = np.column_stack((np.cos(angles), np.sin(angles)))
                    correction_factors = np.zeros(n, dtype=float)
                    for i in range(n):
                        center = self.data[i]
                        # 构造以该点为中心、半径为 r 的圆上采样点
                        circle_points = center + r * unit_offsets
                        inside = 0
                        for pt in circle_points:
                            if self.region_polygon.contains(Point(pt)):
                                inside += 1
                        fraction = inside / num_samples
                        if fraction == 0:
                            fraction = 1e-6  # 避免除零
                        correction_factors[i] = 1 / fraction

                # 对于数据集中（或 interior_indices 内）的每个点，统计其他点中距离小于 r 的个数
                for i in data_indices:
                    point = self.data[i]
                    if self.boundary_method == 'toroidal':
                        # 超环面法：计算考虑边界“包裹”效应的距离
                        Lx = maxx - minx
                        Ly = maxy - miny
                        diff = np.abs(self.data - point)
                        # 对 x 和 y 坐标分别取最小值（直线距离或环面距离）
                        diff[:, 0] = np.minimum(diff[:, 0], Lx - diff[:, 0])
                        diff[:, 1] = np.minimum(diff[:, 1], Ly - diff[:, 1])
                        distances = np.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2)
                        distances[i] = np.inf  # 排除自身
                        count = np.sum(distances <= r)
                    else:
                        # 使用 KDTree 进行查询（普通欧式距离）
                        indices = tree.query_ball_point(point, r)
                        if i in indices:
                            indices.remove(i)
                        count = len(indices)
                        # 若使用周长法，乘以该点对应的校正因子
                        if self.boundary_method == 'perimeter':
                            count *= correction_factors[i]
                    sum_val += count

                # 计算 Ripley K 函数的估计量
                if n > 1:
                    K_r = area_region / (n * (n - 1)) * sum_val
                else:
                    K_r = 0
                K_values[idx] = K_r

                # 更新进度（假设空间函数计算占总进度的50%）
                progress = int(50 * (idx + 1) / self.num_r)
                self.progress_signal.emit(progress)

            # -------------------------------
            # 若计算时空 Ripley 函数，则进一步计算 K(r,t)
            # -------------------------------
            if self.compute_type == 'spatio-temporal':
                # 此处假设 time_data 已经是数值型（例如时间戳，单位秒）
                time_array = self.time_data
                T_range = time_array.max() - time_array.min()
                for i_r, r in enumerate(r_values):
                    if self.boundary_method == 'perimeter':
                        num_samples = 100
                        angles = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
                        unit_offsets = np.column_stack((np.cos(angles), np.sin(angles)))
                        correction_factors = np.zeros(n, dtype=float)
                        for i in range(n):
                            center = self.data[i]
                            circle_points = center + r * unit_offsets
                            inside = 0
                            for pt in circle_points:
                                if self.region_polygon.contains(Point(pt)):
                                    inside += 1
                            fraction = inside / num_samples
                            if fraction == 0:
                                fraction = 1e-6
                            correction_factors[i] = 1 / fraction

                    for j_t, t in enumerate(np.linspace(self.t_min, self.t_max, self.num_t)):
                        sum_val_st = 0.0
                        for i in data_indices:
                            point = self.data[i]
                            if self.boundary_method == 'toroidal':
                                Lx = maxx - minx
                                Ly = maxy - miny
                                diff = np.abs(self.data - point)
                                diff[:, 0] = np.minimum(diff[:, 0], Lx - diff[:, 0])
                                diff[:, 1] = np.minimum(diff[:, 1], Ly - diff[:, 1])
                                distances = np.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2)
                                distances[i] = np.inf
                                spatial_indices = np.where(distances <= r)[0]
                            else:
                                spatial_indices = tree.query_ball_point(point, r)
                                if i in spatial_indices:
                                    spatial_indices.remove(i)
                            # 筛选满足时间差不超过 t 的点
                            time_diffs = np.abs(time_array[spatial_indices] - time_array[i])
                            temporal_count = np.sum(time_diffs <= t)
                            if self.boundary_method == 'perimeter':
                                temporal_count *= correction_factors[i]
                            sum_val_st += temporal_count
                        if n > 1:
                            K_rt = area_region * T_range / (n * (n - 1)) * sum_val_st
                        else:
                            K_rt = 0
                        K_st[i_r, j_t] = K_rt
                        progress = 50 + int(50 * ((i_r * self.num_t + j_t + 1) / (self.num_r * self.num_t)))
                        self.progress_signal.emit(progress)
                results['K_st'] = K_st

            # 保存空间函数的计算结果
            results['K'] = K_values
            # 计算 L 函数：L(r) = sqrt(K(r)/pi)
            L_values = np.sqrt(K_values / np.pi)
            results['L'] = L_values

            # 计算完成，发送结果信号
            self.result_signal.emit(results)
        except Exception as e:
            print("计算过程中发生错误:", e)
            self.result_signal.emit({})


########################################################################
# 定义主窗口类 MainWindow，构建 UI 界面、交互逻辑及响应函数
########################################################################
class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("Ripley函数时空分析")
        self.resize(1200, 800)
        # 初始化全局变量
        self.data = None              # 存储数据点（Nx2 numpy 数组）
        self.time_data = None         # 存储时间数据（数值型数组）
        self.shanghai_map = None      # 存储上海市地图数据（GeoDataFrame）
        self.region_polygon = None    # 计算区域（shapely Polygon）
        self.computation_thread = None

        self.initUI()

    def initUI(self):
        """
        构造主界面，包括左侧参数控制面板与右侧 Tab 页（地图显示、结果显示）
        """
        # 主界面采用水平布局，左侧为参数控制，右侧为显示内容
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

        # 显示已加载数据文件名称
        self.label_data_path = QLabel("未选择数据文件")
        control_layout.addWidget(self.label_data_path)

        # 【2】加载上海市地图按钮
        self.btn_load_map = QPushButton("加载上海市地图")
        self.btn_load_map.clicked.connect(self.load_shanghai_map)
        control_layout.addWidget(self.btn_load_map)

        # 【3】Ripley函数参数设置区域
        param_group = QGroupBox("Ripley函数参数设置")
        param_layout = QGridLayout(param_group)

        # r 最小值设置
        label_r_min = QLabel("r最小值:")
        self.spin_r_min = QDoubleSpinBox()
        self.spin_r_min.setRange(0, 1e6)
        self.spin_r_min.setValue(0)
        param_layout.addWidget(label_r_min, 0, 0)
        param_layout.addWidget(self.spin_r_min, 0, 1)

        # r 最大值设置
        label_r_max = QLabel("r最大值:")
        self.spin_r_max = QDoubleSpinBox()
        self.spin_r_max.setRange(0, 1e6)
        self.spin_r_max.setValue(1000)
        param_layout.addWidget(label_r_max, 1, 0)
        param_layout.addWidget(self.spin_r_max, 1, 1)

        # t 最小值设置
        label_t_min = QLabel("t最小值:")
        self.spin_t_min = QDoubleSpinBox()
        self.spin_t_min.setRange(0, 1e6)
        self.spin_t_min.setValue(0)
        param_layout.addWidget(label_t_min, 2, 0)
        param_layout.addWidget(self.spin_t_min, 2, 1)

        # t 最大值设置
        label_t_max = QLabel("t最大值:")
        self.spin_t_max = QDoubleSpinBox()
        self.spin_t_max.setRange(0, 1e6)
        self.spin_t_max.setValue(10)
        param_layout.addWidget(label_t_max, 3, 0)
        param_layout.addWidget(self.spin_t_max, 3, 1)

        # 计算区域设置：允许用户手动输入区域坐标（xmin, ymin, xmax, ymax）
        label_region = QLabel("计算区域 (xmin, ymin, xmax, ymax):")
        self.edit_region = QLineEdit()
        self.edit_region.setPlaceholderText("例如: 300000,3400000,400000,3500000")
        param_layout.addWidget(label_region, 4, 0, 1, 2)

        # 边界修正方法选择
        label_boundary = QLabel("边界修正方法:")
        self.combo_boundary = QComboBox()
        self.combo_boundary.addItems(["超环面法", "缓冲区法", "周长法"])
        param_layout.addWidget(label_boundary, 5, 0)
        param_layout.addWidget(self.combo_boundary, 5, 1)

        control_layout.addWidget(param_group)

        # 计算类型选择（空间或时空）
        self.combo_compute_type = QComboBox()
        self.combo_compute_type.addItems(["空间Ripley函数", "时空Ripley函数"])
        control_layout.addWidget(self.combo_compute_type)

        # 【4】开始计算按钮
        self.btn_start_compute = QPushButton("开始计算")
        self.btn_start_compute.clicked.connect(self.start_computation)
        control_layout.addWidget(self.btn_start_compute)

        # 【5】进度条，用于显示数据读取与计算的进度
        self.progress_bar = QProgressBar()
        control_layout.addWidget(self.progress_bar)

        # 添加一个弹性间隔，将控件顶端对齐
        control_layout.addStretch()

        # ------------------------------
        # 右侧显示面板（Tab页：地图显示、结果显示）
        # ------------------------------
        self.tab_widget = QTabWidget()

        # 地图显示 Tab
        self.map_tab = QWidget()
        map_layout = QVBoxLayout(self.map_tab)
        # 创建 Matplotlib 图形，用于显示上海市地图和数据点
        self.fig_map = Figure(figsize=(5, 4))
        self.canvas_map = FigureCanvas(self.fig_map)
        map_layout.addWidget(self.canvas_map)
        self.ax_map = self.fig_map.add_subplot(111)
        self.ax_map.set_title("上海市地图及数据点")
        # 绑定鼠标滚轮事件，实现地图缩放
        self.canvas_map.mpl_connect("scroll_event", self.on_scroll)
        # 使用 RectangleSelector 实现区域框选（鼠标左键拖拽）
        self.RS = RectangleSelector(self.ax_map, self.on_select_region,
                                    useblit=True,
                                    button=[1],  # 仅响应鼠标左键
                                    minspanx=5, minspany=5,
                                    spancoords='pixels',
                                    interactive=True)
        self.tab_widget.addTab(self.map_tab, "地图显示")

        # 结果显示 Tab（显示 K 函数与 L 函数曲线）
        self.result_tab = QWidget()
        result_layout = QVBoxLayout(self.result_tab)
        self.fig_result = Figure(figsize=(5, 4))
        self.canvas_result = FigureCanvas(self.fig_result)
        result_layout.addWidget(self.canvas_result)
        # 在结果图中使用两个子图：上图显示 K 函数，下图显示 L 函数
        self.ax_K = self.fig_result.add_subplot(211)
        self.ax_L = self.fig_result.add_subplot(212)
        self.ax_K.set_title("K函数曲线")
        self.ax_L.set_title("L函数曲线")
        self.tab_widget.addTab(self.result_tab, "结果显示")

        # 将左侧控制面板与右侧 Tab 页加入主布局
        main_layout.addWidget(control_widget, 2)
        main_layout.addWidget(self.tab_widget, 5)

        self.setCentralWidget(main_widget)

    def load_data(self):
        """
        通过文件对话框选择 CSV 数据文件，并读取数据；
        数据文件要求包含：longitude, latitude, Date 三列。
        日期将转换为 datetime 类型，并进一步转换为时间戳（单位秒）。
        """
        file_path, _ = QFileDialog.getOpenFileName(self, "选择数据文件", "", "CSV Files (*.csv);;All Files (*)")
        if file_path:
            try:
                # 读取 CSV 文件，指定编码以避免中文乱码
                df = pd.read_csv(file_path, encoding='utf-8')
                # 检查必要的列
                if not set(['longitude', 'latitude', 'Date']).issubset(df.columns):
                    self.label_data_path.setText("数据文件格式错误，缺少必要列")
                    return
                # 将经纬度转换为浮点数
                df['longitude'] = df['longitude'].astype(float)
                df['latitude'] = df['latitude'].astype(float)
                # 将日期转换为 datetime 格式，并转换为时间戳（单位：秒）
                df['Date'] = pd.to_datetime(df['Date'])
                df['timestamp'] = df['Date'].astype(np.int64) // 10**9
                # 存储数据点（假设经度对应 x，纬度对应 y）
                self.data = df[['longitude', 'latitude']].to_numpy()
                self.time_data = df['timestamp'].to_numpy()
                self.label_data_path.setText("已加载数据文件: " + os.path.basename(file_path))
                # 更新地图显示（将数据点绘制到地图上）
                self.update_map()
            except Exception as e:
                self.label_data_path.setText("读取数据文件失败: " + str(e))

    def load_shanghai_map(self):
        """
        通过文件对话框选择上海市地图的 shp 文件，并加载地图数据；
        地图原始坐标系为 EPSG:4326，转换为大地坐标系（例如 EPSG:3857，单位米）
        """
        file_path, _ = QFileDialog.getOpenFileName(self, "选择上海市地图Shapefile", "", "Shapefile (*.shp);;All Files (*)")
        if file_path:
            try:
                self.shanghai_map = gpd.read_file(file_path)
                # 转换坐标系（EPSG:4326 -> EPSG:3857）
                self.shanghai_map = self.shanghai_map.to_crs(epsg=3097)
                self.label_data_path.setText(self.label_data_path.text() + " | 已加载上海市地图: " + os.path.basename(file_path))
                self.update_map()
            except Exception as e:
                self.label_data_path.setText("加载上海市地图失败: " + str(e))

    def update_map(self):
        """
        更新地图显示，将加载的上海市地图、数据点以及用户指定的计算区域显示到图上
        """
        self.ax_map.clear()
        self.ax_map.set_title("上海市地图及数据点")
        # 绘制上海市地图（若已加载）
        if self.shanghai_map is not None:
            self.shanghai_map.plot(ax=self.ax_map, color='lightgray', edgecolor='black')

        # 绘制数据点（假设数据点与地图坐标系一致）
        if self.data is not None:
            """
            from pyproj import Transformer

            # 创建从大地坐标（单位米）到 WGS84（EPSG:4326）的转换器
            transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
            
            # 假设数据点的坐标是 (x, y)
            x, y = 350911.4148, 3450754.43
            lon, lat = transformer.transform(x, y)
            
            print(f"转换后的坐标: 经度={lon}, 纬度={lat}")

            """
            self.ax_map.scatter(self.data[:, 0], self.data[:, 1], c='red', s=10, label='数据点')
            self.ax_map.legend()
        # 如果用户已在输入框中指定计算区域，则绘制蓝色矩形
        region_text = self.edit_region.text().strip()
        if region_text:
            try:
                coords = [float(x) for x in region_text.split(',')]
                if len(coords) == 4:
                    xmin, ymin, xmax, ymax = coords
                    rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, edgecolor='blue', facecolor='none', lw=2)
                    self.ax_map.add_patch(rect)
                    # 同时更新计算区域的 Polygon 对象
                    self.region_polygon = box(xmin, ymin, xmax, ymax)
            except Exception as e:
                print("区域输入错误:", e)
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
        if event.button == 'up':  # 向上滚动，放大
            scale_factor = 1 / base_scale
        elif event.button == 'down':  # 向下滚动，缩小
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

    def on_select_region(self, eclick, erelease):
        """
        当用户在地图上拖拽选定一个矩形区域时触发，
        更新计算区域输入框和内部保存的计算区域 Polygon 对象
        """
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        xmin, xmax = sorted([x1, x2])
        ymin, ymax = sorted([y1, y2])
        region_str = f"{xmin:.2f},{ymin:.2f},{xmax:.2f},{ymax:.2f}"
        self.edit_region.setText(region_str)
        self.region_polygon = box(xmin, ymin, xmax, ymax)
        # 更新地图显示，绘制选中区域
        self.update_map()

    def start_computation(self):
        """
        检查数据和计算区域是否已加载，然后读取用户参数，
        创建并启动计算线程进行 Ripley 函数计算
        """
        if self.data is None:
            self.label_data_path.setText("请先加载数据文件")
            return
        if self.region_polygon is None:
            # 若用户未指定计算区域，则默认使用数据点的最小外接矩形
            xmin, ymin = np.min(self.data, axis=0)
            xmax, ymax = np.max(self.data, axis=0)
            self.region_polygon = box(xmin, ymin, xmax, ymax)
            self.edit_region.setText(f"{xmin:.2f},{ymin:.2f},{xmax:.2f},{ymax:.2f}")

        r_min = self.spin_r_min.value()
        r_max = self.spin_r_max.value()
        t_min = self.spin_t_min.value()
        t_max = self.spin_t_max.value()

        # 根据用户选择的边界修正方法，转换为内部代码标识
        boundary_text = self.combo_boundary.currentText()
        if boundary_text == "超环面法":
            boundary_method = 'toroidal'
        elif boundary_text == "缓冲区法":
            boundary_method = 'buffer'
        elif boundary_text == "周长法":
            boundary_method = 'perimeter'
        else:
            boundary_method = 'toroidal'

        # 根据用户选择的计算类型
        compute_type_text = self.combo_compute_type.currentText()
        if compute_type_text == "空间Ripley函数":
            compute_type = 'spatial'
        else:
            compute_type = 'spatio-temporal'

        # 创建计算线程对象，并连接进度和结果信号
        self.computation_thread = ComputationThread(self.data, self.time_data, self.region_polygon,
                                                    r_min, r_max, t_min, t_max, boundary_method, compute_type)
        self.computation_thread.progress_signal.connect(self.update_progress)
        self.computation_thread.result_signal.connect(self.handle_results)
        self.computation_thread.start()

    def update_progress(self, value):
        """
        根据线程中传递的进度值，更新进度条显示
        """
        self.progress_bar.setValue(value)

    def handle_results(self, results):
        """
        接收到计算线程返回的结果后，在结果 Tab 中绘制 K 函数和 L 函数曲线
        如果计算时空 Ripley 函数，也可进一步处理（此处仅简单打印信息）
        """
        if not results:
            print("计算结果为空")
            return
        r_values = results.get('r_values', None)
        K_values = results.get('K', None)
        L_values = results.get('L', None)
        if r_values is not None and K_values is not None and L_values is not None:
            self.ax_K.clear()
            self.ax_K.plot(r_values, K_values, 'b-', label='K函数')
            self.ax_K.set_title("K函数曲线")
            self.ax_K.set_xlabel("r")
            self.ax_K.set_ylabel("K(r)")
            self.ax_K.legend()

            self.ax_L.clear()
            self.ax_L.plot(r_values, L_values, 'r-', label='L函数')
            self.ax_L.set_title("L函数曲线")
            self.ax_L.set_xlabel("r")
            self.ax_L.set_ylabel("L(r)")
            self.ax_L.legend()

            self.canvas_result.draw()
        if 'K_st' in results:
            print("时空Ripley函数计算完成，K_st shape:", results['K_st'].shape)


########################################################################
# 主程序入口
########################################################################
if __name__ == '__main__':
    # 为避免中文乱码，可设置合适的编码（本例中已指定文件头编码）
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
