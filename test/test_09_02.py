#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
本程序实现了空间 Ripley 函数和时空 Ripley 函数的计算，
支持三种边界修正方法（超环面法、缓冲区法和周长法），
并集成数据文件读取、上海市地图加载（shp文件转换坐标系）、
地图上显示数据点、交互式区域选择以及计算结果的实时显示。
计算过程采用单独线程（QThread），并使用 KDTree 优化数据查询，
同时通过进度条显示数据读取和计算的进度。

功能要求：
1. 页面包括数据文件的读取与显示；
2. Ripley 函数中的半径 r 以及时间 t 的最大值和最小值由用户选择；
3. Ripley 函数的计算区域也由用户选择；
4. 支持边界修正：超环面法、缓冲区法、周长法；
5. 实现空间 Ripley 函数和时空 Ripley 函数；
6. 添加进度条可视化数据读取和计算进度；
7. 计算使用单独线程，避免阻塞主线程；
8. 使用 KDTree 优化数据查询；
9. 读取文件后加载上海市地图（shp文件，原 EPSG:4326），转换为与数据相同的大地坐标系（单位米），并将每个数据点显示到地图上；
10. 地图支持平移、缩放、交互式区域选择，且与用户手动输入的计算区域同步，手动修改区域后地图也相应变换；
11. 用户可以看到所选区域的最小值和最大值；
12. 避免中文乱码；
13. 结果同时显示 K 函数曲线和 L 函数曲线；
14. 确保每个区域都正常显示；
15. 代码添加详细中文注释。

示例数据格式（CSV）：
--------------------------------
longitude,latitude,Date
350911.4148,3450754.43,2022/3/6
348151.7526,3449754.067,2022/3/6
377377.0472,3434023.261,2022/3/6
352301.8765,3458878.786,2022/3/7
361709.9952,3453744.376,2022/3/7
366484.9755,3458879.987,2022/3/7
353834.5328,3460546.683,2022/3/8
337258.3647,3467565.476,2022/3/8
352846.29,3440272.855,2022/3/8
349074.5299,3445284.504,2022/3/9
355033.4159,3457807.576,2022/3/9
--------------------------------
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

########################################################################
# 定义计算线程类，继承 QThread，在后台执行 Ripley 函数的计算工作
########################################################################
class ComputationThread(QThread):
    # 定义信号：progress_signal 用于更新进度；result_signal 用于传递计算结果
    progress_signal = pyqtSignal(int)
    result_signal = pyqtSignal(dict)

    def __init__(self, data, time_data, region_polygon, r_min, r_max, t_min, t_max,
                 boundary_method, compute_type, num_r=50, num_t=10, parent=None):
        """
        初始化计算线程

        参数说明：
        - data: 数据点的二维 numpy 数组，每行 [x, y]
        - time_data: 时间数据，数值型数组（单位秒）
        - region_polygon: 计算区域，shapely Polygon 对象
        - r_min, r_max: Ripley 函数中半径 r 的最小和最大值
        - t_min, t_max: 时空 Ripley 函数中时间 t 的最小和最大值
        - boundary_method: 边界修正方法，取值 'toroidal'（超环面法）、'buffer'（缓冲区法）、'perimeter'（周长法）
        - compute_type: 计算类型，'spatial' 表示空间 Ripley 函数，'spatio-temporal' 表示时空 Ripley 函数
        - num_r: r 的离散取值个数，默认 50
        - num_t: t 的离散取值个数，默认 10（仅时空计算时使用）
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
        并通过 progress_signal 更新进度，通过 result_signal 返回计算结果
        """
        results = {}
        try:
            # 获取计算区域的边界和面积
            minx, miny, maxx, maxy = self.region_polygon.bounds
            area_region = self.region_polygon.area

            # 构造 r 的取值序列（在 [r_min, r_max] 内均匀取值）
            r_values = np.linspace(self.r_min, self.r_max, self.num_r)
            results['r_values'] = r_values

            # 构建 KDTree 加速空间查询
            tree = KDTree(self.data)
            n = len(self.data)
            # 点密度
            lambda_density = n / area_region

            # 若使用缓冲区法，则剔除离边界过近的点（距离边界小于 r_max 的点不计入计算）
            if self.boundary_method == 'buffer':
                interior_indices = []
                for i, point in enumerate(self.data):
                    pt = Point(point)
                    if self.region_polygon.contains(pt) and pt.distance(self.region_polygon.boundary) >= self.r_max:
                        interior_indices.append(i)
                data_indices = np.array(interior_indices)
            else:
                data_indices = np.arange(n)

            # 初始化 K 函数数组（空间函数）
            K_values = np.zeros_like(r_values, dtype=float)

            # 若计算时空 Ripley 函数，则初始化 t 值数组及二维 K 函数数组
            if self.compute_type == 'spatio-temporal':
                t_values = np.linspace(self.t_min, self.t_max, self.num_t)
                results['t_values'] = t_values
                K_st = np.zeros((self.num_r, self.num_t), dtype=float)

            # -------------------------------
            # 计算空间 Ripley 函数
            # -------------------------------
            for idx, r in enumerate(r_values):
                sum_val = 0.0
                # 如果使用周长法，预先对每个数据点计算校正因子（采样圆周点在区域内的比例倒数）
                if self.boundary_method == 'perimeter':
                    num_samples = 100  # 采样点数
                    angles = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
                    unit_offsets = np.column_stack((np.cos(angles), np.sin(angles)))
                    correction_factors = np.zeros(n, dtype=float)
                    for i in range(n):
                        center = self.data[i]
                        circle_points = center + r * unit_offsets  # 构造圆周采样点
                        inside = 0
                        for pt in circle_points:
                            if self.region_polygon.contains(Point(pt)):
                                inside += 1
                        fraction = inside / num_samples
                        # 避免除零
                        if fraction == 0:
                            fraction = 1e-6
                        correction_factors[i] = 1 / fraction

                # 对于每个数据点（或剔除边界效应后的数据点），统计其它点中距离小于 r 的个数
                for i in data_indices:
                    point = self.data[i]
                    if self.boundary_method == 'toroidal':
                        # 超环面法：考虑区域平铺形成的环面效果
                        Lx = maxx - minx
                        Ly = maxy - miny
                        diff = np.abs(self.data - point)
                        # 分别取 x、y 方向的最小距离（直线距离或环面距离）
                        diff[:, 0] = np.minimum(diff[:, 0], Lx - diff[:, 0])
                        diff[:, 1] = np.minimum(diff[:, 1], Ly - diff[:, 1])
                        distances = np.sqrt(diff[:, 0]**2 + diff[:, 1]**2)
                        distances[i] = np.inf  # 排除自身
                        count = np.sum(distances <= r)
                    else:
                        # 使用 KDTree 查询邻域点
                        indices = tree.query_ball_point(point, r)
                        if i in indices:
                            indices.remove(i)
                        count = len(indices)
                        if self.boundary_method == 'perimeter':
                            count *= correction_factors[i]
                    sum_val += count

                # 利用无偏估计公式计算 Ripley K 函数
                if n > 1:
                    K_r = area_region / (n * (n - 1)) * sum_val
                else:
                    K_r = 0
                K_values[idx] = K_r

                # 更新进度（假定空间函数计算占总进度的50%）
                progress = int(50 * (idx + 1) / self.num_r)
                self.progress_signal.emit(progress)

            # -------------------------------
            # 若计算时空 Ripley 函数，则进一步计算 K(r, t)
            # -------------------------------
            if self.compute_type == 'spatio-temporal':
                # 假设 time_data 已经为数值型（单位秒），取整个时间段长度
                T_range = self.time_data.max() - self.time_data.min()
                for i_r, r in enumerate(r_values):
                    # 如果使用周长法，重新计算校正因子（与上面类似）
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
                                distances = np.sqrt(diff[:, 0]**2 + diff[:, 1]**2)
                                distances[i] = np.inf
                                spatial_indices = np.where(distances <= r)[0]
                            else:
                                spatial_indices = tree.query_ball_point(point, r)
                                if i in spatial_indices:
                                    spatial_indices.remove(i)
                            # 对符合空间条件的点，再判断时间差是否满足条件
                            time_diffs = np.abs(self.time_data[spatial_indices] - self.time_data[i])
                            temporal_count = np.sum(time_diffs <= t)
                            if self.boundary_method == 'perimeter':
                                temporal_count *= correction_factors[i]
                            sum_val_st += temporal_count
                        if n > 1:
                            K_rt = area_region * T_range / (n * (n - 1)) * sum_val_st
                        else:
                            K_rt = 0
                        K_st[i_r, j_t] = K_rt
                        # 更新进度（时空函数计算占总进度后 50%）
                        progress = 50 + int(50 * ((i_r * self.num_t + j_t + 1) / (self.num_r * self.num_t)))
                        self.progress_signal.emit(progress)
                results['K_st'] = K_st

            # 保存空间函数计算结果
            results['K'] = K_values
            # 计算 L 函数：L(r)=sqrt(K(r)/π)
            L_values = np.sqrt(K_values / np.pi)
            results['L'] = L_values

            # 计算完成，发送结果信号
            self.result_signal.emit(results)
        except Exception as e:
            print("计算过程中发生错误：", e)
            self.result_signal.emit({})

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
        self.spin_r_min.setValue(0)
        param_layout.addWidget(label_r_min, 0, 0)
        param_layout.addWidget(self.spin_r_min, 0, 1)

        # r 最大值
        label_r_max = QLabel("r 最大值:")
        self.spin_r_max = QDoubleSpinBox()
        self.spin_r_max.setRange(0, 1e6)
        self.spin_r_max.setValue(1000)
        param_layout.addWidget(label_r_max, 1, 0)
        param_layout.addWidget(self.spin_r_max, 1, 1)

        # t 最小值
        label_t_min = QLabel("t 最小值:")
        self.spin_t_min = QDoubleSpinBox()
        self.spin_t_min.setRange(0, 1e6)
        self.spin_t_min.setValue(0)
        param_layout.addWidget(label_t_min, 2, 0)
        param_layout.addWidget(self.spin_t_min, 2, 1)

        # t 最大值
        label_t_max = QLabel("t 最大值:")
        self.spin_t_max = QDoubleSpinBox()
        self.spin_t_max.setRange(0, 1e6)
        self.spin_t_max.setValue(10)
        param_layout.addWidget(label_t_max, 3, 0)
        param_layout.addWidget(self.spin_t_max, 3, 1)

        # 计算区域输入（xmin, ymin, xmax, ymax），用户可手动输入
        label_region = QLabel("计算区域 (xmin,ymin,xmax,ymax):")
        self.edit_region = QLineEdit()
        self.edit_region.setPlaceholderText("例如: 300000,3400000,400000,3500000")
        param_layout.addWidget(label_region, 4, 0, 1, 2)

        # 当用户手动修改区域时，同步更新显示区域的最值信息
        self.edit_region.textChanged.connect(self.update_region_from_text)

        # 显示当前区域的最大最小值
        self.label_region_info = QLabel("当前区域：未选择")
        param_layout.addWidget(self.label_region_info, 5, 0, 1, 2)

        # 边界修正方法选择
        label_boundary = QLabel("边界修正方法:")
        self.combo_boundary = QComboBox()
        self.combo_boundary.addItems(["超环面法", "缓冲区法", "周长法"])
        param_layout.addWidget(label_boundary, 6, 0)
        param_layout.addWidget(self.combo_boundary, 6, 1)

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
            self.label_region_info.setText(f"当前区域：xmin={xmin:.2f}, ymin={ymin:.2f}, xmax={xmax:.2f}, ymax={ymax:.2f}")
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
        region_str = f"{xmin:.2f},{ymin:.2f},{xmax:.2f},{ymax:.2f}"
        self.edit_region.setText(region_str)
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
                    self.label_region_info.setText(f"当前区域：xmin={xmin:.2f}, ymin={ymin:.2f}, xmax={xmax:.2f}, ymax={ymax:.2f}")
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
        # 如果用户未选择计算区域，则默认使用数据的外包矩形
        if self.region_polygon is None:
            xmin, ymin = np.min(self.data, axis=0)
            xmax, ymax = np.max(self.data, axis=0)
            self.region_polygon = box(xmin, ymin, xmax, ymax)
            self.edit_region.setText(f"{xmin:.2f},{ymin:.2f},{xmax:.2f},{ymax:.2f}")
            self.label_region_info.setText(f"当前区域：xmin={xmin:.2f}, ymin={ymin:.2f}, xmax={xmax:.2f}, ymax={ymax:.2f}")

        r_min = self.spin_r_min.value()
        r_max = self.spin_r_max.value()
        t_min = self.spin_t_min.value()
        t_max = self.spin_t_max.value()

        # 根据用户选择的边界修正方法转换为内部标识
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
        if compute_type_text == "空间 Ripley 函数":
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
        根据线程发来的进度值，更新进度条显示
        """
        self.progress_bar.setValue(value)

    def handle_results(self, results):
        """
        接收到计算线程返回的结果后，在结果 Tab 中绘制 K 函数和 L 函数曲线；
        如果计算时空 Ripley 函数，则在控制台打印相应信息（可进一步扩展显示）。
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
            self.ax_K.set_title("K 函数曲线")
            self.ax_K.set_xlabel("r")
            self.ax_K.set_ylabel("K(r)")
            self.ax_K.legend()

            self.ax_L.clear()
            self.ax_L.plot(r_values, L_values, 'r-', label='L函数')
            self.ax_L.set_title("L 函数曲线")
            self.ax_L.set_xlabel("r")
            self.ax_L.set_ylabel("L(r)")
            self.ax_L.legend()

            self.canvas_result.draw()

        if 'K_st' in results:
            print("时空 Ripley 函数计算完成，K_st shape:", results['K_st'])

########################################################################
# 主程序入口
########################################################################
if __name__ == '__main__':
    # 创建 QApplication 对象
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
