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

import sys
import os
import math
import datetime
import numpy as np
from scipy.spatial import cKDTree  # kd树，用于加速空间邻域查询
from shapely.geometry import Point, box  # 用于计算圆与区域交集面积（周长法）

# 使用 GeoPandas 读取地图数据
import geopandas as gpd

# PyQt5 相关模块
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QProgressBar, QLabel, QLineEdit, QTableWidget, QTableWidgetItem,
    QRadioButton, QButtonGroup, QDoubleSpinBox, QGroupBox, QGridLayout
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# matplotlib 用于绘图
import matplotlib
matplotlib.use("Qt5Agg")
# 设置中文字体（如系统中无 SimHei，请替换为可用的中文字体）
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches  # 用于绘制矩形区域
from matplotlib.widgets import RectangleSelector  # 用于区域选择

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
            result = {
                "analysis_type": "spatial",
                "r": r_vals,
                "K": K_vals
            }
        else:
            r_vals, t_vals, K_st = self.compute_spatiotemporal_ripley()
            result = {
                "analysis_type": "spatiotemporal",
                "r": r_vals,
                "t": t_vals,
                "K": K_st
            }
        self.result_signal.emit(result)

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

##############################################
# 主窗口：包含数据读取、参数设置、地图显示及结果显示等界面元素
##############################################
class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("上海市 Ripley 函数分析")
        self.resize(1200, 800)

        # 主窗口中心部件
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QHBoxLayout()
        self.central_widget.setLayout(main_layout)

        # 左侧：参数设置和数据表
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_widget.setLayout(left_layout)

        # “读取数据”按钮
        self.load_button = QPushButton("读取数据")
        self.load_button.clicked.connect(self.load_data)
        left_layout.addWidget(self.load_button)

        # 数据显示表格
        self.table = QTableWidget()
        left_layout.addWidget(self.table, stretch=1)

        # 参数设置区域（使用 QGroupBox 分组）
        param_group = QGroupBox("参数设置")
        param_layout = QGridLayout()
        param_group.setLayout(param_layout)
        left_layout.addWidget(param_group)

        # 研究区域输入：xmin, xmax, ymin, ymax
        param_layout.addWidget(QLabel("区域 xmin:"), 0, 0)
        self.xmin_edit = QLineEdit()
        param_layout.addWidget(self.xmin_edit, 0, 1)
        param_layout.addWidget(QLabel("xmax:"), 0, 2)
        self.xmax_edit = QLineEdit()
        param_layout.addWidget(self.xmax_edit, 0, 3)
        param_layout.addWidget(QLabel("区域 ymin:"), 1, 0)
        self.ymin_edit = QLineEdit()
        param_layout.addWidget(self.ymin_edit, 1, 1)
        param_layout.addWidget(QLabel("ymax:"), 1, 2)
        self.ymax_edit = QLineEdit()
        param_layout.addWidget(self.ymax_edit, 1, 3)

        # 空间半径参数：r_min, r_max, r_step
        param_layout.addWidget(QLabel("r 最小值:"), 2, 0)
        self.rmin_spin = QDoubleSpinBox()
        self.rmin_spin.setRange(0, 1e6)
        self.rmin_spin.setValue(100)
        param_layout.addWidget(self.rmin_spin, 2, 1)
        param_layout.addWidget(QLabel("r 最大值:"), 2, 2)
        self.rmax_spin = QDoubleSpinBox()
        self.rmax_spin.setRange(0, 1e6)
        self.rmax_spin.setValue(1000)
        param_layout.addWidget(self.rmax_spin, 2, 3)
        param_layout.addWidget(QLabel("r 步长:"), 3, 0)
        self.rstep_spin = QDoubleSpinBox()
        self.rstep_spin.setRange(1, 1e5)
        self.rstep_spin.setValue(50)
        param_layout.addWidget(self.rstep_spin, 3, 1)

        # 时间参数（仅时空分析使用）：t_min, t_max, t_step
        param_layout.addWidget(QLabel("t 最小值:"), 4, 0)
        self.tmin_spin = QDoubleSpinBox()
        self.tmin_spin.setRange(0, 1e4)
        self.tmin_spin.setValue(0)
        param_layout.addWidget(self.tmin_spin, 4, 1)
        param_layout.addWidget(QLabel("t 最大值:"), 4, 2)
        self.tmax_spin = QDoubleSpinBox()
        self.tmax_spin.setRange(0, 1e4)
        self.tmax_spin.setValue(5)
        param_layout.addWidget(self.tmax_spin, 4, 3)
        param_layout.addWidget(QLabel("t 步长:"), 5, 0)
        self.tstep_spin = QDoubleSpinBox()
        self.tstep_spin.setRange(0.1, 1e4)
        self.tstep_spin.setValue(1)
        param_layout.addWidget(self.tstep_spin, 5, 1)

        # 分析类型选择：空间分析 vs 时空分析
        analysis_label = QLabel("分析类型:")
        param_layout.addWidget(analysis_label, 6, 0)
        self.spatial_radio = QRadioButton("空间分析")
        self.spatial_radio.setChecked(True)
        self.spatiotemporal_radio = QRadioButton("时空分析")
        analysis_group = QButtonGroup()
        analysis_group.addButton(self.spatial_radio)
        analysis_group.addButton(self.spatiotemporal_radio)
        param_layout.addWidget(self.spatial_radio, 6, 1)
        param_layout.addWidget(self.spatiotemporal_radio, 6, 2)

        # 边界修正方法选择：超环面法、缓冲区法、周长法
        method_label = QLabel("边界修正方法:")
        param_layout.addWidget(method_label, 7, 0)
        self.torus_radio = QRadioButton("超环面法")
        self.torus_radio.setChecked(True)
        self.buffer_radio = QRadioButton("缓冲区法")
        self.perimeter_radio = QRadioButton("周长法")
        method_group = QButtonGroup()
        method_group.addButton(self.torus_radio)
        method_group.addButton(self.buffer_radio)
        method_group.addButton(self.perimeter_radio)
        param_layout.addWidget(self.torus_radio, 7, 1)
        param_layout.addWidget(self.buffer_radio, 7, 2)
        param_layout.addWidget(self.perimeter_radio, 7, 3)

        # “计算 Ripley 函数”按钮
        self.compute_button = QPushButton("计算 Ripley 函数")
        self.compute_button.clicked.connect(self.start_computation)
        left_layout.addWidget(self.compute_button)

        # 进度条
        self.progress_bar = QProgressBar()
        left_layout.addWidget(self.progress_bar)

        main_layout.addWidget(left_widget, stretch=0)

        # 右侧：绘图区域及工具栏
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_widget.setLayout(right_layout)
        self.figure = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas)
        main_layout.addWidget(right_widget, stretch=1)

        # 用于存储数据
        self.data_points = None   # numpy 数组 (N,2)
        self.data_times = None    # numpy 数组 (N,)（以 float 表示日期）
        self.shanghai_map = None  # 存储上海市地图数据（GeoDataFrame）
        self.RS = None          # RectangleSelector 对象

    def load_data(self):
        """
        打开 CSV 文件，读取数据（格式：longitude,latitude,Date），更新数据表，
        同时在地图上显示数据点
        """
        file_path, _ = QFileDialog.getOpenFileName(self, "选择数据文件", "", "CSV Files (*.csv *.txt)")
        if not file_path:
            return
        try:
            data = np.genfromtxt(file_path, delimiter=",", dtype=str, skip_header=1)
            longitudes = data[:, 0].astype(float)
            latitudes = data[:, 1].astype(float)
            dates = []
            for d in data[:, 2]:
                dt = datetime.datetime.strptime(d.strip(), "%Y/%m/%d")
                dates.append(dt.toordinal())
            dates = np.array(dates, dtype=float)
            self.data_points = np.column_stack((longitudes, latitudes))
            self.data_times = dates

            # 更新数据表
            self.table.clear()
            self.table.setColumnCount(3)
            self.table.setHorizontalHeaderLabels(["longitude", "latitude", "Date"])
            self.table.setRowCount(len(data))
            for i in range(len(data)):
                self.table.setItem(i, 0, QTableWidgetItem(str(longitudes[i])))
                self.table.setItem(i, 1, QTableWidgetItem(str(latitudes[i])))
                self.table.setItem(i, 2, QTableWidgetItem(str(data[i, 2])))
            self.table.resizeColumnsToContents()

            # 自动填充研究区域（可稍作扩展）
            xmin = np.min(longitudes) - 10
            xmax = np.max(longitudes) + 10
            ymin = np.min(latitudes) - 10
            ymax = np.max(latitudes) + 10
            self.xmin_edit.setText(str(xmin))
            self.xmax_edit.setText(str(xmax))
            self.ymin_edit.setText(str(ymin))
            self.ymax_edit.setText(str(ymax))

            # 绘制地图及数据点
            self.draw_map_view()

        except Exception as e:
            print("读取数据错误：", e)

    def draw_map_view(self):
        """
        在右侧 canvas 上绘制上海市地图及数据点，
        并添加 RectangleSelector 以支持区域选择和缩放
        """
        self.figure.clear()
        self.ax_map = self.figure.add_subplot(111)
        self.ax_map.set_title("上海市地图 - 数据点")
        # 尝试加载上海市地图 shapefile（请确保文件路径正确）
        try:
            # 假设上海市地图文件位于当前目录下，文件名为 shanghai.shp
            self.shanghai_map = gpd.read_file("shanghai/Shanghai.shp")
            # if self.shanghai_map.crs is not None and self.shanghai_map.crs.to_epsg() != 4326:
            self.shanghai_map = self.shanghai_map.to_crs(epsg=3097)  # 转换为 WGS 84
            self.shanghai_map.plot(ax=self.ax_map, color='lightgray', edgecolor='black')
        except Exception as e:
            print("加载上海市地图失败：", e)
        # 绘制数据点（若已加载）
        if self.data_points is not None:
            self.ax_map.scatter(self.data_points[:, 0], self.data_points[:, 1],
                                c='blue', marker='o', label="数据点")
        self.ax_map.legend()
        self.canvas.draw()

        # 添加 RectangleSelector（仅创建一次）
        if self.RS is None:
            self.RS = RectangleSelector(self.ax_map, self.onselect,
                                        useblit=True, button=[1],
                                        minspanx=5, minspany=5, spancoords='pixels',
                                        interactive=True)

    def onselect(self, eclick, erelease):
        """
        RectangleSelector 回调：鼠标按下和释放时获取区域坐标，
        更新参数文本框，并将地图视图缩放到所选区域
        """
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        if None in (x1, y1, x2, y2):
            return
        xmin = min(x1, x2)
        xmax = max(x1, x2)
        ymin = min(y1, y2)
        ymax = max(y1, y2)
        self.xmin_edit.setText(str(round(xmin, 2)))
        self.xmax_edit.setText(str(round(xmax, 2)))
        self.ymin_edit.setText(str(round(ymin, 2)))
        self.ymax_edit.setText(str(round(ymax, 2)))
        self.ax_map.set_xlim(xmin, xmax)
        self.ax_map.set_ylim(ymin, ymax)
        self.canvas.draw()

    def start_computation(self):
        """
        收集参数后启动计算线程，计算完成后在右侧显示结果：
          - 空间分析：上部显示数据点与区域，下部分别显示 K 函数与 L 函数（左右两个子图）
          - 时空分析：上部显示数据点与区域，下部显示时空热图
        """
        if self.data_points is None:
            print("请先加载数据！")
            return
        try:
            region = (float(self.xmin_edit.text()), float(self.xmax_edit.text()),
                      float(self.ymin_edit.text()), float(self.ymax_edit.text()))
        except Exception as e:
            print("区域参数输入错误：", e)
            return

        r_min = self.rmin_spin.value()
        r_max = self.rmax_spin.value()
        r_step = self.rstep_spin.value()
        t_min = self.tmin_spin.value()
        t_max = self.tmax_spin.value()
        t_step = self.tstep_spin.value()

        if self.spatial_radio.isChecked():
            analysis_type = "spatial"
            times = None
        else:
            analysis_type = "spatiotemporal"
            times = self.data_times

        if self.torus_radio.isChecked():
            correction_method = "torus"
        elif self.buffer_radio.isChecked():
            correction_method = "buffer"
        elif self.perimeter_radio.isChecked():
            correction_method = "perimeter"
        else:
            correction_method = "torus"

        self.progress_bar.setValue(0)
        self.comp_thread = ComputationThread(
            points=self.data_points,
            times=times,
            region=region,
            r_min=r_min,
            r_max=r_max,
            r_step=r_step,
            t_min=t_min,
            t_max=t_max,
            t_step=t_step,
            analysis_type=analysis_type,
            correction_method=correction_method
        )
        self.comp_thread.progress_signal.connect(self.update_progress)
        self.comp_thread.result_signal.connect(self.handle_result)
        self.comp_thread.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def handle_result(self, result):
        """
        计算完成后，根据分析类型在右侧绘图区域中显示结果：
          - 空间分析：利用 GridSpec 将图形分为三部分，
            第一部分显示数据点与选定区域，
            第二部分绘制 K 函数曲线，
            第三部分绘制 L 函数曲线（其中 L(r)=sqrt(K(r)/π)-r）
          - 时空分析：上部显示数据点与区域，下部显示时空热图
        """
        self.figure.clear()
        try:
            xmin = float(self.xmin_edit.text())
            xmax = float(self.xmax_edit.text())
            ymin = float(self.ymin_edit.text())
            ymax = float(self.ymax_edit.text())
        except Exception:
            xmin, xmax = self.data_points[:, 0].min(), self.data_points[:, 0].max()
            ymin, ymax = self.data_points[:, 1].min(), self.data_points[:, 1].max()

        if result["analysis_type"] == "spatial":
            r = result["r"]
            K = result["K"]
            L = np.sqrt(K/np.pi) - r

            gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
            ax_map = self.figure.add_subplot(gs[0, :])
            ax_k = self.figure.add_subplot(gs[1, 0])
            ax_l = self.figure.add_subplot(gs[1, 1])

            # 上部显示数据点与区域
            ax_map.scatter(self.data_points[:, 0], self.data_points[:, 1],
                           c='blue', marker='o', label="数据点")
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                     fill=False, edgecolor='red', linewidth=2,
                                     label="选定区域")
            ax_map.add_patch(rect)
            ax_map.set_title("数据点与选定区域")
            ax_map.set_xlabel("经度")
            ax_map.set_ylabel("纬度")
            ax_map.legend()

            # 左下绘制 K 函数
            ax_k.plot(r, K, marker='o', color='blue', linestyle='-')
            ax_k.set_title("K 函数")
            ax_k.set_xlabel("r")
            ax_k.set_ylabel("K(r)")

            # 右下绘制 L 函数
            ax_l.plot(r, L, marker='x', color='red', linestyle='--')
            ax_l.set_title("L 函数 (sqrt(K/π)-r)")
            ax_l.set_xlabel("r")
            ax_l.set_ylabel("L(r)")

        else:
            ax_map = self.figure.add_subplot(211)
            ax_heat = self.figure.add_subplot(212)
            ax_map.scatter(self.data_points[:, 0], self.data_points[:, 1],
                           c='blue', marker='o', label="数据点")
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                     fill=False, edgecolor='red', linewidth=2,
                                     label="选定区域")
            ax_map.add_patch(rect)
            ax_map.set_title("数据点与选定区域")
            ax_map.set_xlabel("经度")
            ax_map.set_ylabel("纬度")
            ax_map.legend()

            r = result["r"]
            t = result["t"]
            K_st = result["K"]
            im = ax_heat.imshow(K_st, extent=[t[0], t[-1], r[0], r[-1]],
                                aspect='auto', origin='lower')
            ax_heat.set_xlabel("t")
            ax_heat.set_ylabel("r")
            ax_heat.set_title("时空 Ripley 函数")
            self.figure.colorbar(im, ax=ax_heat)

        self.canvas.draw()

##############################################
# 主程序入口
##############################################
def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
