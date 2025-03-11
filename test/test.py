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
import numpy as np
import cupy as cp
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
from shapely.geometry import Polygon
from pyproj import Transformer
from scipy.spatial import KDTree, cKDTree
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.widgets import RectangleSelector
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QProgressBar, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QLineEdit, QComboBox, QDoubleSpinBox, QGroupBox, QGridLayout, QTabWidget
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from mpl_toolkits.mplot3d import Axes3D  # 3D 绘图支持

# 设置 Matplotlib 使用支持中文的字体，避免中文乱码
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

class ComputationThread(QThread):
    progress_signal = pyqtSignal(int)  # 发射进度百分比
    result_signal = pyqtSignal(object)  # 发射计算结果

    def __init__(self, points, times, region, r_min, r_max, r_step, t_min, t_max, t_step, analysis_type, correction_method, parent=None):
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
        """ 计算空间或时空 Ripley 函数 """
        if self.analysis_type == "spatial":
            r_vals, K_vals = self.compute_spatial_ripley()
            results = {"analysis_type": "spatial", "r": r_vals, "K": K_vals}
        else:
            r_vals, t_vals, K_st = self.compute_spatiotemporal_ripley()
            results = {"analysis_type": "spatiotemporal", "r": r_vals, "t": t_vals, "K": K_st}
        self.result_signal.emit(results)

    def compute_spatial_ripley(self):
        points = self.points
        n = len(points)
        A = (self.region[1] - self.region[0]) * (self.region[3] - self.region[2])
        r_values = cp.arange(self.r_min, self.r_max + self.r_step, self.r_step)
        K_values = cp.zeros_like(r_values, dtype=float)
        num_r = len(r_values)

        if self.correction_method == "torus":
            Lx = self.region[1] - self.region[0]
            Ly = self.region[3] - self.region[2]
            offsets = np.array([[dx, dy] for dx in (-Lx, 0, Lx) for dy in (-Ly, 0, Ly)])
            replicated_points = cp.concatenate([points + offset for offset in offsets], axis=0)
            tree = cKDTree(replicated_points)
            for idx, r in enumerate(r_values):
                total = 0
                for i, p in enumerate(points):
                    neighbors = tree.query_ball_point(p, r)
                    total += len(neighbors) - 1  # 除去自己
                K_values[idx] = A / (n * (n - 1)) * total
                self.progress_signal.emit(int(100 * (idx + 1) / num_r))

        # 使用缓冲区法、周长法等类似的逻辑处理
        # 省略了“buffer”和“perimeter”方法的具体实现，按需调整
        return r_values, K_values

    def compute_spatiotemporal_ripley(self):
        points = self.points
        times = self.times
        n = len(points)
        A = (self.region[1] - self.region[0]) * (self.region[3] - self.region[2])
        T_total = np.max(times) - np.min(times)
        V = A * T_total
        r_values = cp.arange(self.r_min, self.r_max + self.r_step, self.r_step)
        t_values = cp.arange(self.t_min, self.t_max + self.t_step, self.t_step)
        K_st = cp.zeros((len(r_values), len(t_values)), dtype=float)
        num_total = len(r_values) * len(t_values)
        count_calculations = 0
        tree = cKDTree(points)
        for i_r, r in enumerate(r_values):
            for i_t, t_lim in enumerate(t_values):
                total = 0
                for i, p in enumerate(points):
                    count = 0
                    for j, q in enumerate(points):
                        if i == j:
                            continue
                        dx = abs(p[0] - q[0])
                        dy = abs(p[1] - q[1])
                        d = math.hypot(dx, dy)
                        if d <= r and abs(times[i] - times[j]) <= t_lim:
                            count += 1
                    total += count
                K_st[i_r, i_t] = V / (n * (n - 1)) * total
                count_calculations += 1
                self.progress_signal.emit(int(100 * count_calculations / num_total))

        return r_values, t_values, K_st

# 主窗口类
class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("空间/时空 Ripley 函数分析")
        self.resize(1200, 800)
        self.data = None
        self.time_data = None
        self.shanghai_map = None
        self.region_polygon = None
        self._isPanning = False
        self._panStart = None
        self._xlim_start = None
        self._ylim_start = None
        self.computation_thread = None
        self.initUI()

    def initUI(self):
        """ 构建主界面，左侧为参数控制面板，右侧为 Tab 页（地图显示、结果显示） """
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)

        # 创建 UI 控件（文件加载、参数设置、计算按钮、进度条）
        # 省略了具体控件的创建代码，请根据需要调整

        # 结果显示：时空 Ripley 函数的三维图
        self.result_tab = QWidget()
        result_layout = QVBoxLayout(self.result_tab)
        self.fig_result = plt.Figure(figsize=(7, 5))
        self.canvas_result = FigureCanvas(self.fig_result)
        result_layout.addWidget(self.canvas_result)

        self.ax_K = self.fig_result.add_subplot(111, projection='3d')  # 使用 3D 投影
        self.ax_K.set_title("K 函数三维曲面")
        self.ax_K.set_xlabel("r")
        self.ax_K.set_ylabel("t")
        self.ax_K.set_zlabel("K(r, t)")

        # 显示结果的函数
        self.result_tab.addTab(self.result_tab, "结果显示")
        main_layout.addWidget(control_widget, 2)
        main_layout.addWidget(self.tab_widget, 5)

        self.setCentralWidget(main_widget)

    def handle_results(self, results):
        """
        显示时空 Ripley 函数的三维图
        """
        if 'K_st' in results:
            r_values = results['r']
            t_values = results['t']
            K_st = results['K']

            r, t = np.meshgrid(r_values, t_values)
            self.ax_K.plot_surface(r, t, K_st, cmap='viridis')
            self.canvas_result.draw()

    def start_computation(self):
        """ 启动计算线程，开始计算空间或时空 Ripley 函数 """
        if self.data is None:
            self.label_data_path.setText("请先加载数据文件")
            return

        # 获取计算区域和参数设置
        xmin, ymin = np.min(self.data, axis=0)
        xmax, ymax = np.max(self.data, axis=0)
        self.region_polygon = box(xmin, ymin, xmax, ymax)

        # 创建计算线程对象，并连接进度和结果信号
        r_min = 50
        r_max = 1000
        r_step = 50
        t_min = 1
        t_max = 10
        t_step = 1
        boundary_method = "torus"  # 选择边界修正方法
        compute_type = "spatial"  # 或 "spatiotemporal"

        # 创建计算线程
        self.computation_thread = ComputationThread(self.data, self.time_data, (xmin, xmax, ymin, ymax),
                                                    r_min, r_max, r_step, t_min, t_max, t_step,
                                                    compute_type, boundary_method)
        self.computation_thread.progress_signal.connect(self.update_progress)
        self.computation_thread.result_signal.connect(self.handle_results)
        self.computation_thread.start()

    def update_progress(self, value):
        """ 更新进度条 """
        self.progress_bar.setValue(value)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
