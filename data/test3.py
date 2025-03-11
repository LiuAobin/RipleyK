import sys
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QVBoxLayout, QWidget, QLabel, \
    QComboBox, QTableWidget, QHeaderView, QTableWidgetItem, QProgressBar, QMessageBox
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from scipy.spatial import KDTree


class RipleyComputationThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(list, list, str, str)
    canceled = pyqtSignal()

    def __init__(self, data, method, correction):
        super().__init__()
        self.data = data
        self.method = method
        self.correction = correction
        self.running = True

    def apply_boundary_correction(self, coords, max_dist):
        if self.correction == "Torus":
            coords = np.vstack([coords, coords + max_dist, coords - max_dist])
        elif self.correction == "Buffer":
            buffer_size = max_dist * 0.1
            coords = np.vstack([coords, coords + buffer_size, coords - buffer_size])
        elif self.correction == "Perimeter":
            pass  # Placeholder for perimeter correction
        return coords

    def run(self):
        try:
            coords = self.data[['longitude', 'latitude']].values
            times = (self.data['Date'] - self.data['Date'].min()).dt.total_seconds().values
            max_dist = np.linalg.norm(coords.max(axis=0) - coords.min(axis=0)) / 2

            coords = self.apply_boundary_correction(coords, max_dist)
            tree = KDTree(coords)

            if self.method == "Spatiotemporal Ripley":
                max_time = times.max() - times.min()
                time_intervals = np.linspace(0, max_time, 50)
                radii = np.linspace(0, max_dist, 50)
                K_values = []

                for i, r in enumerate(radii):
                    if not self.running:
                        self.canceled.emit()
                        return
                    self.progress.emit(int((i / len(radii)) * 90))
                    counts = tree.query_ball_tree(tree, r)
                    count_sum = sum(len(c) - 1 for c in counts)  # Exclude self-count
                    K_values.append(count_sum / len(coords))

                L_values = np.sqrt(K_values / np.pi)
                self.finished.emit(radii.tolist(), L_values.tolist(), self.method, self.correction)
            else:
                radii = np.linspace(0, max_dist, 50)
                K_values = []

                for i, r in enumerate(radii):
                    if not self.running:
                        self.canceled.emit()
                        return
                    self.progress.emit(int((i / len(radii)) * 90))
                    counts = tree.query_ball_tree(tree, r)
                    count_sum = sum(len(c) - 1 for c in counts)  # Exclude self-count
                    K_values.append(count_sum / len(coords))

                L_values = np.sqrt(K_values / np.pi)
                self.finished.emit(radii.tolist(), L_values.tolist(), self.method, self.correction)
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Computation failed: {str(e)}")
            self.progress.emit(0)

    def cancel(self):
        self.running = False


class RipleyQtApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ripley K & L Function Calculator")
        self.setGeometry(100, 100, 1000, 600)

        self.data = None  # 存储数据
        self.thread = None

        # 主窗口布局
        layout = QVBoxLayout()

        # 文件选择按钮
        self.btn_load = QPushButton("Load CSV File")
        self.btn_load.clicked.connect(self.load_csv)
        layout.addWidget(self.btn_load)

        # 数据表格
        self.table = QTableWidget()
        layout.addWidget(self.table)

        # 计算类型选择
        self.label_method = QLabel("Select Method:")
        layout.addWidget(self.label_method)

        self.combo_method = QComboBox()
        self.combo_method.addItems(["Spatial Ripley", "Spatiotemporal Ripley"])
        layout.addWidget(self.combo_method)

        # 边界修正方法选择
        self.label_correction = QLabel("Select Boundary Correction:")
        layout.addWidget(self.label_correction)

        self.combo_correction = QComboBox()
        self.combo_correction.addItems(["None", "Torus", "Buffer", "Perimeter"])
        layout.addWidget(self.combo_correction)

        # 计算按钮
        self.btn_compute = QPushButton("Compute Ripley Functions")
        self.btn_compute.clicked.connect(self.start_computation)
        layout.addWidget(self.btn_compute)

        # Matplotlib 画布
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # 设置主窗口
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def start_computation(self):
        if self.data is None:
            QMessageBox.warning(self, "Warning", "No data loaded.")
            return

        # 取消当前正在运行的计算
        if self.thread and self.thread.isRunning():
            self.thread.cancel()
            self.thread.wait()

        method = self.combo_method.currentText()
        correction = self.combo_correction.currentText()

        self.thread = RipleyComputationThread(self.data, method, correction)
        self.thread.progress.connect(self.progress_bar.setValue)
        self.thread.finished.connect(self.plot_results)
        self.thread.canceled.connect(self.on_computation_canceled)
        self.thread.start()

    def plot_results(self, radii, L_values, method, correction):
        self.ax.clear()
        self.ax.plot(radii, L_values, label=f"{method} - {correction}")
        self.ax.legend()
        self.canvas.draw()
        self.progress_bar.setValue(100)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RipleyQtApp()
    window.show()
    sys.exit(app.exec_())
