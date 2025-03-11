import sys
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QVBoxLayout, QWidget, QLabel, \
    QComboBox, QTableWidget, QHeaderView, QTableWidgetItem, QHBoxLayout, QProgressBar, QMessageBox
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt


class RipleyQtApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ripley K & L Function Calculator")
        self.setGeometry(100, 100, 1000, 600)

        self.data = None  # 存储数据

        # 主窗口布局
        layout = QVBoxLayout()

        # 文件选择按钮
        self.btn_load = QPushButton("Load CSV File")
        self.btn_load.clicked.connect(self.load_csv)
        layout.addWidget(self.btn_load)

        # 数据表格
        self.table = QTableWidget()
        self.table.setSizeAdjustPolicy(QTableWidget.AdjustToContents)
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
        self.btn_compute.clicked.connect(self.compute_ripley)
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

    def load_csv(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv);;All Files (*)",
                                                   options=options)
        if file_path:
            try:
                self.progress_bar.setValue(10)
                QApplication.processEvents()
                self.data = pd.read_csv(file_path)
                if self.data.empty:
                    raise ValueError("CSV file is empty or invalid.")
                self.display_data()
                self.progress_bar.setValue(100)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load CSV: {str(e)}")
                self.progress_bar.setValue(0)

    def display_data(self):
        if self.data is not None:
            self.table.setRowCount(self.data.shape[0])
            self.table.setColumnCount(self.data.shape[1])
            self.table.setHorizontalHeaderLabels(self.data.columns)

            # 设置列宽自适应横向铺满
            self.table.horizontalHeader().setStretchLastSection(True)
            self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

            # 设置标题加粗
            font = self.table.horizontalHeader().font()
            font.setBold(True)
            self.table.horizontalHeader().setFont(font)

            for i in range(self.data.shape[0]):
                for j in range(self.data.shape[1]):
                    item = QTableWidgetItem(str(self.data.iat[i, j]))
                    item.setTextAlignment(Qt.AlignCenter)  # 内容居中
                    self.table.setItem(i, j, item)

    def compute_ripley(self):
        if self.data is None:
            QMessageBox.warning(self, "Warning", "No data loaded.")
            return

        self.progress_bar.setValue(10)
        QApplication.processEvents()
        method = self.combo_method.currentText()
        correction = self.combo_correction.currentText()

        # 这里只是示意，实际计算Ripley K和L需要进一步实现
        try:
            x = np.linspace(0, 10, 100)
            y = np.sin(x)

            self.ax.clear()
            self.ax.plot(x, y, label=f"{method} - {correction}")
            self.ax.legend()
            self.canvas.draw()
            self.progress_bar.setValue(100)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Computation failed: {str(e)}")
            self.progress_bar.setValue(0)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RipleyQtApp()
    window.show()
    sys.exit(app.exec_())
