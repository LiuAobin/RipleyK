import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon

# 定义研究区域（矩形边界）
boundary = Polygon([(0, 0), (8, 0), (8, 6), (0, 6)])

# 定义点 i 和 j 的坐标
i = np.array([3, 3])
j = np.array([6, 4])

# 计算距离
radius = np.linalg.norm(j - i)

# 生成圆的边界
theta = np.linspace(0, 2 * np.pi, 300)
circle_x = i[0] + radius * np.cos(theta)
circle_y = i[1] + radius * np.sin(theta)

# 找出圆与矩形边界的交点
circle = Point(i).buffer(radius)  # 以 i 为中心，radius 为半径的圆
intersected_circle = circle.intersection(boundary)  # 计算交集

# 绘图
fig, ax = plt.subplots(figsize=(6, 6))

# 绘制矩形研究区域
x, y = boundary.exterior.xy
ax.plot(x, y, 'k-', label="Study Region")

# 绘制完整圆（蓝色虚线）
ax.plot(circle_x, circle_y, 'b--', alpha=0.5, label="Full Circle")

# 绘制修正后的圆弧（与边界相交部分）
if intersected_circle.geom_type == 'MultiLineString':
    for line in intersected_circle.geoms:
        x, y = line.xy
        ax.plot(x, y, 'b-', linewidth=2, label="Corrected Perimeter")
elif intersected_circle.geom_type == 'LineString':
    x, y = intersected_circle.xy
    ax.plot(x, y, 'b-', linewidth=2, label="Corrected Perimeter")

# 绘制点 i 和 j
ax.scatter(*i, color='red', s=100, label="Point i")
ax.scatter(*j, color='green', s=100, label="Point j")

# 连接点 i 和 j
ax.plot([i[0], j[0]], [i[1], j[1]], 'k--', label="Distance |ij|")

# 标注
ax.text(i[0] + 0.2, i[1], 'i', fontsize=12, color='red')
ax.text(j[0] + 0.2, j[1], 'j', fontsize=12, color='green')

# 设定坐标范围
ax.set_xlim(-1, 9)
ax.set_ylim(-1, 7)
ax.set_aspect('equal')
ax.legend()
plt.title("Ripley's Perimeter Correction Method")

plt.show()
