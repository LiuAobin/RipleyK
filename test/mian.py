import numpy as np
import matplotlib.pyplot as plt

def generate_points(num_points=20, size=10):
    """在 size x size 的区域内生成随机点"""
    x = np.random.uniform(0, size, num_points)
    y = np.random.uniform(0, size, num_points)
    return np.column_stack((x, y))

def toroidal_edge_correction(points, size):
    """对点集进行超环面边界修正，生成8个镜像区域"""
    offsets = [-size, 0, size]  # 偏移量
    replicated_points = []

    for dx in offsets:
        for dy in offsets:
            replicated_points.append(points + np.array([dx, dy]))

    return np.vstack(replicated_points)

# 设定区域大小和点的数量
size = 10
num_points = 20

# 生成原始点
original_points = generate_points(num_points, size)

# 进行超环面边界修正
toroidal_points = toroidal_edge_correction(original_points, size)

# 绘制结果
plt.figure(figsize=(8, 8))
plt.scatter(toroidal_points[:, 0], toroidal_points[:, 1], color='gray', alpha=0.5)
plt.scatter(original_points[:, 0], original_points[:, 1], color='black')

# 绘制原始区域边界
plt.xlim(-size, 2 * size)
plt.ylim(-size, 2 * size)
plt.axvline(0, color='r', linestyle='--')
plt.axvline(size, color='r', linestyle='--')
plt.axhline(0, color='r', linestyle='--')
plt.axhline(size, color='r', linestyle='--')

plt.show()
