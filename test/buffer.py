import numpy as np
import matplotlib.pyplot as plt


def generate_random_points(num_points, x_min, x_max, y_min, y_max):
    """生成在给定范围内的随机点"""
    x_coords = np.random.uniform(x_min, x_max, num_points)
    y_coords = np.random.uniform(y_min, y_max, num_points)
    return np.column_stack((x_coords, y_coords))


def main():
    # 定义外部正方形区域（整体 400 点）
    outer_size = 20  # 外部区域边长
    inner_size = 10  # 内部区域边长
    buffer_size = (outer_size - inner_size) / 2  # 缓冲区宽度

    # 计算点密度（每单位面积点数）
    total_points = 400
    inner_points = 100
    outer_area = outer_size ** 2
    inner_area = inner_size ** 2
    buffer_area = outer_area - inner_area
    buffer_points = total_points - inner_points  # 300 个点

    # 生成点
    inner_points_set = generate_random_points(inner_points, buffer_size, buffer_size + inner_size,
                                              buffer_size, buffer_size + inner_size)
    buffer_points_set = generate_random_points(buffer_points, 0, outer_size, 0, outer_size)

    # 确保 buffer 点不在 inner 区域
    mask = ~((buffer_points_set[:, 0] > buffer_size) & (buffer_points_set[:, 0] < buffer_size + inner_size) &
             (buffer_points_set[:, 1] > buffer_size) & (buffer_points_set[:, 1] < buffer_size + inner_size))
    buffer_points_set = buffer_points_set[mask]

    # 重新补充 buffer points 使总数达到 300
    while len(buffer_points_set) < buffer_points:
        extra_points = generate_random_points(buffer_points - len(buffer_points_set), 0, outer_size, 0, outer_size)
        mask = ~((extra_points[:, 0] > buffer_size) & (extra_points[:, 0] < buffer_size + inner_size) &
                 (extra_points[:, 1] > buffer_size) & (extra_points[:, 1] < buffer_size + inner_size))
        buffer_points_set = np.vstack((buffer_points_set, extra_points[mask]))
    buffer_points_set = buffer_points_set[:buffer_points]

    # 绘制点图
    plt.figure(figsize=(6, 6))
    plt.scatter(buffer_points_set[:, 0], buffer_points_set[:, 1], c='blue', s=10)
    plt.scatter(inner_points_set[:, 0], inner_points_set[:, 1], c='red', s=10 )

    # 绘制边界
    plt.plot([buffer_size, buffer_size + inner_size, buffer_size + inner_size, buffer_size, buffer_size],
             [buffer_size, buffer_size, buffer_size + inner_size, buffer_size + inner_size, buffer_size],)
    plt.xlim(0, outer_size)
    plt.ylim(0, outer_size)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()