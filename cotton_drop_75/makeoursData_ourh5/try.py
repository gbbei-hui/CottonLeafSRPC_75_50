import numpy as np
from scipy.spatial import cKDTree
import A2_txt2tensor

#
# point_cloud = A2_txt2tensor.loadtxt_my("./cow.txt")
# print(point_cloud.shape)
#
# # 假设point_cloud是一个形状为(N, 3)的numpy数组，表示有N个三维点
# num_points = 1024  # 定义要生成的点的数量
# tree = cKDTree(point_cloud)
# distances, indices = tree.query(np.random.rand(num_points, 3), k=2)
# resampled_point_cloud = point_cloud[indices]
#
# print(resampled_point_cloud.shape)
#
# np.savetxt('./111.txt', resampled_point_cloud, fmt="%0.4f")



import torch

def sliding_least_squares(points, window_size):
    """
    points: 输入点云，形状为 (N, 3)
    window_size: 窗口大小
    返回值： 采样后的点云，形状为 (N - window_size + 1, 3)
    """
    N = points.shape[0]
    A = torch.zeros((window_size, window_size))
    b = torch.zeros(window_size)

    for i in range(window_size):
        A[i] = torch.arange(i, N - window_size + i + 1)
        b[i] = points[i + window_size // 2, 0]

    # x, _ = torch.lstsq(b.unsqueeze(1), A)
    x, _ = torch.lstsq(b, A)
    x = x.squeeze(1)

    sampled_points = points[x.long()]
    return sampled_points

# 示例
points = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
window_size = 3
sampled_points = sliding_least_squares(points, window_size)
print(sampled_points)
