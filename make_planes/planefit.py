#!/usr/bin/evn python

import numpy as np
import scipy.linalg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from pyntcloud import PyntCloud
import numpy as np
import os
import time

path_in = "/home/jackqian/avod/make_planes/"
path_kitti_training = "/home/jackqian/KITTI/training/velodyne/"
path_kitti_testing = "/home/jackqian/KITTI/testing/velodyne/"
path_save = "/media/jackqian/新加卷/Ubuntu/avod/make_planes/"
file1 = "000002.bin"
file2 = "0.bin"

# some 3-dim points
# mean = np.array([0.0, 0.0, 0.0])
# cov = np.array([[1.0, -0.5, 0.8], [-0.5, 1.1, 0.0], [0.8, 0.0, 1.0]])
# data = np.random.multivariate_normal(mean, cov, 50)
# regular grid covering the domain of the data
# X, Y = np.meshgrid(np.arange(-3.0, 3.0, 0.5), np.arange(-3.0, 3.0, 0.5))
# XX = X.flatten()
# YY = Y.flatten()

"""
using Ransac in PyntCloud to find the groud plane.
Note the lidar points have transformed to the camera coordinate.
:return: groud plane parameters (A, B, C, D) for Ax+By+Cz+D=0.
"""

last_time = time.time()
cloud = PyntCloud.from_file(path_in + file2)
data_raw = np.array(cloud.points)

is_floor = cloud.add_scalar_field("plane_fit", n_inliers_to_stop=len(cloud.points) / 30, max_dist=0.001, max_iterations=500)
#cloud.plot(use_as_color=is_floor, cmap = "cool")

cloud.points = cloud.points[cloud.points[is_floor] > 0]
data = np.array(cloud.points)

mn = np.min(data, axis=0)
mx = np.max(data, axis=0)
X,Y = np.meshgrid(np.linspace(mn[0], mx[0], 20), np.linspace(mn[1], mx[1], 20))
XX = X.flatten()
YY = Y.flatten()

# normal_final = np.zeros(4)
# for i in range(1):
#
#     three_points = cloud.get_sample("points_random", n=3, as_PyntCloud=False)
#
#     three_points_np = []
#     for i in range(len(three_points)):
#         three_points_np.append(np.array([three_points["x"][i], three_points["y"][i], three_points["z"][i]]))
#     vector_one = three_points_np[1] - three_points_np[0]
#     vector_two = three_points_np[2] - three_points_np[0]
#
#     normal = np.cross(vector_one, vector_two)
#     D = - (normal[0]*three_points_np[0][0] + normal[1]*three_points_np[0][1] + normal[2]*three_points_np[0][2])
#     normal = np.hstack((normal, D))
#     normal_final = normal_final + normal
# #normal_final = normal_final/10
#
# if normal_final[3] < 0:
#     normal_final = -normal_final
# off = normal_final[3]/1.65
# normal_final = normal_final / off
# normal_normalized = normal_final / np.linalg.norm(normal_final)
#
#
# current_time = time.time()
# #print("cost_time: ", current_time - last_time)
#
# #print("normal:", normal_final)
# #print("normal_normalized:", normal_normalized)

order = 1  # 1: linear, 2: quadratic
if order == 1:
    # best-fit linear plane
    A = np.c_[data[:, 0], data[:, 1], np.ones(data.shape[0])]
    C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])  # coefficients

    # evaluate it on grid
    Z = C[0] * X + C[1] * Y + C[2]

    result = np.array([C[0], C[1], 1, C[2]])
    result = - result/result[1]
    print(result)

    # or expressed using matrix/vector product
    # Z = np.dot(np.c_[XX, YY, np.ones(XX.shape)], C).reshape(X.shape)

elif order == 2:
    # best-fit quadratic curve
    A = np.c_[np.ones(data.shape[0]), data[:, :2], np.prod(data[:, :2], axis=1), data[:, :2] ** 2]
    C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])

    # evaluate it on a grid
    Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX * YY, XX ** 2, YY ** 2], C).reshape(X.shape)

# plot points and fitted surface
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='r', s=1)
#ax.scatter(data_raw[:, 0], data_raw[:, 1], data_raw[:, 2], c='g', s=5)
plt.xlabel('X')
plt.ylabel('Y')
ax.set_zlabel('Z')
ax.axis('equal')
ax.axis('tight')
plt.show()