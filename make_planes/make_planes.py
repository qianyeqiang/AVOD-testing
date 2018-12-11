from pyntcloud import PyntCloud
import numpy as np
import os
import time
import scipy.linalg
import matplotlib.pyplot as plt


path_in = "/home/jackqian/avod/make_planes/"
path_kitti_training = "/home/jackqian/KITTI/training/velodyne/"
path_kitti_testing = "/home/jackqian/KITTI/testing/velodyne/"
path_save = "/media/jackqian/新加卷/Ubuntu/avod/make_planes/"
file1 = "000303.bin"
file2 = "0.bin"

def lidar4to3():
    """
    convert the lidar points for Nx4 shape to Nx3 shape, i.e., remove the reflectivity.
    :return:
    """
    filename = path_in + file1
    print("Processing: ", filename)
    scan = np.fromfile(filename, dtype=np.float32)
    print(np.shape(scan))
    scan = scan.reshape((-1, 4))
    scan = scan[:, :3]
    scan = scan.reshape(-1)

    #calib = calib_at("000000")

    # scan input: nx3; scan output: nx3;
    #scan = lidar_point_to_img(scan, calib[3], calib[2], calib[0])
    #scan = scan.astype(np.float32)

    #np.save(str(0)+ ".txt", scan)

    scan.tofile(file2)

def lidar4to3_kitti():
    """
    convert the KITTI lidar points for Nx4 shape to Nx3 shape, i.e., remove the reflectivity.
    :return:
    """
    for i in range(7481):
        filename = path_kitti_training + str(i).zfill(6) + ".bin"
        print("Processing: ", filename)
        scan = np.fromfile(filename, dtype=np.float32)
        #print(np.shape(scan))
        scan = scan.reshape((-1, 4))
        scan = scan[:, :3]

        calib = calib_at(str(i).zfill(6))

        # scan input: nx3; scan output: nx3;
        scan = lidar_point_to_img_calib2(scan, calib[3], calib[2], calib[0])
        scan = scan.astype(np.float32)

        file2 = path_save + "kittilidar_training_qyqmake_calib2/" + str(i).zfill(6) + ".bin"

        scan.tofile(file2)


def cau_planes():
    """
    using Ransac in PyntCloud to find the groud plane.
    Note the lidar points have transformed to the camera coordinate.
    :return: groud plane parameters (A, B, C, D) for Ax+By+Cz+D=0.
    """

    last_time = time.time()
    cloud = PyntCloud.from_file(path_save + "kittilidar_training_qyqmake_calib2/" + file1)
    #cloud = PyntCloud.from_file(path_in + file2)

    #cloud.plot()
    cloud.points = cloud.points[cloud.points["y"] > 1]
    # cloud.points = cloud.points[cloud.points["x"] > -2]
    # cloud.points = cloud.points[cloud.points["x"] < 2]
    # cloud.points = cloud.points[cloud.points["z"] > -20]
    # cloud.points = cloud.points[cloud.points["z"] < 20]
    data_raw = np.array(cloud.points)

    is_floor = cloud.add_scalar_field("plane_fit", n_inliers_to_stop=len(cloud.points) / 30, max_dist=0.001,  max_iterations=100)
    #cloud.plot(use_as_color=is_floor, cmap = "cool")

    cloud.points = cloud.points[cloud.points[is_floor] > 0]

    data = np.array(cloud.points)

    mn = np.min(data, axis=0)
    mx = np.max(data, axis=0)
    X, Y = np.meshgrid(np.linspace(mn[0], mx[0], 20), np.linspace(mn[1], mx[1], 20))
    X_avod, Y_avod = np.meshgrid(np.linspace(mn[0], mx[0], 20), np.linspace(mn[1], mx[1], 20))
    X_flat, Z_flat = np.meshgrid(np.arange(-5, 5, 1), np.arange(-5, 5, 1))

    #### best-fit linear plane
    #### Z = C[0] * X + C[1] * Y + C[2]
    A = np.c_[data[:, 0], data[:, 1], np.ones(data.shape[0])]
    C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])  # coefficients
    Z = C[0] * X + C[1] * Y + C[2]

    Z_avod = (8.587492e-03*X_avod + 9.995657e-01*Y_avod - 1.519515e+00)/2.818885e-02
    Z_avod = (1.316190e-02 * X_avod + 9.997416e-01 * Y_avod - 1.543552e+00) / 1.853603e-02

    Y_flat = 1.65

    normal = np.array([C[0], C[1], 1, C[2]])
    normal = - normal / normal[1]
    print(normal)

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    #
    # #ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
    # ax.plot_surface(X_avod, Y_avod, Z_avod, rstride=1, cstride=1, alpha=0.2)
    # ax.plot_surface(X_flat, Y_flat, Z_flat, rstride=1, cstride=1, alpha=0.2)
    #
    # ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='r', s=1)
    # #ax.scatter(data_raw[:, 0], data_raw[:, 1], data_raw[:, 2], c='g', s=0.1)
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.ylabel('Z')
    # #ax.set_zlabel('Z')
    # ax.axis('equal')
    # ax.axis('tight')
    #
    # ax.axis([-5, 5, -5, 5])
    # ax.set_zlim(-5, 5)
    # #ax.zaxis.set_major_locator(LinearLocator(20))
    #
    # plt.show()

    current_time = time.time()
    print("cost_time: ", current_time - last_time)

    #print("normal:", normal_final)
    #print("normal_normalized:", normal_normalized)

def cau_planes_kitti():
    """
    using Ransac in PyntCloud to find the groud plane in KITTI.
    Note the lidar points have transformed to the camera coordinate.
    :return: groud plane parameters (A, B, C, D) for Ax+By+Cz+D=0.
    """

    # regular grid covering the domain of the data

    last_time = time.time()
    k = 0

    while k != 7481:

        print(path_save + "kittilidar_training_qyqmake_calib2/" + str(k).zfill(6)+ ".bin")
        cloud = PyntCloud.from_file(path_save + "kittilidar_training_qyqmake_calib2/" + str(k).zfill(6)+ ".bin")

        cloud.points = cloud.points[cloud.points["y"] > 1]

        is_floor = cloud.add_scalar_field("plane_fit", n_inliers_to_stop=len(cloud.points) / 30, max_dist=0.001, max_iterations=100)

        cloud.points = cloud.points[cloud.points[is_floor] > 0]
        data = np.array(cloud.points)

        #### best-fit linear plane
        #### Z = C[0] * X + C[1] * Y + C[2]
        A = np.c_[data[:, 0], data[:, 1], np.ones(data.shape[0])]
        C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])  # coefficients

        normal = np.array([C[0], C[1], 1, C[2]])
        normal = - normal / normal[1]
        print(normal)

        # Check if the result is almost the groud plane.
        # if the result is right, parameter B should be nearly 1 when the D is the height of the camera.
        if (normal[3] > 2.0 or normal[3] < 1.4) :
            print("error_result")
            continue

        # elif (normal[3] > 2.0 or normal[3] < 1.4) and count >= 10:
        #     txtname = path_save + "kittilidar_training_planes_qyqmake_calib2_fit/" + str(k).zfill(6) + ".txt"
        #     f = open(txtname, "a")
        #     f.write("# Plane\n")
        #     f.write("Width 4\n")
        #     f.write("Height 1\n")
        #     str_normal = "0" + " " + "-1" + " " + "0" + " " + "1.65"
        #     f.write(str_normal)
        #     f.close()
        #
        #     k = k + 1
        #     count = 0
        #     continue

        txtname = path_save + "kittilidar_training_planes_qyqmake_calib2_fit/" + str(k).zfill(6) + ".txt"
        f = open(txtname, "a")
        f.write("# Plane\n")
        f.write("Width 4\n")
        f.write("Height 1\n")
        str_normal = str(normal[0]) + " " + str(normal[1]) + " " + str(normal[2]) + " " + str(normal[3])
        f.write(str_normal)
        f.close()

        k = k + 1

        # normal_normalized = normal / np.linalg.norm(normal)
        # print("normal_normalized:", normal_normalized)

        # current_time = time.time()

        # print("cost_time: ", current_time - last_time)
        # print("normal:", normal)

def lidar_point_to_img_calib2(point, Tr, R0, P2):
    """
    rewrite by jackqian
    convert lidar points to the camera
    input: points with shape Nx3; output: point with shape NX3 (N is the number of the points)
    output = R0*Tr*point
    if you want to convert the lidar points to the image: output = P2*R0*Tr*point
    """
    P2 = P2.reshape((3, 4))
    R0 = R0.reshape((4, 3))
    Tr = Tr.reshape((3, 4))

    T = np.zeros((1,4))
    T[0,3] = 1

    P2 = np.vstack((P2, T))
    Tr = np.vstack((Tr, T))

    T2 = np.zeros((4,1))
    T2[3,0] = 1
    R0 = np.hstack((R0, T2))

    assert Tr.shape == (4, 4)
    assert R0.shape == (4, 4)
    assert P2.shape == (4, 4)

    point = point.transpose((1, 0))

    point = np.vstack((point, np.ones(point.shape[1])))

    # mat1 =  np.dot(P2, R0)
    # mat2 = np.dot(mat1, Tr)
    # img_cor = np.dot(mat2, point)

    #mat = np.dot(R0, Tr)
    img_cor = np.dot(Tr, point)

    #img_cor = img_cor/img_cor[2]

    img_cor = img_cor.transpose((1, 0))

    img_cor = img_cor[:, :3]

    return img_cor


def lidar_point_to_img(point, Tr, R0, P2):
    """
    rewrite by jackqian
    convert lidar points to the camera
    input: points with shape Nx3; output: point with shape NX3 (N is the number of the points)
    output = R0*Tr*point
    if you want to convert the lidar points to the image: output = P2*R0*Tr*point
    """
    P2 = P2.reshape((3, 4))
    R0 = R0.reshape((4, 3))
    Tr = Tr.reshape((3, 4))

    T = np.zeros((1,4))
    T[0,3] = 1

    P2 = np.vstack((P2, T))
    Tr = np.vstack((Tr, T))

    T2 = np.zeros((4,1))
    T2[3,0] = 1
    R0 = np.hstack((R0, T2))

    assert Tr.shape == (4, 4)
    assert R0.shape == (4, 4)
    assert P2.shape == (4, 4)

    point = point.transpose((1, 0))

    point = np.vstack((point, np.ones(point.shape[1])))

    # mat1 =  np.dot(P2, R0)
    # mat2 = np.dot(mat1, Tr)
    # img_cor = np.dot(mat2, point)

    mat = np.dot(R0, Tr)
    img_cor = np.dot(mat, point)

    #img_cor = img_cor/img_cor[2]

    img_cor = img_cor.transpose((1, 0))

    img_cor = img_cor[:, :3]

    return img_cor

def calib_at(index):
    """
    Return the calib sequence.
    """
    calib_ori = load_kitti_calib(index)
    calib = np.zeros((4, 12))
    calib[0,:] = calib_ori['P2'].reshape(12)
    calib[1,:] = calib_ori['P3'].reshape(12)
    calib[2,:9] = calib_ori['R0'].reshape(9)
    calib[3,:] = calib_ori['Tr_velo2cam'].reshape(12)

    return calib

def load_kitti_calib(index):
    """
    load projection matrix

    """
    data_path = '/home/jackqian//KITTI/'
    prefix = 'training/calib'
    #prefix = 'testing/calib'
    calib_dir = os.path.join(data_path, prefix, index + '.txt')

    with open(calib_dir) as fi:
        lines = fi.readlines()

    obj = lines[2].strip().split(' ')[1:]
    P2 = np.array(obj, dtype=np.float32)
    obj = lines[3].strip().split(' ')[1:]
    P3 = np.array(obj, dtype=np.float32)
    obj = lines[4].strip().split(' ')[1:]
    R0 = np.array(obj, dtype=np.float32)
    obj = lines[5].strip().split(' ')[1:]
    Tr_velo_to_cam = np.array(obj, dtype=np.float32)

    return {'P2': P2.reshape(3, 4),
            'P3': P3.reshape(3, 4),
            'R0': R0.reshape(3, 3),
            'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)}

def main():
    #lidar4to3()
    #lidar4to3_kitti()
    cau_planes()
    pass

if __name__ == '__main__':
    main()