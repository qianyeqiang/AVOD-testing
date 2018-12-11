from pyntcloud import PyntCloud
import numpy as np
import os
import time
from pandas.plotting import scatter_matrix


path_in = "/home/jackqian/avod/make_planes/"
path_kitti_training = "/home/jackqian/KITTI/training/velodyne/"
path_kitti_testing = "/home/jackqian/KITTI/testing/velodyne/"
path_save = "/media/jackqian/新加卷/Ubuntu/avod/make_planes/"
file1 = "000000.bin"
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

    calib = calib_at("000000")

    # scan input: nx3; scan output: nx3;
    scan = lidar_point_to_img(scan, calib[3], calib[2], calib[0])
    scan = scan.astype(np.float32)

    #np.save(str(0)+ ".txt", scan)

    #scan.tofile(file2)

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
    cloud = PyntCloud.from_file(path_in + file2)
    #print(cloud)

    cloud.plot()

    is_floor = cloud.add_scalar_field("plane_fit", n_inliers_to_stop=len(cloud.points) / 30, max_dist=0.001)
    #cloud.plot(use_as_color=is_floor, cmap = "cool")

    cloud.points = cloud.points[cloud.points[is_floor] > 0]

    normal_final = np.zeros(4)
    for i in range(1):

        three_points = cloud.get_sample("points_random", n=3, as_PyntCloud=False)

        three_points_np = []
        for i in range(len(three_points)):
            three_points_np.append(np.array([three_points["x"][i], three_points["y"][i], three_points["z"][i]]))
        vector_one = three_points_np[1] - three_points_np[0]
        vector_two = three_points_np[2] - three_points_np[0]

        normal = np.cross(vector_one, vector_two)
        D = - (normal[0]*three_points_np[0][0] + normal[1]*three_points_np[0][1] + normal[2]*three_points_np[0][2])
        normal = np.hstack((normal, D))
        normal_final = normal_final + normal
    #normal_final = normal_final/10

    if normal_final[3] < 0:
        normal_final = -normal_final
    off = normal_final[3]/1.65
    normal_final = normal_final / off
    normal_normalized = normal_final / np.linalg.norm(normal_final)


    current_time = time.time()
    #print("cost_time: ", current_time - last_time)

    #print("normal:", normal_final)
    #print("normal_normalized:", normal_normalized)

def cau_planes_kitti():
    """
    using Ransac in PyntCloud to find the groud plane in KITTI.
    Note the lidar points have transformed to the camera coordinate.
    :return: groud plane parameters (A, B, C, D) for Ax+By+Cz+D=0.
    """

    last_time = time.time()
    k = 0

    while k != 7481:

        print(path_save + "kittilidar_training_qyqmake_calib2/" + str(k).zfill(6)+ ".bin")
        cloud = PyntCloud.from_file(path_save + "kittilidar_training_qyqmake_calib2/" + str(k).zfill(6)+ ".bin")

        is_floor = cloud.add_scalar_field("plane_fit", n_inliers_to_stop=len(cloud.points) / 30, max_dist=0.001, max_iterations=100)

        cloud.points = cloud.points[cloud.points[is_floor] > 0]

        three_points = cloud.get_sample("points_random", n=3, as_PyntCloud=False)
        three_points_np = []
        for i in range(len(three_points)):
            three_points_np.append(np.array([three_points["x"][i], three_points["y"][i], three_points["z"][i]]))
        vector_one = three_points_np[1] - three_points_np[0]
        vector_two = three_points_np[2] - three_points_np[0]

        normal = np.cross(vector_one, vector_two)
        D = - (normal[0]*three_points_np[0][0] + normal[1]*three_points_np[0][1] + normal[2]*three_points_np[0][2])
        normal = np.hstack((normal, D))

        # if normal[3] < 0:
        #     normal = -normal
        # off = normal[3]/1.65
        # normal = normal / off
        #
        # # Check if the result is almost the groud plane.
        # # if the result is right, parameter B should be nearly 1 when the D is the height of the camera.
        # if normal[1] > -0.5 or normal[1] < -1.5:
        #     print("error_result")
        #     continue

        normal = normal / normal[1]
        normal = - normal

        # Check if the result is almost the groud plane.
        # if the result is right, parameter B should be nearly 1 when the D is the height of the camera.
        if normal[3] > 2.0 or normal[3] < 1.3:
            print("error_result")
            continue

        else:

            txtname = path_save + "kittilidar_training_planes_qyqmake_calib2/" + str(k).zfill(6) + ".txt"
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

        current_time = time.time()

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