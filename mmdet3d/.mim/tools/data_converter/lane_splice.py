import numpy as np
import os
import string
import json
import math
import cv2
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

# camera_model_param:
# cx: 959.3435349245617  # 相机主点cx
# cy: 548.5132971653981  # 相机主点cy
# fx: 932.1969518276043  # 相机焦距fx
# fy: 932.1969518276043  # 相机焦距fy
# k1: 0.2998493816933048  # 径向畸变系数k1 / 等距模型k1
# k2: -0.0198318282342113  # 径向畸变系数k2 / 等距模型k2
# k3: 0.00108686964197308  # 径向畸变系数k3 / 等距模型k3
# k4: 0.6510364285533242  # 径向畸变系数k4 / 等距模型k4
# p1: 0.000339452904797898  # 切向畸变系数p1 (if model_type == radtan)
# p2: -3.395515874104971e-05  # 切向畸变系数p2 (if model_type == radtan)

def pinhole_distort_points(undist_points):
    """
    k: intrinsic matirx
    D: undistort coefficient k1, k2, k3, p1, p2, k4
    """
    cx = 953.6932983398438
    cy = 551.2189331054688
    fx = 931.8326416015625
    fy = 931.8326416015625

    undist_points = undist_points.T
    x = (undist_points[:, 0] - cx) / fx
    y = (undist_points[:, 1] - cy) / fy
    k1, k2, k3, p1, p2, k4 = 0.30465176701545715, -0.02011118270456791, 0.001044161501340568, 0.0, 0.0, 0.6589123010635376
    r2 = x * x + y * y
    # Radial distorsion
    x_dist = x * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2 + k4 * r2 * r2 * r2 * r2)
    y_dist = y * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2 + k4 * r2 * r2 * r2 * r2)

    # Tangential distorsion
    x_dist = x_dist + (2 * p1 * x * y + p2 * (r2 + 2 * x * x))
    y_dist = y_dist + (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)

    # Back to absolute coordinates.
    x_dist = x_dist * fx + cx
    y_dist = y_dist * fy + cy
    dist_points = np.stack([x_dist, y_dist])
    return dist_points

# def ego2image(ego_points, camera_intrinsic, camera2ego_matrix):
#     ego2camera_matrix = np.linalg.inv(camera2ego_matrix)
#     camera_points = ego2camera_matrix[:3, :3] @ ego_points.T + ego2camera_matrix[:3, 3].reshape(3, 1)
#     image_points_ = camera_intrinsic[:3, :3] @ camera_points
#     image_points = image_points_[:2, :] / image_points_[2]
#     image_points = pinhole_distort_points(image_points)
#     return image_points

def ego2image(ego_points, camera_intrinsic, camera2ego_matrix):
    ego2camera_matrix = np.linalg.inv(camera2ego_matrix)
    camera_points = ego2camera_matrix[:3, :3] @ ego_points.T + ego2camera_matrix[:3, 3].reshape(3, 1)
    image_points_ = camera_intrinsic[:3, :3] @ camera_points
    image_points = image_points_[:2, :] / image_points_[2]
    # image_points = pinhole_distort_points(image_points)
    return image_points

def perspective_fit(vectors, extrinsics, intrinsic, distortion_params, img_name, data_raw):
    key_frame_name = img_name
    img_path = data_raw + '/' + str(key_frame_name) + '/' + 'front' + '/' + str(key_frame_name) + '_' + str(key_frame_name) +'.jpg'
    # image = cv2.imread('/home/slj/Documents/workspace/ThomasVision/data/smart_lane/data_raw/20230621134244513/front/20230621134244513_20230621134244513.jpg')
    image = cv2.imread(img_path)
    color_map = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (100, 255, 0), (100, 0, 255), (255, 100, 0),
        (0, 100, 255), (255, 0, 100), (0, 255, 100),
        (255, 255, 255), (0, 0, 0), (0, 100, 100)
    ]

    img_lines = []
    for i, vector in enumerate(vectors):
        img_line = []
        for line in vector:
            # if line == []:
            #     continue
            line = np.array(line)
            img_points = ego2image(line, intrinsic, extrinsics)
            img_points = img_points.T
            img_line.append(img_points.tolist())
        img_lines.append(img_line)

    lines = []
    for i in range(len(img_lines)):
        lines += img_lines[i]

    lens = len(lines) // 2

    cat_lines = []
    for i in range(lens):
        cat_lane = lines[i] + lines[i + lens]
        cat_lines.append(lines[i] + lines[i + lens])

    fit_lanes = []
    # for i, list_lane in enumerate(list_lanes):
    for list_line in cat_lines:
        np_lane = np.array(list_line)
        arrSortIndex = np.lexsort([np_lane[:, 1]])
        np_lane = np_lane[arrSortIndex, :]
        xs_gt = np_lane[:, 0]
        ys_gt = np_lane[:, 1]

        poly_params_yx = np.polyfit(ys_gt, xs_gt, deg=3)

        y_min, y_max = np.min(ys_gt), np.max(ys_gt)
        y_min = math.floor(y_min)
        y_max = math.ceil(y_max)
        y_sample = np.array(range(y_min, y_max, 5))

        ys_out = np.array(y_sample, dtype=np.float32)

        xs_out = np.polyval(poly_params_yx, ys_out)

        fit_lane = np.zeros((len(xs_out), 2))
        fit_lane[:, 0] = xs_out
        fit_lane[:, 1] = ys_out

        # mask_idex = fit_lane[:, 0] > 0
        # if not any(mask_idex):
        #     continue
        #
        # fit_lane = fit_lane[mask_idex]

        fit_lanes.append(fit_lane)

    for k in range(len(fit_lanes)):
        image_lane = fit_lanes[k - 1]
        for i in range(1, image_lane.shape[0]):
            image = cv2.line(image, (int(image_lane[i - 1][0]), int(image_lane[i - 1][1])),
                            (int(image_lane[i][0]), int(image_lane[i][1])), color_map[k], 1)
    cv2.imwrite('./align_time.jpg', image)
    # cv2.imshow("frame", image)
    # cv2.waitKey(0)

def perspective(vectors, extrinsics, intrinsic, distortion_params, img_name, data_raw):
    key_frame_name = img_name
    img_path = data_raw + '/' + str(key_frame_name) + '/' + 'front' + '/' + str(key_frame_name) + '_' + str(key_frame_name) +'.jpg'
    # image = cv2.imread('/home/slj/Documents/workspace/ThomasVision/data/smart_lane/data_raw/20230621134244513/front/20230621134244513_20230621134244513.jpg')
    image = cv2.imread(img_path)
    color_map = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (100, 255, 0), (100, 0, 255), (255, 100, 0),
        (0, 100, 255), (255, 0, 100), (0, 255, 100),
        (255, 255, 255), (0, 0, 0), (0, 100, 100)
    ]


    for i, vector in enumerate(vectors):
        for line in vector:
            line = np.array(line)

            img_points = ego2image(line, intrinsic, extrinsics)
            img_points = img_points.T
            for k in range(1, img_points.shape[0]):
                image = cv2.line(image, (int(img_points[k - 1][0]), int(img_points[k - 1][1])),
                                 (int(img_points[k][0]), int(img_points[k][1])), color_map[i], 1)
    cv2.imwrite('./align_time.jpg', image)

def pose_align(ego_pose):
    x_position, y_position, key_yaw = ego_pose
    # T_OB = [math.cos(yaw), -math.sin(yaw), x_position, math.sin(yaw), math.cos(yaw), y_position, 0, 0, 1]
    # T_OB = [math.cos(yaw), math.sin(yaw), x_position, -math.sin(yaw), math.cos(yaw), y_position, 0, 0, 1]
    R_OB = [math.cos(yaw), -math.sin(yaw), 0, math.sin(yaw), math.cos(yaw), 0, 0, 0, 1]
    R_OB = np.array(R_OB).reshape(3, 3)
    T_OB = np.array([x_position, y_position, 0])
    # T_OB = np.array([y_position, x_position, 0])

    extrinsics = np.zeros((4, 4))
    extrinsics[:3, :3] = R_OB
    extrinsics[:3, 3] = T_OB
    extrinsics[3, 3] = 1.0

    return extrinsics

def motion_unalign(all_point_lines, all_poses):
    vectors = []

    for i, all_point_info in enumerate(zip(all_point_lines, all_poses)):
        all_point_line, all_pose = all_point_info
        if i == 0:
            key_pose = all_pose
            extrinsics_OA = pose_align(key_pose)
            local_vector = []
            for line in all_point_line:
                line = np.array(line)
                local_line = line@extrinsics_OA[:3, :3] + extrinsics_OA[:3, 3]
                local_vector.append(local_line)
            vectors.append(local_vector)
        else:
            pose = all_pose
            # x_position, y_position, yaw = pose
            extrinsics_OB = pose_align(pose)

            # extrinsics_BA = np.linalg.inv(extrinsics_OA)@extrinsics_OB
            # extrinsics_BA = extrinsics_OA@np.linalg.inv(extrinsics_OB)

            align_lines = []
            for line in all_point_line:
                line = np.array(line)
                # align_line = line@extrinsics_BA
                align_line = line@extrinsics_OB[:3, :3] + extrinsics_OB[:3, 3]
                align_lines.append(align_line)

            vectors.append(align_lines)

    return vectors

def motion_align(all_point_lines, all_poses):
    vectors = []
    # key_pose = all_poses[1]
    # extrinsics_OA = pose_align(key_pose)
    # local_vector = []
    # for line in all_point_lines[1]:
    #     line = np.array(line)
    #     line = line[:, ]
    #     local_line = line @ np.linalg.inv(extrinsics_OA[:3, :3]) + extrinsics_OA[:3, 3]
    #     # local_line = line + extrinsics_OA[:3, 3]
    #     local_vector.append(local_line)
    # # vectors.append(local_vector)
    # vectors.append(all_point_lines[1])

    for i, all_point_info in enumerate(zip(all_point_lines, all_poses)):
        all_point_line, all_pose = all_point_info
        if i == 0:
            # continue
            key_pose = all_pose
            extrinsics_OA = pose_align(key_pose)
            local_vector = []
            for line in all_point_line:
                # if line == []:
                #     continue
                line = np.array(line)
                line = line[:, ]
                # try:
                local_line = line@np.linalg.inv(extrinsics_OA[:3, :3]) + extrinsics_OA[:3, 3]
                # except:
                #     pass
                # local_line = line + extrinsics_OA[:3, 3]
                local_vector.append(local_line)
            # vectors.append(local_vector)
            vectors.append(all_point_line)

        else:
            pose = all_pose
            # x_position, y_position, yaw = pose
            extrinsics_OB = pose_align(pose)

            extrinsics_AB = np.linalg.inv(extrinsics_OA)@extrinsics_OB
            extrinsics_BA = np.linalg.inv(extrinsics_AB)
            # extrinsics_BA = extrinsics_AB

            align_lines = []
            for line in all_point_line:
                # if line == []:
                #     continue
                line = np.array(line)
                align_line = line@extrinsics_BA[:3, :3] - extrinsics_BA[:3, 3]
                # align_line = line@np.linalg.inv(extrinsics_OB[:3, :3]) + extrinsics_OB[:3, 3]
                # align_line = line + extrinsics_OB[:3, 3]
                align_lines.append(align_line.tolist())

            vectors.append(align_lines)

    # ego_vectors = []
    # for vector in vectors:
    #     ego_vector = []
    #     for line in vector:
    #         ego_line = line @ extrinsics_OA[:3, :3] - extrinsics_OA[:3, 3]
    #         ego_vector.append(ego_line)
    #
    #     ego_vectors.append(ego_vector)

    return vectors
    # return ego_vectors

def motion_align_relativity(all_point_lines, all_poses):
    vectors = []

    for i, all_point_info in enumerate(zip(all_point_lines, all_poses)):
        all_point_line, all_pose = all_point_info
        if i == 0:
            key_pose = all_pose
            key_x_position, key_y_position, key_yaw = key_pose
            vectors.append(all_point_line)
        else:
            pose = all_pose
            x_position, y_position, yaw = pose
            R = np.array(
                [math.cos(yaw - key_yaw), -math.sin(yaw - key_yaw), 0, math.sin(yaw - key_yaw), math.cos(yaw - key_yaw), 0, 0, 0, 1]).reshape(3, 3)
                # [math.cos(yaw - key_yaw), math.sin(yaw - key_yaw), 0, -math.sin(yaw - key_yaw), math.cos(yaw - key_yaw), 0, 0, 0, 1]).reshape(3, 3)
            T = np.array([y_position - key_y_position, x_position - key_x_position, 0])
            # T = np.array([ x_position - key_x_position,y_position - key_y_position, 0])

            align_lines = []
            for line in all_point_line:
                line = np.array(line)
                # align_line = line@np.linalg.inv(R) - T
                # align_line = line - T
                # align_lines.append(align_line)
                align_lines.append(line)

            vectors.append(align_lines)

    return vectors

def lane_fit(gt_lanes, poly_order=3, sample_step=10, interp=False):

    lanes = []
    for i in range(len(gt_lanes)):
        lanes += gt_lanes[i]

    lens = len(lanes) // 2

    cat_lanes = []
    for i in range(lens):
        cat_lane = lanes[i] + lanes[i + lens]
        # cat_lane.sort()
        cat_lanes.append(lanes[i] + lanes[i + lens])

    fit_lanes = []
    # for i, list_lane in enumerate(list_lanes):
    for list_lane in cat_lanes:
        np_lane = np.array(list_lane)
        arrSortIndex = np.lexsort([np_lane[:, 0]])
        np_lane = np_lane[arrSortIndex, :]
        xs_gt = np_lane[:, 0]
        ys_gt = np_lane[:, 1]
        zs_gt = np_lane[:, 2]

        poly_params_xy = np.polyfit(xs_gt, ys_gt, deg=poly_order)
        poly_params_xz = np.polyfit(xs_gt, zs_gt, deg=poly_order)

        x_min, x_max = np.min(xs_gt), np.max(xs_gt)
        x_min = math.floor(x_min)
        x_max = math.ceil(x_max)
        x_sample = np.array(range(x_min, x_max, sample_step))

        xs_out = np.array(x_sample, dtype=np.float32)

        ys_out = np.polyval(poly_params_xy, xs_out)
        zs_out = np.polyval(poly_params_xz, xs_out)

        fit_lane = np.zeros((len(ys_out), 3))
        fit_lane[:, 0] = xs_out
        fit_lane[:, 1] = ys_out
        fit_lane[:, 2] = zs_out

        mask_idex = fit_lane[:, 0] > 0
        if not any(mask_idex):
            continue

        fit_lane = fit_lane[mask_idex]

        fit_lanes.append(fit_lane)

    return fit_lanes


def get_lane_imu_img_2D(points, iego_lanes=None):
    filepath = "./ego_align.png"
    # print("filepath: " + filepath)
    fig_2d = plt.figure(figsize=(6.4, 6.4))
    plt.grid(linestyle='--', color='y', linewidth=0.5)
    x_major_locator = MultipleLocator(1)
    y_major_locator = MultipleLocator(4)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)

    for i, vector in enumerate(points):
        if i == 0:
            color = 'b'
        if i == 1:
            color = 'r'
        if i == 2:
            color = 'g'

        for line in vector:
            x_data = []
            y_data = []
            for poi in line:
                x_data.append(poi[1])
                y_data.append(poi[0])
            plt.plot(x_data, y_data, linestyle='-', color=color, linewidth=1)

    plt.xlabel('X: ')
    plt.ylabel('Y: distance')
    plt.title("Only show X_Y : align pic")
    plt.savefig(filepath)
    plt.cla()
    plt.close()
    return filepath

def get_lane_imu_img_2D_fit(points, iego_lanes=None):
    filepath = "./ego_align_cat.png"
    # print("filepath: " + filepath)
    fig_2d = plt.figure(figsize=(6.4, 6.4))
    plt.grid(linestyle='--', color='y', linewidth=0.5)
    x_major_locator = MultipleLocator(1)
    y_major_locator = MultipleLocator(4)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)


    for i, vector in enumerate(points):
        line = vector.tolist()
        # if i == 0:
        #     color = 'b'
        # if i == 1:
        #     color = 'r'
        # if i == 2:
        #     color = 'g'
        x_data = []
        y_data = []
        # for line in vector:
        for poi in line:
            x_data.append(poi[1])
            y_data.append(poi[0])
        plt.plot(x_data, y_data, linestyle='-', color='b', linewidth=1)

    plt.xlabel('X: ')
    plt.ylabel('Y: distance')
    plt.title("Only show X_Y : align pic")
    plt.savefig(filepath)
    plt.cla()
    plt.close()
    return filepath



if __name__ == '__main__':
    Odometry_path = '/home/slj/Documents/workspace/ThomasVision/data/smart_lane/Odometry.txt'
    calib_path = '/home/slj/Documents/workspace/ThomasVision/data/smart_lane/front_new.json'
    data_annotation = '/home/slj/Documents/workspace/ThomasVision/data/smart_lane/data_annotation'
    data_raw = '/home/slj/Documents/workspace/ThomasVision/data/smart_lane/data_raw'
    lidar_calib_path = '/home/slj/Documents/workspace/ThomasVision/data/smart_lane/lidar_calib.json'

    with open(calib_path) as ft:
        calibs = json.load(ft)
        extrinsic_rotation_matrix = calibs['extrinsic_rotation_matrix']
        extrinsic_translation_matrix = calibs['extrinsic_translation_matrix']
        intrinsic_matrix = calibs['intrinsic_matrix']
        distortion_params = calibs['distortion_params']

    with open(lidar_calib_path) as lidar_ft:
        lidar_calibs = json.load(lidar_ft)
        lidar_extrinsic_rotation_matrix = lidar_calibs['extrinsic_rotation_matrix']
        lidar_extrinsic_translation_matrix = lidar_calibs['extrinsic_translation_matrix']


    extrinsics = np.zeros((4, 4))
    extrinsics[:3, :3] = extrinsic_rotation_matrix
    extrinsics[:3, 3] = extrinsic_translation_matrix
    extrinsics[3, 3] = 1.0

    intrinsic = np.zeros((3, 4))
    intrinsic[:3, :3] = intrinsic_matrix

    calib = np.matmul(intrinsic, extrinsics)

    data_files = os.listdir(data_annotation)
    data_files.sort()

    data_lines = {}
    with open(Odometry_path, 'r') as infile:
        for line in infile:
            data = line.strip().split(',')
            data_lines[data[0]] = data[1:]

    all_point_lines = []
    all_poses = []
    img_names = []
    i = 30
    for data_file in data_files[i:i+2]:
        img_names.append(data_file.split('.')[0])
        point_lines = []
        ann_path = os.path.join(data_annotation, data_file)
        with open(ann_path) as ft:
            data = json.load(ft)
            point_3ds = data['annotations']
            for point_3d in point_3ds:
                point_line = []
                points = point_3d['points']
                for point in points:
                    x, y, z = point['x'], point['y'], point['z']
                    if x < 0 :
                        continue
                    point_line.append([x, y, z])
                point_lines.append(point_line)
        all_point_lines.append(point_lines)
        data_name = data_file.split('.')[0][:15]
        data = data_lines[data_name]
        x_position, y_position, yaw = data
        x_position = float(x_position)
        y_position = float(y_position)
        yaw = float(yaw.split(';')[0])

        all_poses.append([x_position, y_position, yaw])

    # vectors = motion_align_relativity(all_point_lines, all_poses)
    vectors = motion_align(all_point_lines, all_poses)
    filepath = get_lane_imu_img_2D(vectors)

    # vectors = lane_fit(vectors)
    # filepath_1 = get_lane_imu_img_2D_fit(vectors)

    # perspective(vectors, extrinsics, intrinsic, distortion_params, img_names[0], data_raw)
    perspective_fit(vectors, extrinsics, intrinsic, distortion_params, img_names[0], data_raw)





# [0.003400596494017081 -0.007649317362956704 0.9999649701868492 0.0
# -0.9998360386253572 0.017757926699115575 0.0035359989444003737 0.0
# -0.017784352882268525 -0.9998130291738248 -0.0075876761702830375 0.0
# 0.02605885443814961 1.0919294769483647 -1.8699881027882832 1.0 ]




