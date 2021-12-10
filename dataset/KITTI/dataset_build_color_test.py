"""

For sequence: 2011_09_26

"""
import mathutils
import numpy as np
import os
import glob
import argparse
from natsort import natsorted as ns
from skimage import io

import imageio as smc

IMG_HT = 375
IMG_WDT = 1242

fx = 7.183351e+02
fy = 7.183351e+02
cx = 6.003891e+02
cy = 1.815122e+02

K = np.array([7.183351e+02, 0.000000e+00, 6.003891e+02,
              0.000000e+00, 7.183351e+02, 1.815122e+02,
              0.000000e+00, 0.000000e+00, 1.000000e+00]).reshape(3,3)

velo_to_cam_R = np.array([7.755449e-03, -9.999694e-01, -1.014303e-03,
                          2.294056e-03, 1.032122e-03, -9.999968e-01,
                          9.999673e-01, 7.753097e-03, 2.301990e-03]).reshape(3,3)
velo_to_cam_T = np.array([-7.275538e-03, -6.324057e-02, -2.670414e-01]).reshape(3,1)
velo_to_cam = np.vstack((np.hstack((velo_to_cam_R, velo_to_cam_T)), np.array([[0,0,0,1]])))

R_rect_00 =  np.array([9.999478e-01, 9.791707e-03, -2.925305e-03, 0.0,
                       -9.806939e-03, 9.999382e-01, -5.238719e-03, 0.0,
                       2.873828e-03, 5.267134e-03, 9.999820e-01, 0.0,
                       0.0, 0.0, 0.0, 1.0]).reshape(4, 4)

cam_02_transform = np.array([1.0, 0.0, 0.0, 4.450382e+01/fx,
                             0.0, 1.0, 0.0, -5.951107e-01/fy,
                             0.0, 0.0, 1.0, 2.616315e-03,
                             0.0, 0.0, 0.0, 1.0]).reshape(4,4)


main_path = "/data/2011_09_29/2011_09_29_drive_0026"


def timestamp_sync(path):
    txt1 = np.loadtxt(path + "_extract/velodyne_points/timestamps.txt", dtype = str)
    txt2 = np.loadtxt(path + "_sync/velodyne_points/timestamps.txt", dtype = str)
    file_list = ns(glob.glob(path + "_extract/velodyne_points/data/*.txt"))

    times1 = txt1[:,1]
    times2 = txt2[:,1]

    for idx in range(times1.shape[0]):
        times1[idx] = times1[idx].split(":")[2]

    for idx in range(times2.shape[0]):
        times2[idx] = times2[idx].split(":")[2]

    start_pt = times2[0]
    end_pt = times2[-1]

    index_start = np.where(times1 == start_pt)[0][0]
    index_end = np.where(times1 == end_pt)[0][0]

    return file_list[index_start:index_end+1]


if not os.path.exists(main_path + "_sync/depth_maps"):
    os.makedirs(main_path + "_sync/depth_maps")

if not os.path.exists(main_path + "_sync/target_imgs"):
    os.makedirs(main_path + "_sync/target_imgs")

if not os.path.exists(main_path + "_sync/depth_maps_transformed"):
    os.makedirs(main_path + "_sync/depth_maps_transformed")

depth_maps_folder = main_path + "_sync/depth_maps"
target_img_folder = main_path + "_sync/target_imgs"
depth_maps_transformed_folder = main_path + "_sync/depth_maps_transformed"

point_files = timestamp_sync(main_path)
imgs_files = ns(glob.glob(main_path + "_sync/image_02/data/*.png"))

angle_limit = 0.34722965035593395/1.25
tr_limit = 0.34722965035593395/1.25

angle_list = np.zeros((1,16), dtype = np.float32)
pointcloud_file = open(depth_maps_transformed_folder + "/../pointcloud_list.txt", 'w')

for img_name, cloud_name in zip(imgs_files, point_files):

    print(img_name, cloud_name)

    omega_x = angle_limit*np.random.random_sample() - (angle_limit/2.0)
    omega_y = angle_limit*np.random.random_sample() - (angle_limit/2.0)
    omega_z = angle_limit*np.random.random_sample() - (angle_limit/2.0)
    tr_x = tr_limit*np.random.random_sample() - (tr_limit/2.0)
    tr_y = tr_limit*np.random.random_sample() - (tr_limit/2.0)
    tr_z = tr_limit*np.random.random_sample() - (tr_limit/2.0)

    r_org = mathutils.Euler((omega_x, omega_y, omega_z))
    t_org = mathutils.Vector((tr_x, tr_y, tr_z))

    R = r_org.to_matrix()
    R.resize_4x4()
    T = mathutils.Matrix.Translation(t_org)
    RT = T @ R

    random_transform = np.array(RT)
    to_write_tr = np.expand_dims(np.ndarray.flatten(random_transform), 0)
    angle_list = np.vstack((angle_list, to_write_tr))
    print("data_set_build_color.py : \n", random_transform)
    pointcloud_file.write(cloud_name + "\n")

    points = np.loadtxt(cloud_name)
    points = points[:90000,:3]
    ones_col = np.ones(shape=(points.shape[0],1))
    points = np.hstack((points,ones_col))
    current_img = smc.imread(img_name)
    # current_img_color = smc.imread(color_name)

    img = smc.imread(img_name)
    img_ht = img.shape[0]
    img_wdt = img.shape[1]

    # Velodyne Point Cloud와 Camera_0 센서 간의 맞춤
    points_in_cam_axis = np.matmul(R_rect_00, (np.matmul(velo_to_cam, points.T)))
    transformed_points = np.matmul(random_transform, points_in_cam_axis)
    points_2d = np.matmul(K, np.matmul(cam_02_transform, transformed_points)[:-1,:])

    Z = points_2d[2,:]
    x = (points_2d[0,:]/Z).T
    y = (points_2d[1,:]/Z).T

    x = np.clip(x, 0.0, img_wdt - 1)
    y = np.clip(y, 0.0, img_ht - 1)

    reprojected_img = np.zeros_like(img)
    for x_idx, y_idx,z_idx in zip(x,y,Z):
        if(z_idx>0):
            reprojected_img[int(y_idx), int(x_idx)] = z_idx

    smc.imsave(depth_maps_transformed_folder + "/" + img_name[-14:], reprojected_img)

    GT_RTMatrix = np.matmul(cam_02_transform, np.matmul(R_rect_00, velo_to_cam))

    points_2d = np.matmul(K, np.matmul(GT_RTMatrix, points.T)[:-1, :])

    Z = points_2d[2,:]
    x = (points_2d[0,:]/Z).T
    y = (points_2d[1,:]/Z).T

    x = np.clip(x, 0.0, img_wdt - 1)
    y = np.clip(y, 0.0, img_ht - 1)

    reprojected_img = np.zeros_like(img)
    for x_idx, y_idx,z_idx in zip(x,y,Z):
        if(z_idx > 0):
            reprojected_img[int(y_idx), int(x_idx)] = z_idx
    pooled_img = reprojected_img

    reconstructed_img = current_img*(pooled_img>0.)
    smc.imsave(depth_maps_folder + "/" + img_name[-14:], pooled_img)
    smc.imsave(target_img_folder + "/" + img_name[-14:], reconstructed_img)

np.savetxt(depth_maps_transformed_folder + "/../angle_list.txt", angle_list[1:], fmt = "%.4f")
pointcloud_file.close()