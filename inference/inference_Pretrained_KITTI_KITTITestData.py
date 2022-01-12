import torch
import numpy as np
from model.CalibDNN import *
from utils.AverageMeter import *
import imageio as smc
from natsort import natsorted as ns
import glob, os
import cv2
import torchgeometry as tgm
import torchvision.transforms as transforms
from PIL import Image
from utils.mathutils_func import *

dataset_path = '/data/2011_09_29/'


IMG_HT = 374
IMG_WDT = 1238

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

print("------------ GPU Setting Start ----------------")
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
pretrained_path = cf.paths['pretrained_path']
gpu_check = is_gpu_avaliable()
devices = torch.device("cuda") if gpu_check else torch.device("cpu")
if gpu_check:
    print("I have GPU!")
else:
    print("I don't have GPU!")
print("------------ GPU Setting Finish ----------------")
print("")
print("")
print("------------ Model Setting Start ----------------")
models = []
weights = cf.inference_info['weights']
for idx in range(len(weights)):
    model = CalibDNN18(18,  pretrained=weights[idx]).to(devices)
    if os.path.isfile(weights[idx]):
        print("Pretrained Model Open : ", weights[idx])
        checkpoint = load_weight_file(weights[idx])
        load_weight_parameter(model, checkpoint['state_dict'])
    else:
        print("Pretrained Parameter Open : No Pretrained Model")
    model.eval()
    models.append(model)
    print("Iterative Refinement ", idx, " Model Load")
print("")
K_final = torch.tensor(K, dtype=torch.float32).to(devices)

print("")
print("------------ Inference Start ----------------")

model.eval()
transform = transforms.Compose([transforms.ToTensor()])
AICamera_image = ns(glob.glob(dataset_path + "2011_09_29_drive_0026_sync/image_02/data/*.png"))
AICamera_Point = ns(glob.glob(dataset_path + "2011_09_29_drive_0026_sync/depth_maps_transformed/*.png"))
transforms_list = np.loadtxt(dataset_path + "2011_09_29_drive_0026_sync/pointcloud_list.txt", dtype=str)
angle_list = np.loadtxt(dataset_path + "2011_09_29_drive_0026_sync/angle_list.txt", dtype=str)

rotation_X = np.array([0.0], dtype=np.float32)
rotation_Y = np.array([0.0], dtype=np.float32)
rotation_Z = np.array([0.0], dtype=np.float32)
translation_X = np.array([0.0], dtype=np.float32)
translation_Y = np.array([0.0], dtype=np.float32)
translation_Z = np.array([0.0], dtype=np.float32)

count = 0
image_count = 0
for image_file, depth_map, pointfile, angle in zip(AICamera_image, AICamera_Point, transforms_list,angle_list):
    count += 1
    # Image, LIDAR Data Read
    filename, _ = image_file.split('.')
    file = filename.split('/')
    input_image = smc.imread(image_file)
    points = np.loadtxt(pointfile)
    points = points[:150000, :3]
    ones_col = np.ones(shape=(points.shape[0], 1))
    points = np.hstack((points, ones_col))

    # Ground Truth Projection Image Get
    GT_RTMatrix = np.matmul(cam_02_transform, np.matmul(R_rect_00, velo_to_cam))
    points_2d_Gt = np.matmul(K, np.matmul(GT_RTMatrix, points.T)[:-1, :])
    Z = points_2d_Gt[2, :]
    x = (points_2d_Gt[0, :] / (Z + 1e-10)).T
    y = (points_2d_Gt[1, :] / (Z + 1e-10)).T
    mask = (x > 0) & (x < cf.KITTI_Info["WIDTH"]) & (y > 0) & (y < cf.KITTI_Info["HEIGHT"]) & (Z > 0)
    x = x[mask].astype(np.int32)
    y = y[mask].astype(np.int32)
    Z = Z[mask]
    reprojected_img = input_image.copy()
    for x_idx, y_idx,z_idx in zip(x, y, Z):
        if(z_idx>0):
            cv2.circle(reprojected_img, (int(x_idx), int(y_idx)), 1, (255, 0, 0), -1)
    smc.imsave(cf.paths['inference_img_result_path'] + "/Pretrained_KITTI_2011_09_29/KITTITestData_" + str(image_count) +"_Target"  + ".png", reprojected_img)

    omega_x = np.random.uniform(-20, 20) * (3.141592 / 180.0)
    omega_y = np.random.uniform(-20, 20) * (3.141592 / 180.0)
    omega_z = np.random.uniform(-20, 20) * (3.141592 / 180.0)
    tr_x = np.random.uniform(-2, 2)
    tr_y = np.random.uniform(-2, 2)
    tr_z = np.random.uniform(-2, 2)

    r_org = mathutils.Euler((omega_x, omega_y, omega_z))
    t_org = mathutils.Vector((tr_x, tr_y, tr_z))

    R = r_org.to_matrix()
    R.resize_4x4()
    T = mathutils.Matrix.Translation(t_org)
    RT = T @ R

    random_transform = np.array(RT)
    points_in_cam_axis = np.matmul(cf.KITTI_Info["R_rect_00"], (np.matmul(cf.KITTI_Info["velo_to_cam"], points.T)))
    transformed_points = np.matmul(random_transform, points_in_cam_axis)
    points_2d = np.matmul(cf.KITTI_Info["K"],
                          np.matmul(cf.KITTI_Info["cam_02_transform"], transformed_points)[:-1, :])

    Z = points_2d[2, :]
    x = (points_2d[0, :] / (Z + 1e-10)).T
    y = (points_2d[1, :] / (Z + 1e-10)).T
    mask = (x > 0) & (x < cf.KITTI_Info["WIDTH"]) & (y > 0) & (y < cf.KITTI_Info["HEIGHT"]) & (Z > 0)
    x = x[mask].astype(np.int)
    y = y[mask].astype(np.int)
    Z = Z[mask]
    source_map = np.zeros((cf.KITTI_Info["HEIGHT"], cf.KITTI_Info["WIDTH"]))
    source_map[y, x] = Z
    # smc.imsave("depth_map_predicted.png", source_map.astype(np.uint8))
    source_map = np.repeat(np.expand_dims(source_map, axis=2), 3, axis=2)
    source_map = (source_map - 40.0) / 40.0
    source_map = np.float32(source_map)
    source_map = transform(source_map).to(devices)
    source_map = torch.unsqueeze(source_map, 0)

    # Predict
    source_image = input_image.copy()
    source_image = (source_image - 127.5) / 127.5
    source_image = transform(source_image).to(devices)
    source_image = torch.unsqueeze(source_image, 0)

    random_transform_inv = np.linalg.inv(random_transform)
    transform_matrix = torch.tensor(random_transform_inv, dtype=torch.float32).to(devices)
    point_cloud = torch.tensor(points, dtype=torch.float32).to(devices)
    GT_RTMatrix_torch = torch.tensor(GT_RTMatrix, dtype = torch.float32).to(devices)
    if gpu_check:
        source_depth_map = source_map.to(torch.float32)
        source_image = source_image.to(torch.float32)

    model_count = 0
    RTs = [transform_matrix.inverse()]
    with torch.no_grad():
        for model in models:
            rtvec = model(source_image, source_depth_map)
            RT_predicted = tgm.rtvec_to_pose(rtvec)
            RTs.append(torch.mm(RTs[model_count], RT_predicted[0]))

            points_in_cam_axis = torch.mm(GT_RTMatrix_torch, point_cloud.T)
            transformed_points = torch.mm(RTs[-1], points_in_cam_axis)
            points_2d = torch.mm(K_final, transformed_points[:-1, :])

            Z = points_2d[2, :]
            x = (points_2d[0, :] / (Z + 1e-10)).T
            y = (points_2d[1, :] / (Z + 1e-10)).T
            mask = (x > 0) & (x < cf.KITTI_Info["WIDTH"]) & (y > 0) & (y < cf.KITTI_Info["HEIGHT"]) & (Z > 0)
            x = x[mask].to(torch.long)
            y = y[mask].to(torch.long)
            Z = Z[mask]
            source_map = torch.zeros((cf.KITTI_Info["HEIGHT"], cf.KITTI_Info["WIDTH"])).to(devices)
            source_map[y, x] = Z

            smc.imsave(
                cf.paths["inference_img_result_path"] + "/Pretrained_KITTI_2011_09_29/predicted_" + str(
                    image_count) + "_" + str(model_count) + ".png",
                source_map.detach().cpu().numpy())

            model_count += 1
            source_map = torch.repeat_interleave(torch.unsqueeze(source_map, dim=0), 3, dim=0)
            source_map = torch.unsqueeze(source_map, dim=0)
            source_map = (source_map - 40.0) / 40.0
            source_depth_map = source_map

    predicted_Matrix = torch.mm(transform_matrix, RTs[-1]).detach().cpu().numpy()
    GT_RTMatrix = np.linalg.inv(random_transform)
    rot_gt, tra_gt = convert_RTMatrix_to_6DoF(GT_RTMatrix)
    rot_pre, tra_pre = convert_RTMatrix_to_6DoF(predicted_Matrix)
    rot_pre_degree = np.rad2deg(rot_pre)
    rot_gt_degree = np.rad2deg(rot_gt)
    rotation_X += np.abs(rot_gt_degree[0] - rot_pre_degree[0])
    rotation_Y += np.abs(rot_gt_degree[1] - rot_pre_degree[1])
    rotation_Z += np.abs(rot_gt_degree[2] - rot_pre_degree[2])
    translation_X += np.abs(tra_gt[0] - tra_pre[0])
    translation_Y += np.abs(tra_gt[1] - tra_pre[1])
    translation_Z += np.abs(tra_gt[2] - tra_pre[2])

    RT_predicted = RTs[-1].detach().cpu().numpy()
    # Save Predicted Depth Map
    GT_RTMatrix = np.matmul(cam_02_transform, np.matmul(R_rect_00, velo_to_cam))
    transformed_points = np.matmul(RT_predicted, np.matmul(GT_RTMatrix, points.T))
    points_2d_Init = np.matmul(K, transformed_points[:-1, :])
    Z = points_2d_Init[2, :]
    x = (points_2d_Init[0, :] / (Z + 1e-10)).T
    y = (points_2d_Init[1, :] / (Z + 1e-10)).T
    mask = (x > 0) & (x < cf.KITTI_Info["WIDTH"]) & (y > 0) & (y < cf.KITTI_Info["HEIGHT"]) & (Z > 0)
    x = x[mask].astype(np.int32)
    y = y[mask].astype(np.int32)
    Z = Z[mask].astype(np.int32)
    projected_img = input_image.copy()
    for x_idx, y_idx, z_idx in zip(x, y, Z):
        if (z_idx > 0):
            cv2.circle(projected_img, (int(x_idx), int(y_idx)), 1, (255, 0, 0), -1)
    smc.imsave(cf.paths['inference_img_result_path'] + "/Pretrained_KITTI_2011_09_29/KITTITestData_" + str(image_count) + "_Predicted.png", projected_img)
    image_count += 1
    if count % cf.network_info['freq_print'] == 0:
        print("[ROT X] : ", rotation_X / count, " [ROT Y] : ", rotation_Y / count, " [ROT Z] : ", rotation_Z / count,
              " [TRANSLATION X] : ", translation_X / count, " [TRANSLATION Y] : ", translation_Y / count,
              "[TRANSLATION Z] :",
              translation_Z / count)

print("[ROT X] : ", rotation_X/count, " [ROT Y] : ", rotation_Y/count, " [ROT Z] : ", rotation_Z/count,
      " [TRANSLATION X] : ", translation_X/count, " [TRANSLATION Y] : ", translation_Y/count, "[TRANSLATION Z] :",
      translation_Z/count)
