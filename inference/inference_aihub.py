import numpy as np
import torch

from model.CalibDNN import *
from model.loss_changwon import *
from utils.AverageMeter import *
import imageio as smc
from natsort import natsorted as ns
import glob, os
import cv2
import open3d as o3d
import torchvision.transforms as transforms


dataset_path = '/data/AIHub/'

K = np.array([1067.249617, 0, 967.123811,
              0, 1070.402406, 600.618148,
              0, 0, 1]).reshape(3, 3)

rpy = np.array([np.deg2rad(90.0), np.deg2rad(90.0), 0.0])
translation_vec = np.array([-0.07, 0.05, -0.14])

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
model = CalibDNN18(18, pretrained=os.path.join(pretrained_path, "CalibDNN_18_ALL" + '.pth')).to(devices)
print("------------ Model Summary ----------------")
summary(model, [(1, 3, 480, 640), (1, 3, 480, 640)], devices)
print("------------ Model Setting Finish ----------------")
print("")
K_final = torch.tensor(K, dtype=torch.float32).to(devices)

if os.path.isfile(os.path.join(pretrained_path, "CalibDNN_18_ALL" + '.pth')):
    print("Pretrained Model Open : ", model.get_name() + ".pth")
    checkpoint = load_weight_file(os.path.join(pretrained_path, "CalibDNN_18_ALL" + '.pth'))
    start_epoch = checkpoint['epoch']
    load_weight_parameter(model, checkpoint['state_dict'])
else:
    print("Pretrained Parameter Open : No Pretrained Model")
    start_epoch = 0
print("")
print("------------ Inference Start ----------------")

model.eval()
transform = transforms.Compose([transforms.ToTensor()])
AICamera_image = ns(glob.glob(dataset_path + "image/*.jpg"))
AICamera_lidar = ns(glob.glob(dataset_path + "lidar/*.pcd"))

if not os.path.exists(dataset_path + "init_depth_map"):
    os.makedirs(dataset_path  + "init_depth_map")
if not os.path.exists(dataset_path + "predicted_depth_map"):
    os.makedirs(dataset_path + "predicted_depth_map")

for image_file, lidar_file in zip(AICamera_image, AICamera_lidar):
    # Image, LIDAR Data Read
    filename, _ = lidar_file.split('.')
    file = filename.split('/')
    input_image = smc.imread(image_file)
    points = o3d.io.read_point_cloud(lidar_file)
    points = np.asarray(points.points)
    ones_col = np.ones(shape=(points.shape[0],1))
    points = np.hstack((points,ones_col))

    euler = mathutils.Euler((rpy[0], rpy[1], rpy[2]))
    R = euler.to_matrix()
    R.resize_4x4()
    T = mathutils.Matrix.Translation(mathutils.Vector((translation_vec[0], translation_vec[1], translation_vec[2])))
    RT = T @ R
    np_RT = np.array(RT)
    # Ground Truth Projection Image Get
    points_2d_Gt = np.matmul(np_RT, points.T)
    points_2d_Gt = np.matmul(K, points_2d_Gt[:-1, :])
    Z = points_2d_Gt[2,:]
    x = (points_2d_Gt[0,:]/Z).T
    y = (points_2d_Gt[1,:]/Z).T
    x = np.clip(x, 0.0, input_image.shape[1] - 1)
    y = np.clip(y, 0.0, input_image.shape[0] - 1)
    init_depthmap = np.zeros((input_image.shape[0], input_image.shape[1]))
    reprojected_img = input_image.copy()
    for x_idx, y_idx,z_idx in zip(x, y, Z):
        if(z_idx>0):
            init_depthmap[int(y_idx), int(x_idx)] = z_idx
            cv2.circle(reprojected_img, (int(x_idx), int(y_idx)), 1, (0, 0, 255), -1)
    smc.imsave(dataset_path + "init_depth_map/" + file[-1] + ".png", reprojected_img)

    # Predict
    source_image = input_image.copy()
    source_image[0:5, :] = 0.0
    source_image[:, 0:5] = 0.0
    source_image[source_image.shape[0] - 5:, :] = 0.0
    source_image[:, source_image.shape[1] - 5:] = 0.0
    source_image = (source_image - 127.5) / 127.5
    source_image = transform(source_image).to(devices)
    source_image = torch.unsqueeze(source_image, 0)

    source_map = init_depthmap.copy()
    source_map = np.repeat(np.expand_dims(source_map, axis=2), 3, axis=2)
    source_map[0:5, :] = 0.0
    source_map[:, 0:5] = 0.0
    source_map[source_map.shape[0] - 5:, :] = 0.0
    source_map[:, source_map.shape[1] - 5:] = 0.0
    source_map = (source_map - 100.0) / 100.0
    source_map = transform(source_map).to(devices)
    source_map = torch.unsqueeze(source_map, 0)

    if gpu_check:
        source_map = source_map.to(torch.float32)
        source_image = source_image.to(torch.float32)

    # Problem

    rotation, translation = model(source_image, source_map)

    R_predicted = quat2mat(rotation[0])
    T_predicted = tvector2mat(translation[0])
    predicted_RTMatrix = torch.mm(T_predicted, R_predicted).detach().cpu().numpy()

    transformed_points = np.matmul(predicted_RTMatrix, points.T)
    points_2d_Init = np.matmul(K, transformed_points[:-1, :])
    Z = points_2d_Init[2, :]
    x = (points_2d_Init[0, :] / Z).T
    y = (points_2d_Init[1, :] / Z).T
    x = np.clip(x, 0.0, input_image.shape[1] - 1)
    y = np.clip(y, 0.0, input_image.shape[0] - 1)
    init_depthmap = np.zeros((input_image.shape[0], input_image.shape[1]))
    projected_img = input_image.copy()
    for x_idx, y_idx, z_idx in zip(x, y, Z):
        if (z_idx > 0):
            init_depthmap[int(y_idx), int(x_idx)] = z_idx
            cv2.circle(projected_img, (int(x_idx), int(y_idx)), 1, (255, 0, 0), -1)
    smc.imsave(dataset_path + "predicted_depth_map/" + file[-1] + ".png", projected_img)

    print(file[-1] + "Predicted Success ! ")

print("Inference Finished!!")
