import numpy as np
import torch
import math

from model.CalibDNN import *
from model.loss_changwon import *
from utils.AverageMeter import *
import imageio as smc
from natsort import natsorted as ns
import glob, os
import cv2
import torchvision.transforms as transforms


lidar_path = '/data/Unmanned/lidar/000000.txt'
image_path = '/data/Unmanned/image/000000.png'

K = np.array([355.128821, 0, 315.192763,
              0, 353.926131, 233.417237,
              0, 0, 1]).reshape(3, 3)

Ground_Truth = np.array([350.8093, 319.9850, 2.3275, 4.0484051,
                         -0.8352, 233.4116, -353.9289, 10.6080617,
                         -0.0136, 0.9999, -0.0, -0.0001580,
                         ]).reshape(3, 4)


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
model = CalibDNN18(18, pretrained=os.path.join(pretrained_path, "CalibDNN_18_KITTI" + '.pth')).to(devices)
print("------------ Model Summary ----------------")
summary(model, [(1, 3, 480, 640), (1, 3, 480, 640)], devices)
print("------------ Model Setting Finish ----------------")
print("")
K_final = torch.tensor(K, dtype=torch.float32).to(devices)

if os.path.isfile(os.path.join(pretrained_path, "CalibDNN_18_KITTI" + '.pth')):
    print("Pretrained Model Open : ", model.get_name() + ".pth")
    checkpoint = load_weight_file(os.path.join(pretrained_path, "CalibDNN_18_KITTI" + '.pth'))
    start_epoch = checkpoint['epoch']
    load_weight_parameter(model, checkpoint['state_dict'])
else:
    print("Pretrained Parameter Open : No Pretrained Model")
    start_epoch = 0
print("")
print("------------ Inference Start ----------------")

model.eval()
transform = transforms.Compose([transforms.ToTensor()])

if not os.path.exists("/data/Unmanned/init_depth_map"):
    os.makedirs("/data/Unmanned/init_depth_map")
if not os.path.exists("/data/Unmanned/predicted_depth_map" ):
    os.makedirs("/data/Unmanned/predicted_depth_map" )

# Image, LIDAR Data Read
filename, _ = lidar_path.split('.')
file = filename.split('/')
input_image = smc.imread(image_path)
points = np.loadtxt(lidar_path)
ones_col = np.ones(shape=(points.shape[0], 1))
points = np.hstack((points, ones_col))

euler = mathutils.Euler((math.radians(-90.0), 0.0, 0.0))
R = euler.to_matrix()
R.resize_4x4()
T = mathutils.Matrix.Translation(mathutils.Vector((0.0, 0.0, 0.0)))
RT = T @ R
np_RT = np.array(RT)
# Ground Truth Projection Image Get
points_2d_Gt = np.matmul(np_RT, points.T)
points_2d_Gt = np.matmul(K, points_2d_Gt[:-1, :])
# points_2d_Gt = np.matmul(Ground_Truth, points.T)
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
smc.imsave("/data/Unmanned/init_depth_map/" + file[-1] + ".png", reprojected_img)

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

print(rotation[0])
if rotation[0].norm() != 1.:
    rotation[0] = rotation[0] / rotation[0].norm()
print(rotation[0])
print(translation[0])

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
smc.imsave("/data/Unmanned/predicted_depth_map/" + file[-1] + ".png", projected_img)

print(file[-1] + "Predicted Success ! ")

print("Inference Finished!!")
