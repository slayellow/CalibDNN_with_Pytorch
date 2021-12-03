import torch

from model.CalibDNN import *
from model.loss_changwon import *
from utils.AverageMeter import *
import imageio as smc
from natsort import natsorted as ns
import glob, os
import cv2
from scipy import io
import torchvision.transforms as transforms


dataset_path = '/data/AICamera/'

K = np.array([734.9760, 0.0, 625.2400, 0.0, 736.0504, 523.6243, 0.0, 0.0, 1.0]).reshape(3, 3)
GT_RTMatrix = np.array([710.628311, 648.392377, -75.561497, -167956.992177,
                        -26.577065, 442.323933, -787.144066, -372986.043192,
                        -0.038042, 0.993686, -0.105549, -191.677303]).reshape((3, 4))


print("------------ GPU Setting Start ----------------")
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False

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
model = CalibDNN18(18).to(devices)
print("------------ Model Summary ----------------")
summary(model, [(1, 3, 375, 1242), (1, 3, 375, 1242)], devices)
print("------------ Model Setting Finish ----------------")
print("")
K_final = torch.tensor(K, dtype=torch.float32).to(devices)

pretrained_path = cf.paths['pretrained_path']
if os.path.isfile(os.path.join(pretrained_path, model.get_name() + '.pth')):
    print("Pretrained Model Open : ", model.get_name() + ".pth")
    checkpoint = load_weight_file(os.path.join(pretrained_path, model.get_name() + '.pth'))
    start_epoch = checkpoint['epoch']
    load_weight_parameter(model, checkpoint['state_dict'])
else:
    print("Pretrained Parameter Open : No Pretrained Model")
    start_epoch = 0
print("")
print("------------ Inference Start ----------------")

model.eval()
transform = transforms.Compose([transforms.ToTensor()])
AICamera_image = ns(glob.glob(dataset_path + "image/*.png"))
AICamera_lidar = ns(glob.glob(dataset_path + "lidar/*.txt"))

for image_file, lidar_file in zip(AICamera_image, AICamera_lidar):
    # Image, LIDAR Data Read
    filename, _ = image_file.split('.')
    file = filename.split('/')
    input_image = smc.imread(image_file)
    points = np.loadtxt(lidar_file, delimiter=',')
    points = points.reshape((-1, 5))
    points = points[:, 1:4] * 10.0
    # useless_idx = np.where(points[:, 0] != 0)
    # points = points[useless_idx, :]
    # useless_idx = np.where(points[0] != -8018.9)
    # points = points[useless_idx]
    ones_col = np.ones(shape=(points.shape[0],1))
    points = np.hstack((points,ones_col))

    # Ground Truth Projection Image Get
    points_2d_Gt = np.matmul(GT_RTMatrix, points.T)
    Z = points_2d_Gt[2,:]
    x = (points_2d_Gt[0,:]/Z).T
    y = (points_2d_Gt[1,:]/Z).T
    x = np.clip(x, 0.0, input_image.shape[1] - 1)
    y = np.clip(y, 0.0, input_image.shape[0] - 1)
    reprojected_img = input_image.copy()
    for x_idx, y_idx,z_idx in zip(x, y, Z):
        if(z_idx>0):
            cv2.circle(reprojected_img, (int(x_idx), int(y_idx)), 2, (0, 0, 255), -1)
    smc.imsave(dataset_path + "MatlabResult/" + file[-1] + ".png", reprojected_img)

    # Init Depth Map
    R = mathutils.Euler((math.radians(-90.0), 0.0, 0.0))
    T = mathutils.Vector((0.0, 0.0, 0.0))
    R = R.to_matrix()
    R.resize_4x4()
    T = mathutils.Matrix.Translation(T)
    Init_RT = T @ R

    points = np.loadtxt(lidar_file, delimiter=',')
    points = points.reshape((-1, 5))
    points = points[:, 1:4] * 0.01
    ones_col = np.ones(shape=(points.shape[0], 1))
    points = np.hstack((points, ones_col))

    transformed_points = np.matmul(Init_RT, points.T)
    points_2d_Init = np.matmul(K, transformed_points[:-1, :])
    Z = points_2d_Init[2, :]
    Z_idx = np.where(Z < 200)
    x = (points_2d_Init[0, :] / Z).T
    y = (points_2d_Init[1, :] / Z).T
    Z = Z[Z_idx]
    x = x[Z_idx]
    y = y[Z_idx]
    x = np.clip(x, 0.0, input_image.shape[1] - 1)
    y = np.clip(y, 0.0, input_image.shape[0] - 1)
    init_depthmap = np.zeros((input_image.shape[0], input_image.shape[1]))
    projected_img = input_image.copy()
    for x_idx, y_idx, z_idx in zip(x, y, Z):
        if (z_idx > 0):
            init_depthmap[int(y_idx), int(x_idx)] = z_idx
            cv2.circle(projected_img, (int(x_idx), int(y_idx)), 2, (255, 0, 0), -1)
    smc.imsave(dataset_path + "InitDepthMap/" + file[-1] + ".png", init_depthmap)
    smc.imsave(dataset_path + "InitProjectionResult/" + file[-1] + ".png", projected_img)

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
    source_map = (source_map - 40.0) / 40.0
    source_map = transform(source_map).to(devices)
    source_map = torch.unsqueeze(source_map, 0)

    if gpu_check:
        source_map = source_map.to(torch.float32)
        source_image = source_image.to(torch.float32)

    # Problem
    print("LIDAR : ", torch.max(source_map))
    print("IMAGE : ", torch.max(source_image))
    rotation, translation = model(source_image, source_map)

    # Save Predicted Depth Map
    rotation = rotation.detach().cpu().numpy()
    translation = translation.detach().cpu().numpy()
    print("Rotation ", rotation[0])
    print("Translation ", translation[0])
    predicted_RTMatrix = convert_6DoF_to_RTMatrix(rotation[0], translation[0])

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
            cv2.circle(projected_img, (int(x_idx), int(y_idx)), 2, (255, 0, 0), -1)
    smc.imsave(dataset_path + "PredictedDepthMap/" + file[-1] + ".png", init_depthmap)
    smc.imsave(dataset_path + "PredictedProjectionResult/" + file[-1] + ".png", projected_img)

    print(file[-1] + "Predicted Success ! ")

print("Inference Finished!!")
