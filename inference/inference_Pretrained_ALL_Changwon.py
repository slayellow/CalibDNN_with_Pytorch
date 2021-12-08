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


dataset_path = '/data/2021_12_02/test/2021-05-24_18-55-48/'


IMG_HT = 480
IMG_WDT = 640

K = np.array([384.8557, 0, 328.4401,
              0, 345.4014, 245.6107,
              0, 0, 1]).reshape(3, 3)
ExtrinsicParameter_meter = np.array([372.987160, 339.659994, -38.710366, -82.675118467,
                               -11.669059, 208.264253, -368.939818, -171.438409027,
                               -0.035360, 0.994001, -0.103494, -0.179826270,
                                     0.0, 0.0, 0.0, 1.0]).reshape(4, 4)

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
model = CalibDNN18(18,  pretrained=os.path.join(pretrained_path, "CalibDNN_18_ALL" + '.pth')).to(devices)
print("------------ Model Summary ----------------")
summary(model, [(1, 3, 375, 1242), (1, 3, 375, 1242)], devices)
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

AICamera_image = ns(glob.glob(dataset_path + "images/*.png"))
AICamera_Point = ns(glob.glob(dataset_path + "depth_maps_transformed/*.png"))
transforms_list = ns(glob.glob(dataset_path + "lidars/*.txt"))

rotation_X = np.array([0.0], dtype=np.float32)
rotation_Y = np.array([0.0], dtype=np.float32)
rotation_Z = np.array([0.0], dtype=np.float32)
translation_X = np.array([0.0], dtype=np.float32)
translation_Y = np.array([0.0], dtype=np.float32)
translation_Z = np.array([0.0], dtype=np.float32)

count = 0
for image_file, depth_map, pointfile in zip(AICamera_image, AICamera_Point, transforms_list):
    count += 1
    # Image, LIDAR Data Read
    filename, _ = image_file.split('.')
    file = filename.split('/')
    input_image = smc.imread(image_file)
    points = np.loadtxt(pointfile)
    points = points[:, :3]
    ones_col = np.ones(shape=(points.shape[0], 1))
    points = np.hstack((points,ones_col))

    # Ground Truth Projection Image Get

    points_2d_Gt = np.matmul(ExtrinsicParameter_meter, points.T)
    Z = points_2d_Gt[2,:]
    x = (points_2d_Gt[0,:]/Z).T
    y = (points_2d_Gt[1,:]/Z).T
    x = np.clip(x, 0.0, input_image.shape[1] - 1)
    y = np.clip(y, 0.0, input_image.shape[0] - 1)
    reprojected_img = input_image.copy()
    for x_idx, y_idx,z_idx in zip(x, y, Z):
        if(z_idx>0):
            cv2.circle(reprojected_img, (int(x_idx), int(y_idx)), 1, (255, 0, 0), -1)
    smc.imsave(cf.paths['inference_img_result_path'] + "/Pretrained_ALL_Changwon_Target.png", reprojected_img)

    # Predict
    source_image = input_image.copy()
    source_image[0:5, :] = 0.0
    source_image[:, 0:5] = 0.0
    source_image[source_image.shape[0] - 5:, :] = 0.0
    source_image[:, source_image.shape[1] - 5:] = 0.0
    source_image = (source_image - 127.5) / 127.5
    source_image = transform(source_image).to(devices)
    source_image = torch.unsqueeze(source_image, 0)

    init_depthmap = cv2.imread(depth_map, flags=cv2.IMREAD_GRAYSCALE)
    source_map = init_depthmap.copy()
    source_map = np.repeat(np.expand_dims(source_map, axis=2), 3, axis=2)
    source_map[0:5, :] = 0.0
    source_map[:, 0:5] = 0.0
    source_map[source_map.shape[0] - 5:, :] = 0.0
    source_map[:, source_map.shape[1] - 5:] = 0.0
    source_map = (source_map - 127.5) / 127.5
    source_map = transform(source_map).to(devices)
    source_map = torch.unsqueeze(source_map, 0)

    if gpu_check:
        source_map = source_map.to(torch.float32)
        source_image = source_image.to(torch.float32)

    rotation, translation = model(source_image, source_map)

    K_inverse = np.linalg.inv(K)
    transformed_matrix = np.matmul(K_inverse, ExtrinsicParameter_meter[:-1])
    transformed_matrix = np.vstack((transformed_matrix, [0.0, 0.0, 0.0, 1.0]))
    GT_RTMatrix = transformed_matrix
    tra_gt = np.array(GT_RTMatrix[:-1, 3]).T
    rot_gt = quaternion_from_matrix(torch.Tensor(GT_RTMatrix)).detach().cpu().numpy()
    for rot_pre, tra_pre in zip(rotation, translation):
        R_predicted = quat2mat(rot_pre)
        T_predicted = tvector2mat(tra_pre)
        RT_predicted = torch.mm(T_predicted, R_predicted)
        rot_pre_norm = quaternion_from_matrix(RT_predicted)
        rot_pre_euler = yaw_pitch_roll(rot_pre_norm.detach().cpu().numpy())
        rot_gt_euler = yaw_pitch_roll(rot_gt)
        rot_pre_degree = np.rad2deg(rot_pre_euler)
        rot_gt_degree = np.rad2deg(rot_gt_euler)
        tra_pre = tra_pre.detach().cpu().numpy()
        rotation_X += np.abs(rot_gt_degree[0] - rot_pre_degree[0])
        rotation_Y += np.abs(rot_gt_degree[1] - rot_pre_degree[1])
        rotation_Z += np.abs(rot_gt_degree[2] - rot_pre_degree[2])
        translation_X += np.abs(tra_gt[0] - tra_pre[0])
        translation_Y += np.abs(tra_gt[1] - tra_pre[1])
        translation_Z += np.abs(tra_gt[2] - tra_pre[2])

    R_predicted = quat2mat(rotation[0])
    T_predicted = tvector2mat(translation[0])
    RT_predicted = torch.mm(T_predicted, R_predicted).detach().cpu().numpy()

    # Save Predicted Depth Map

    transformed_points = np.matmul(RT_predicted, points.T)
    points_2d_Init = np.matmul(K, transformed_points[:-1, :])
    Z = points_2d_Init[2, :]
    x = (points_2d_Init[0, :] / Z).T
    y = (points_2d_Init[1, :] / Z).T
    x = np.clip(x, 0.0, input_image.shape[1] - 1)
    y = np.clip(y, 0.0, input_image.shape[0] - 1)
    projected_img = input_image.copy()
    for x_idx, y_idx, z_idx in zip(x, y, Z):
        if (z_idx > 0):
            cv2.circle(projected_img, (int(x_idx), int(y_idx)), 1, (255, 0, 0), -1)
    smc.imsave(cf.paths['inference_img_result_path'] + "/Pretrained_ALL_Changwon_Predicted.png", projected_img)
    if count % cf.network_info['freq_print'] == 0:
        print("[ROT X] : ", rotation_X / count, " [ROT Y] : ", rotation_Y / count, " [ROT Z] : ", rotation_Z / count,
              " [TRANSLATION X] : ", translation_X / count, " [TRANSLATION Y] : ", translation_Y / count,
              "[TRANSLATION Z] :",
              translation_Z / count)
        break  # 2021.12.05.

print("[ROT X] : ", rotation_X / count, " [ROT Y] : ", rotation_Y / count, " [ROT Z] : ", rotation_Z / count,
      " [TRANSLATION X] : ", translation_X / count, " [TRANSLATION Y] : ", translation_Y / count, "[TRANSLATION Z] :",
      translation_Z / count)
