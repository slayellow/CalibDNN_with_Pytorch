import cffi.model
import numpy as np
import torch.autograd

from dataset.dataset import *
from model.CalibDNN import *
from model.loss import *
from utils.AverageMeter import *
import time
import imageio as smc


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
print("------------ Dataset Setting Start ----------------")
validationset = CalibDNNDataset(cf.paths['dataset_path'], training=False)
valid_loader = get_loader(validationset, batch_size=2, shuffle=False,
                          num_worker=cf.network_info['num_worker'])
print("------------ Validation Dataset Setting Finish ----------------")
print("")
print("------------ Model Setting Start ----------------")
model = CalibDNN18(18,  pretrained=os.path.join(pretrained_path, "CalibDNN_18_KITTI" + '.pth')).to(devices)
print("------------ Model Summary ----------------")
summary(model, [(1, 3, 375, 1242), (1, 3, 375, 1242)], devices)
print("------------ Model Setting Finish ----------------")
print("")
K_final = torch.tensor(cf.K, dtype=torch.float32).to(devices)


if os.path.isfile(os.path.join(pretrained_path, model.get_name() + '.pth')):
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

rotation_X = np.array([0.0], dtype=np.float32)
rotation_Y = np.array([0.0], dtype=np.float32)
rotation_Z = np.array([0.0], dtype=np.float32)
translation_X = np.array([0.0], dtype=np.float32)
translation_Y = np.array([0.0], dtype=np.float32)
translation_Z = np.array([0.0], dtype=np.float32)

count = 0
for i_batch, sample_bathced in enumerate(valid_loader):
    source_depth_map = sample_bathced['source_depth_map']
    source_image = sample_bathced['source_image']
    target_depth_map = sample_bathced['target_depth_map']
    expected_transform = sample_bathced['transform_matrix']
    point_cloud = sample_bathced['point_cloud']
    rotation_vector = sample_bathced['rotation_vector'].to(torch.float32)
    translation_vector = sample_bathced['translation_vector'].to(torch.float32)
    transform_matrix = sample_bathced['transform_matrix'].to(torch.float32)

    if gpu_check:
        source_depth_map = source_depth_map.to(devices)
        source_image = source_image.to(devices)
        target_depth_map = target_depth_map.to(devices)
        expected_transform = expected_transform.to(devices)
        point_cloud = point_cloud.to(devices)
        rotation_vector = rotation_vector.to(devices)
        translation_vector = translation_vector.to(devices)
        transform_matrix = transform_matrix.to(devices)

    rotation, translation = model(source_image, source_depth_map)

    print(rotation[0])
    if rotation[0].norm() != 1.:
        rotation[0] = rotation[0] / rotation[0].norm()
    print(rotation[0])
    print(translation[0])

    rotation_predicted = rotation
    translation_predicted = translation
    rotation_gt = rotation_vector.detach().cpu().numpy()
    translation_gt = translation_vector.detach().cpu().numpy()

    count += translation_gt.shape[0]

    for rot_pre, rot_gt, tra_pre, tra_gt in zip(rotation_predicted, rotation_gt, translation_predicted, translation_gt):
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

    # R_predicted = quat2mat(rotation[0])
    # T_predicted = tvector2mat(translation[0])
    # RT_predicted = torch.mm(T_predicted, R_predicted).detach().cpu().numpy()

    # Save Predicted Depth Map
    # source_map = source_depth_map[0]
    # gt_depth_map = target_depth_map[0]
    # current_img = source_image[0]
    # predicted_img = source_image[0]
    # point_clouds = point_cloud[0][0].detach().cpu().numpy()
    # predicted_translation_vector = translation[0].detach().cpu().numpy()
    # predicted_rotation_vector = rotation[0].detach().cpu().numpy()
    # gt_rt_matrix = transform_matrix[0].detach().cpu().numpy()
    # K = K_final.detach().cpu().numpy()
    # current_img = current_img * 127.5 + 127.5
    # current_img = current_img.permute(1,2,0).detach().cpu().numpy()

    # transformed_points = np.matmul(RT_predicted, point_clouds.T)
    # points_2d_Init = np.matmul(K, transformed_points[:-1, :])
    # Z = points_2d_Init[2, :]
    # x = (points_2d_Init[0, :] / Z).T
    # y = (points_2d_Init[1, :] / Z).T
    # x = np.clip(x, 0.0, cf.camera_info['WIDTH'] - 1)
    # y = np.clip(y, 0.0, cf.camera_info['HEIGHT'] - 1)
    # projected_img = current_img.copy()
    # for x_idx, y_idx, z_idx in zip(x, y, Z):
    #     if (z_idx > 0):
    #         cv2.circle(projected_img, (int(x_idx), int(y_idx)), 1, (255, 0, 0), -1)
    #
    # point_cloud_gt = np.matmul(gt_rt_matrix, point_clouds.T)
    # points_2d_gt = np.matmul(K, point_cloud_gt[:-1, :])
    # Z = points_2d_gt[2, :]
    # x = (points_2d_gt[0, :] / Z).T
    # y = (points_2d_gt[1, :] / Z).T
    # x = np.clip(x, 0.0, cf.camera_info["WIDTH"] - 1)
    # y = np.clip(y, 0.0, cf.camera_info['HEIGHT'] - 1)
    # reprojected_img = current_img.copy()
    # for x_idx, y_idx, z_idx in zip(x, y, Z):
    #     if (z_idx > 0):
    #         cv2.circle(reprojected_img, (int(x_idx), int(y_idx)), 1, (255, 0, 0), -1)
    # smc.imsave(
    #     cf.paths["inference_img_result_path"] + "/Pretrained_KITTI_KITTIDatset_Target.png",
    #     reprojected_img)
    # smc.imsave(cf.paths["inference_img_result_path"] + "/Pretrained_KITTI_KITTIDatset_Predicted.png",
    #            projected_img)

    if i_batch % cf.network_info['freq_print'] == 0:
        print('Test: [{0}/{1}] Evaluation Metrics Success'.format(i_batch, len(valid_loader)))
        print("[ROT X] : ", rotation_X / count, " [ROT Y] : ", rotation_Y / count, " [ROT Z] : ", rotation_Z / count,
              " [TRANSLATION X] : ", translation_X / count, " [TRANSLATION Y] : ", translation_Y / count,
              "[TRANSLATION Z] :",
              translation_Z / count)

print("[ROT X] : ", rotation_X/count, " [ROT Y] : ", rotation_Y/count, " [ROT Z] : ", rotation_Z/count,
      " [TRANSLATION X] : ", translation_X/count, " [TRANSLATION Y] : ", translation_Y/count, "[TRANSLATION Z] :",
      translation_Z/count)

