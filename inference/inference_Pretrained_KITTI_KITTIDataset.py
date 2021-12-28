import numpy as np
import torch.autograd
from dataset.dataset import *
from model.CalibDNN import *
from model.loss import *
from utils.AverageMeter import *
import time
import imageio as smc


print("------------ Ground Truth Rotation Translation matrix Start -----------------")
GT_RTMatrix = np.matmul(cf.KITTI_Info['cam_02_transform'], np.matmul(cf.KITTI_Info['R_rect_00'],
                                                                     cf.KITTI_Info['velo_to_cam']))
print(GT_RTMatrix)
print("------------ Ground Truth Rotation Translation matrix Finish -----------------")
print("")
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
K_final = torch.tensor(cf.KITTI_Info['K'], dtype=torch.float32).to(devices)
GT_RTMatrix_torch = torch.tensor(GT_RTMatrix, dtype=torch.float32).to(devices)
print("------------ GPU Setting Finish ----------------")
print("")
print("------------ Dataset Setting Start ----------------")
validationset = CalibDNNDataset(cf.paths['dataset_path'], cf.inference_info["rotation_range"],
                                cf.inference_info["translation_range"], training=False)
valid_loader = get_loader(validationset, batch_size=cf.inference_info['batch_size'], shuffle=False,
                          num_worker=cf.inference_info['num_worker'])
print("------------ Validation Dataset Setting Finish ----------------")
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
print("------------ Inference Start ----------------")

rotation_X = np.array([0.0], dtype=np.float32)
rotation_Y = np.array([0.0], dtype=np.float32)
rotation_Z = np.array([0.0], dtype=np.float32)
translation_X = np.array([0.0], dtype=np.float32)
translation_Y = np.array([0.0], dtype=np.float32)
translation_Z = np.array([0.0], dtype=np.float32)
count = 0
image_count = 0

for i_batch, sample_bathced in enumerate(valid_loader):
    count += 1
    source_depth_map = sample_bathced['source_depth_map']
    source_image = sample_bathced['source_image']
    expected_transform = sample_bathced['transform_matrix']
    point_cloud = sample_bathced['point_cloud']
    rotation_vector = sample_bathced['rotation_vector'].to(torch.float32)
    translation_vector = sample_bathced['translation_vector'].to(torch.float32)
    transform_matrix = sample_bathced['transform_matrix'].to(torch.float32)

    if gpu_check:
        source_depth_map = source_depth_map.to(devices)
        source_image = source_image.to(devices)
        expected_transform = expected_transform.to(devices)
        point_cloud = point_cloud.to(devices)
        rotation_vector = rotation_vector.to(devices)
        translation_vector = translation_vector.to(devices)
        transform_matrix = transform_matrix.to(devices)
    model_count = 0
    RTs = [transform_matrix[0].inverse()]
    with torch.no_grad():
        for model in models:
            rotation, translation = model(source_image, source_depth_map)
            R_predicted = quat2mat(rotation[0])
            T_predicted = tvector2mat(translation[0])
            RT_predicted = torch.mm(T_predicted, R_predicted)
            RTs.append(torch.mm(RTs[model_count], RT_predicted))

            points = point_cloud[0][0]
            points_in_cam_axis = torch.mm(GT_RTMatrix_torch, points.T)

            transformed_points = torch.mm(RTs[-1], points_in_cam_axis)
            points_2d = torch.mm(K_final, transformed_points[:-1, :])

            Z = points_2d[2, :]
            x = (points_2d[0, :] / Z).T
            y = (points_2d[1, :] / Z).T

            x = torch.clamp(x, 0.0, cf.KITTI_Info["WIDTH"] - 1).to(torch.long)
            y = torch.clamp(y, 0.0, cf.KITTI_Info["HEIGHT"] - 1).to(torch.long)

            Z_Index = torch.where(Z > 0)[0]
            source_map = torch.zeros((cf.KITTI_Info["HEIGHT"], cf.KITTI_Info["WIDTH"])).to(devices)
            source_map[y[Z_Index], x[Z_Index]] = Z[Z_Index]

            smc.imsave(
                cf.paths["inference_img_result_path"] + "/Pretrained_KITTI/predicted_" + str(
                    image_count) + "_" + str(model_count) + ".png",
                source_map.detach().cpu().numpy())

            model_count += 1
            source_map[0:5, :] = 0.0
            source_map[:, 0:5] = 0.0
            source_map[source_map.shape[0] - 5:, :] = 0.0
            source_map[:, source_map.shape[1] - 5:] = 0.0
            source_map = (source_map - 40.0) / 40.0
            source_map = torch.repeat_interleave(torch.unsqueeze(source_map, dim=0), 3, dim=0)
            source_map = torch.unsqueeze(source_map, dim=0)
            # source_map = torch.repeat_interleave(torch.unsqueeze(source_map, dim=0), 1, dim=0)
            source_depth_map = source_map

    predicted_Matrix = torch.mm(transform_matrix[0], RTs[-1])

    rotation_gt = rotation_vector[0].detach().cpu().numpy()
    translation_gt = translation_vector[0].detach().cpu().numpy()
    tra_pre = predicted_Matrix[:, 3]
    rot_pre_norm = quaternion_from_matrix(predicted_Matrix)
    rot_pre_euler = yaw_pitch_roll(rot_pre_norm.detach().cpu().numpy())
    rot_gt_euler = yaw_pitch_roll(rotation_gt)
    rot_pre_degree = np.rad2deg(rot_pre_euler)
    rot_gt_degree = np.rad2deg(rot_gt_euler)
    tra_pre = tra_pre.detach().cpu().numpy()
    rotation_X += np.abs(rot_gt_degree[0] - rot_pre_degree[0])
    rotation_Y += np.abs(rot_gt_degree[1] - rot_pre_degree[1])
    rotation_Z += np.abs(rot_gt_degree[2] - rot_pre_degree[2])
    translation_X += np.abs(translation_gt[0] - tra_pre[0])
    translation_Y += np.abs(translation_gt[1] - tra_pre[1])
    translation_Z += np.abs(translation_gt[2] - tra_pre[2])

    R_predicted = quat2mat(rotation[0])
    T_predicted = tvector2mat(translation[0])
    RT_predicted = predicted_Matrix.detach().cpu().numpy()

    source_map = source_depth_map[0]
    current_img = source_image[0]
    predicted_img = source_image[0]
    point_clouds = point_cloud[0][0].detach().cpu().numpy()
    predicted_translation_vector = translation[0].detach().cpu().numpy()
    predicted_rotation_vector = rotation[0].detach().cpu().numpy()
    random_transform_inverse = transform_matrix[0].detach().cpu().numpy()
    random_transform = np.linalg.inv(random_transform_inverse)
    K = K_final.detach().cpu().numpy()
    current_img = current_img * 127.5 + 127.5
    current_img = current_img.permute(1,2,0).detach().cpu().numpy()

    transformed_points = np.matmul(RT_predicted, np.matmul(random_transform, np.matmul(GT_RTMatrix, point_clouds.T)))
    points_2d_Init = np.matmul(K, transformed_points[:-1, :])
    Z = points_2d_Init[2, :]
    x = (points_2d_Init[0, :] / Z).T
    y = (points_2d_Init[1, :] / Z).T
    x = np.clip(x, 0.0, cf.KITTI_Info['WIDTH'] - 1)
    y = np.clip(y, 0.0, cf.KITTI_Info['HEIGHT'] - 1)
    projected_img = current_img.copy()
    for x_idx, y_idx, z_idx in zip(x, y, Z):
        if (z_idx > 0):
            cv2.circle(projected_img, (int(x_idx), int(y_idx)), 1, (255, 0, 0), -1)

    point_cloud_gt = np.matmul(GT_RTMatrix, point_clouds.T)
    points_2d_gt = np.matmul(K, point_cloud_gt[:-1, :])
    Z = points_2d_gt[2, :]
    x = (points_2d_gt[0, :] / Z).T
    y = (points_2d_gt[1, :] / Z).T
    x = np.clip(x, 0.0, cf.KITTI_Info["WIDTH"] - 1)
    y = np.clip(y, 0.0, cf.KITTI_Info['HEIGHT'] - 1)
    reprojected_img = current_img.copy()
    for x_idx, y_idx, z_idx in zip(x, y, Z):
        if (z_idx > 0):
            cv2.circle(reprojected_img, (int(x_idx), int(y_idx)), 1, (255, 0, 0), -1)
    smc.imsave(
        cf.paths["inference_img_result_path"] + "/Pretrained_KITTI/KITTIDatset_" + str(image_count) + "_Target.png",
        reprojected_img)
    smc.imsave(cf.paths["inference_img_result_path"] + "/Pretrained_KITTI/KITTIDatset_" + str(image_count) + "_Predicted.png",
               projected_img)

    image_count += 1
    if i_batch % cf.network_info['freq_print'] == 0:
        print('Test: [{0}/{1}] Evaluation Metrics Success'.format(i_batch, len(valid_loader)))
        print("[ROT X] : ", rotation_X / count, " [ROT Y] : ", rotation_Y / count, " [ROT Z] : ", rotation_Z / count,
              " [TRANSLATION X] : ", translation_X / count, " [TRANSLATION Y] : ", translation_Y / count,
              "[TRANSLATION Z] :",
              translation_Z / count)

print("[ROT X] : ", rotation_X/count, " [ROT Y] : ", rotation_Y/count, " [ROT Z] : ", rotation_Z/count,
      " [TRANSLATION X] : ", translation_X/count, " [TRANSLATION Y] : ", translation_Y/count, "[TRANSLATION Z] :",
      translation_Z/count)

