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
valid_loader = get_loader(validationset, batch_size=cf.network_info['batch_size'], shuffle=False,
                          num_worker=cf.network_info['num_worker'])
print("------------ Validation Dataset Setting Finish ----------------")
print("")
print("------------ Model Setting Start ----------------")
model = CalibDNN18(18).to(devices)
print("------------ Model Summary ----------------")
summary(model, [(1, 3, 375, 1242), (1, 3, 375, 1242)], devices)
print("------------ Model Setting Finish ----------------")
print("")
K_final = torch.tensor(cf.K, dtype=torch.float32).to(devices)

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

    # Save Predicted Depth Map
    source_map = source_depth_map[0]
    gt_depth_map = target_depth_map[0]
    current_img = source_image[0]
    predicted_img = source_image[0]
    point_clouds = point_cloud[0]
    predicted_translation_vector = translation[0]
    predicted_rotation_vector = rotation[0]
    gt_rt_matrix = transform_matrix[0]

    point_cloud_gt = torch.mm(gt_rt_matrix, point_clouds[0].t()).to(predicted_rotation_vector.device)
    points_2d_gt = torch.mm(K_final, point_cloud_gt[:-1, :])
    Z = points_2d_gt[2, :]
    x = (points_2d_gt[0, :] / Z).t()
    y = (points_2d_gt[1, :] / Z).t()
    x = torch.clamp(x, 0.0, cf.camera_info["WIDTH"] - 1).to(torch.long)
    y = torch.clamp(y, 0.0, cf.camera_info['HEIGHT'] - 1).to(torch.long)
    Z_Index = torch.where(Z > 0)
    gt_rgb_map = torch.zeros_like(gt_depth_map[0])
    gt_rgb_map[y[Z_Index], x[Z_Index]] = Z[Z_Index]
    gt_rgb_map[0:5, :] = 0.0
    gt_rgb_map[:, 0:5] = 0.0
    gt_rgb_map[gt_rgb_map.shape[0] - 5:, :] = 0.0
    gt_rgb_map[:, gt_rgb_map.shape[1] - 5:] = 0.0
    gt_rgb_map = torch.unsqueeze(gt_rgb_map, dim=2)
    gt_rgb_map = gt_rgb_map.repeat(1,1,3)
    current_img = current_img * 127.5 + 127.5
    current_img = current_img.permute(1,2,0)
    current_img[y[Z_Index], x[Z_Index]] = torch.Tensor([0, 0, 255]).to(devices)
    reconstructed_img = current_img
    # reconstructed_img = current_img * (gt_rgb_map > 0.)
    smc.imsave(
        cf.paths["inference_img_result_path"] + "/iteration_" + str(i_batch) + "_target.png",
        (reconstructed_img.detach().cpu().numpy().astype(np.uint8)))

    R_predicted = quat2mat(predicted_rotation_vector)
    T_predicted = tvector2mat(predicted_translation_vector)
    RT_predicted = torch.mm(T_predicted, R_predicted)
    point_cloud_out = torch.mm(RT_predicted, point_clouds[0].t())
    points_2d_predicted = torch.mm(K_final, point_cloud_out[:-1, :])
    Z = points_2d_predicted[2, :]
    x = (points_2d_predicted[0, :] / Z).t()
    y = (points_2d_predicted[1, :] / Z).t()
    x = torch.clamp(x, 0.0, cf.camera_info["WIDTH"] - 1).to(torch.long)
    y = torch.clamp(y, 0.0, cf.camera_info['HEIGHT'] - 1).to(torch.long)
    Z_Index = torch.where(Z > 0)
    predicted_depth_map = torch.zeros_like(gt_depth_map[0])
    predicted_depth_map[y[Z_Index], x[Z_Index]] = Z[Z_Index]
    predicted_depth_map[0:5, :] = 0.0
    predicted_depth_map[:, 0:5] = 0.0
    predicted_depth_map[predicted_depth_map.shape[0] - 5:, :] = 0.0
    predicted_depth_map[:, predicted_depth_map.shape[1] - 5:] = 0.0
    predicted_depth_map = torch.unsqueeze(predicted_depth_map, dim=2)
    predicted_depth_map = predicted_depth_map.repeat(1,1,3)
    predicted_img = predicted_img * 127.5 + 127.5
    predicted_img = predicted_img.permute(1,2,0)
    predicted_img[y[Z_Index], x[Z_Index]] = torch.Tensor([0, 0, 255]).to(devices)
    reconstructed_img = predicted_img
    # reconstructed_img = current_img * (predicted_depth_map > 0.)
    smc.imsave(cf.paths["inference_img_result_path"] + "/iteration_" + str(i_batch) + "_predicted.png",
               reconstructed_img.detach().cpu().numpy().astype(np.uint8))

    if i_batch % cf.network_info['freq_print'] == 0:
        print('Test: [{0}/{1}] Image Save Success'.format(i_batch, len(valid_loader)))

print("Inference Finished!!")
