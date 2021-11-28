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
# dataset test code
trainingset = CalibDNNDataset(cf.paths['dataset_path'], training=True)
data_loader = get_loader(trainingset, batch_size=cf.network_info['batch_size'], shuffle=True,
                         num_worker=cf.network_info['num_worker'])
print("------------ Training Dataset Setting Finish ----------------")
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
print("------------ Loss Function Setting Start ----------------")
loss_function = TotalLoss(rotation_weight=cf.network_info['rotation_weight'],
                          translation_weight=cf.network_info['translation_weight'],
                          depth_map_loss_weight=cf.network_info['depth_map_loss_weight'],
                          point_cloud_loss_weight=cf.network_info['point_cloud_loss_weight']).to(devices)
print("------------ Loss Function Setting Finish ----------------")
learning_rate = cf.network_info['learning_rate']

optimizer = set_Adam(model, learning_rate=learning_rate)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 50, 70], gamma=0.5)

pretrained_path = cf.paths['pretrained_path']
if os.path.isfile(os.path.join(pretrained_path, model.get_name() + '.pth')):
    print("Pretrained Model Open : ", model.get_name() + ".pth")
    checkpoint = load_weight_file(os.path.join(pretrained_path, model.get_name() + '.pth'))
    start_epoch = checkpoint['epoch']
    load_weight_parameter(model, checkpoint['state_dict'])
    load_weight_parameter(optimizer, checkpoint['optimizer'])
else:
    print("Pretrained Parameter Open : No Pretrained Model")
    start_epoch = 0
print("")
print("------------ Train Start ----------------")
for epoch in range(start_epoch, cf.network_info['epochs']):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    trans_loss = AverageMeter()
    depth_loss = AverageMeter()
    point_loss = AverageMeter()

    model.train()

    end = time.time()

    for i_batch, sample_bathced in enumerate(data_loader):
        data_time.update(time.time() - end)

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

        optimizer.zero_grad()

        rotation, translation = model(source_image, source_depth_map)

        loss = loss_function(point_cloud, translation_vector, rotation_vector,
                             translation, rotation, transform_matrix, K_final, target_depth_map)

        loss['total_loss'].backward()
        optimizer.step()

        losses.update(loss['total_loss'].item(), source_depth_map.size(0))
        trans_loss.update(loss['transformation_loss'].item(), source_depth_map.size(0))
        depth_loss.update(loss['depth_map_loss'].item(), source_depth_map.size(0))
        point_loss.update(loss['point_clouds_loss'].item(), source_depth_map.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # Save Predicted Depth Map
        # source_map = source_depth_map[0]
        # gt_depth_map = target_depth_map[0]
        # point_clouds = point_cloud[0]
        # predicted_translation_vector = translation[0]
        # predicted_rotation_vector = rotation[0]
        # R_predicted = quat2mat(predicted_rotation_vector)
        # T_predicted = tvector2mat(predicted_translation_vector)
        # RT_predicted = torch.mm(T_predicted, R_predicted)
        # point_cloud_out = torch.mm(RT_predicted,point_clouds[0].t())
        # points_2d_predicted = torch.mm(K_final, point_cloud_out[:-1, :])
        # Z = points_2d_predicted[2, :]
        # x = (points_2d_predicted[0, :] / Z).t()
        # y = (points_2d_predicted[1, :] / Z).t()
        #
        # x = torch.clamp(x, 0.0, cf.camera_info["WIDTH"] - 1).to(torch.long)
        # y = torch.clamp(y, 0.0, cf.camera_info['HEIGHT'] - 1).to(torch.long)
        #
        # # High Speed ( 2021. 11. 25. )
        # Z_Index = torch.where(Z > 0)
        # predicted_depth_map = torch.zeros_like(gt_depth_map[0])
        # predicted_depth_map[y[Z_Index], x[Z_Index]] = Z[Z_Index]
        # predicted_depth_map[0:5, :] = 0.0
        # predicted_depth_map[:, 0:5] = 0.0
        # predicted_depth_map[predicted_depth_map.shape[0] - 5:, :] = 0.0
        # predicted_depth_map[:, predicted_depth_map.shape[1] - 5:] = 0.0
        # smc.imsave(cf.paths["training_img_result_path"] + "/epoch_" + str(epoch) + "_predicted.png",predicted_depth_map.detach().cpu().numpy().astype(np.uint8))
        # smc.imsave(cf.paths["training_img_result_path"] + "/epoch_" + str(epoch) + "_target.png",(gt_depth_map[0] * 40.0 + 40.0).detach().cpu().numpy().astype(np.uint8))

        if i_batch % cf.network_info['freq_print'] == 0:
            print('Epoch: [{0}][{1}/{2}] \t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) \t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) \t'
                  'Total Loss {loss.val:.4f} ({loss.avg:.4f}) \t'
                  'Transformation Loss {trans_loss.val:.4f} ({trans_loss.avg:.4f}) \t'
                  'Depth Map Loss {depth_loss.val:.4f} ({depth_loss.avg:.4f}) \t'
                  'Point Cloud Loss {point_loss.val:.4f} ({point_loss.avg:.4f}) \t'.format(epoch + 1, i_batch,
                                                                                           len(data_loader),
                                                                                           batch_time=batch_time,
                                                                                           data_time=data_time,
                                                                                           loss=losses,
                                                                                           trans_loss=trans_loss,
                                                                                           depth_loss=depth_loss,
                                                                                           point_loss=point_loss))

    scheduler.step()
    valid_batch_time = AverageMeter()
    valid_data_time = AverageMeter()
    valid_losses = AverageMeter()

    model.eval()

    end = time.time()
    for i_batch, sample_bathced in enumerate(valid_loader):
        valid_data_time.update(time.time() - end)

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

        optimizer.zero_grad()

        rotation, translation = model(source_image, source_depth_map)

        loss = loss_function(point_cloud, translation_vector, rotation_vector,
                             translation, rotation, transform_matrix, K_final, target_depth_map)

        valid_losses.update(loss['total_loss'].item(), source_depth_map.size(0))

        valid_batch_time.update(time.time() - end)
        end = time.time()

        # Save Predicted Depth Map
        source_map = source_depth_map[0]
        gt_depth_map = target_depth_map[0]
        point_clouds = point_cloud[0]
        predicted_translation_vector = translation[0]
        predicted_rotation_vector = rotation[0]
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

        # High Speed ( 2021. 11. 25. )
        Z_Index = torch.where(Z > 0)
        predicted_depth_map = torch.zeros_like(gt_depth_map[0])
        predicted_depth_map[y[Z_Index], x[Z_Index]] = Z[Z_Index]
        predicted_depth_map[0:5, :] = 0.0
        predicted_depth_map[:, 0:5] = 0.0
        predicted_depth_map[predicted_depth_map.shape[0] - 5:, :] = 0.0
        predicted_depth_map[:, predicted_depth_map.shape[1] - 5:] = 0.0
        smc.imsave(cf.paths["validation_img_result_path"] + "/epoch_" + str(epoch) + "_predicted.png", predicted_depth_map.detach().cpu().numpy().astype(np.uint8))
        smc.imsave(
            cf.paths["validation_img_result_path"] + "/epoch_" + str(epoch) + "_target.png",
            (gt_depth_map[0] * 40.0 + 40.0).detach().cpu().numpy().astype(np.uint8))

        if i_batch % cf.network_info['freq_print'] == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i_batch, len(valid_loader),
                                                                  batch_time=valid_batch_time,
                                                                  data_time=valid_data_time, loss=valid_losses))



    save_checkpoint({
        'epoch': epoch + 1,
        'arch': model.get_name(),
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()}, False, os.path.join(pretrained_path, model.get_name()), 'pth')

print("Train Finished!!")
