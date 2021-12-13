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
model = CalibDNN18(18, pretrained=os.path.join(pretrained_path, "CalibDNN_18_KITTI" + '.pth')).to(devices)
print("------------ Model Summary ----------------")
summary(model, [(1, 3, 375, 1242), (1, 3, 375, 1242)], devices)
print("------------ Model Setting Finish ----------------")
print("")
K_final = torch.tensor(cf.K, dtype=torch.float32).to(devices)
print("------------ Loss Function Setting Start ----------------")
loss_function = TotalLoss(rotation_weight=cf.network_info['rotation_weight'],
                          translation_weight=cf.network_info['translation_weight'],
                          point_cloud_loss_weight=cf.network_info['point_cloud_loss_weight']).to(devices)
print("------------ Loss Function Setting Finish ----------------")
learning_rate = cf.network_info['learning_rate']

optimizer = set_Adam(model, learning_rate=learning_rate)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40, 50, 60, 70, 80, 90], gamma=0.1)


if os.path.isfile(os.path.join(pretrained_path, "CalibDNN_18_KITTI" + '.pth')):
    print("Pretrained Model Open : ", model.get_name() + ".pth")
    checkpoint = load_weight_file(os.path.join(pretrained_path, "CalibDNN_18_KITTI" + '.pth'))
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
    point_loss = AverageMeter()

    model.train()

    end = time.time()

    for i_batch, sample_bathced in enumerate(data_loader):
        data_time.update(time.time() - end)

        source_depth_map = sample_bathced['source_depth_map']
        source_image = sample_bathced['source_image']
        target_depth_map = sample_bathced['target_depth_map']
        point_cloud = sample_bathced['point_cloud']
        rotation_vector = sample_bathced['rotation_vector'].to(torch.float32)
        translation_vector = sample_bathced['translation_vector'].to(torch.float32)
        transform_matrix = sample_bathced['transform_matrix'].to(torch.float32)

        if gpu_check:
            source_depth_map = source_depth_map.to(devices)
            source_image = source_image.to(devices)
            target_depth_map = target_depth_map.to(devices)
            point_cloud = point_cloud.to(devices)
            rotation_vector = rotation_vector.to(devices)
            translation_vector = translation_vector.to(devices)
            transform_matrix = transform_matrix.to(devices)

        optimizer.zero_grad()

        rotation, translation = model(source_image, source_depth_map)

        loss = loss_function(point_cloud, translation_vector, rotation_vector,
                             translation, rotation, transform_matrix)

        if not torch.isfinite(loss['total_loss']):
            print("Loss Function --> Non-Finite Loss, Don't Calculate Loss")
            continue

        loss['total_loss'].backward()
        optimizer.step()

        losses.update(loss['total_loss'].item(), source_depth_map.size(0))
        trans_loss.update(loss['transformation_loss'].item(), source_depth_map.size(0))
        point_loss.update(loss['point_clouds_loss'].item(), source_depth_map.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i_batch % cf.network_info['freq_print'] == 0:
            print('Epoch: [{0}][{1}/{2}] \t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) \t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) \t'
                  'Total Loss {loss.val:.4f} ({loss.avg:.4f}) \t'
                  'Transformation Loss {trans_loss.val:.4f} ({trans_loss.avg:.4f}) \t'
                  'Point Cloud Loss {point_loss.val:.4f} ({point_loss.avg:.4f}) \t'.format(epoch + 1, i_batch,
                                                                                           len(data_loader),
                                                                                           batch_time=batch_time,
                                                                                           data_time=data_time,
                                                                                           loss=losses,
                                                                                           trans_loss=trans_loss,
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
        point_cloud = sample_bathced['point_cloud']
        rotation_vector = sample_bathced['rotation_vector'].to(torch.float32)
        translation_vector = sample_bathced['translation_vector'].to(torch.float32)
        transform_matrix = sample_bathced['transform_matrix'].to(torch.float32)

        if gpu_check:
            source_depth_map = source_depth_map.to(devices)
            source_image = source_image.to(devices)
            target_depth_map = target_depth_map.to(devices)
            point_cloud = point_cloud.to(devices)
            rotation_vector = rotation_vector.to(devices)
            translation_vector = translation_vector.to(devices)
            transform_matrix = transform_matrix.to(devices)

        optimizer.zero_grad()

        rotation, translation = model(source_image, source_depth_map)

        loss = loss_function(point_cloud, translation_vector, rotation_vector,
                             translation, rotation, transform_matrix)

        valid_losses.update(loss['total_loss'].item(), source_depth_map.size(0))

        valid_batch_time.update(time.time() - end)
        end = time.time()

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
        'optimizer': optimizer.state_dict()}, True, os.path.join(pretrained_path, "CalibDNN_18_KITTI"))

print("Train Finished!!")
