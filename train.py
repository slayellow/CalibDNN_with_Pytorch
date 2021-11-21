import torch.autograd

from dataset.dataset import *
from model.CalibDNN import *
from model.loss import *
from utils.AverageMeter import *


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False

gpu_check = is_gpu_avaliable()
devices = torch.device("cuda") if gpu_check else torch.device("cpu")

# dataset test code
trainingset = CalibDNNDataset(cf.paths['dataset_path'], training=True)
data_loader = get_loader(trainingset, batch_size=cf.network_info['batch_size'], shuffle=True, num_worker=cf.network_info['num_worker'])

validationset = CalibDNNDataset(cf.paths['dataset_path'], training=False)
valid_loader = get_loader(validationset, batch_size=cf.network_info['batch_size'], shuffle=False, num_worker=cf.network_info['num_worker'])

# model test code
model = CalibDNN18(18).to(devices)
summary(model, [(1, 3, 375, 1242), (1, 3, 375, 1242)], devices)

K_final = torch.tensor(cf.camera_intrinsic_parameter, dtype=torch.float32).to(devices)

loss_function = TotalLoss().to(devices)

learning_rate = cf.network_info['learning_rate']

optimizer = set_Adam(model, learning_rate=learning_rate)

pretrained_path = cf.paths['pretrained_path']
if os.path.isfile(os.path.join(pretrained_path, model.get_name() + '.pth')):
    print("Pretrained Model Open : ", model.get_name() + ".pth")
    checkpoint = load_weight_file(os.path.join(pretrained_path, model.get_name() + '.pth'))
    start_epoch = checkpoint['epoch']
    load_weight_parameter(model, checkpoint['state_dict'])
    load_weight_parameter(optimizer, checkpoint['optimizer'])
else:
    print("No Pretrained Model")
    start_epoch = 0

for epoch in range(start_epoch, cf.network_info['epochs']):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    if is_gpu_avaliable():
        torch.cuda.empty_cache()  # 사용하지 않으면서 캐시된 메모리들을 해제해준다.

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

        losses.update(loss['total_loss'].item())
        batch_time.update(time.time() - end)
        end = time.time()

        if i_batch % cf.network_info['freq_print'] == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch+1, i_batch, len(data_loader),
                                                        batch_time=batch_time, data_time=data_time, loss=losses))

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
        rotation_vector = sample_bathced['rotation_vector']
        translation_vector = sample_bathced['translation_vector']
        transform_matrix = sample_bathced['transform_matrix']

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

        valid_losses.update(loss.item(), source_depth_map.size(0))

        valid_batch_time.update(time.time() - end)
        end = time.time()

        if i_batch % cf.network_info['freq_print'] == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i_batch, len(valid_loader),
                                                        batch_time=batch_time, data_time=data_time, loss=losses))

    save_checkpoint({
        'epoch': epoch + 1,
        'arch' : model.get_name(),
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()}, False, os.path.join(pretrained_path, model.get_name()),'pth')

    # Learning Rate 조절하기
    lr = learning_rate - 0.00001  # ResNet Lerarning Rate
    # lr = self.learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
print("Train Finished!!")
