import numpy as np

from utils.mathutils_func import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
import cv2
import imageio as smc
import utils.config as cf


class CalibDNNDataset(Dataset):
    def __init__(self, path, max_rot, max_tr, training=True):
        """
        Args:
            training (bool) : training 인지 validation 인지 여부 확
        """
        self.max_rotation = max_rot
        self.max_translation = max_tr
        self.dataset = np.loadtxt(path, dtype=str)
        self.training = training
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.data_num = len(self.dataset) * 0.8

        if self.training:
            self.dataset = self.dataset[:int(self.data_num)]
        else:
            self.dataset = self.dataset[int(self.data_num):]

        self.source_image = self.dataset[:, 0]
        self.point_cloud = self.dataset[:, 1]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        self.point_cloud_list = np.zeros((0, 4), dtype=np.float32)
        if torch.is_tensor(idx):
            idx = idx.tolist()

        max_rot = self.max_rotation
        max_tr = self.max_translation
        omega_x = np.random.uniform(-max_rot, max_rot) * (3.141592 / 180.0)
        omega_y = np.random.uniform(-max_rot, max_rot) * (3.141592 / 180.0)
        omega_z = np.random.uniform(-max_rot, max_rot) * (3.141592 / 180.0)
        tr_x = np.random.uniform(-max_tr, max_tr)
        tr_y = np.random.uniform(-max_tr, max_tr)
        tr_z = np.random.uniform(-max_tr, max_tr)

        r_org = mathutils.Euler((omega_x, omega_y, omega_z))
        t_org = mathutils.Vector((tr_x, tr_y, tr_z))

        R = r_org.to_matrix()
        R.resize_4x4()
        T = mathutils.Matrix.Translation(t_org)
        RT = T @ R

        random_transform = np.array(RT)

        source_img = np.float32(cv2.imread(self.source_image[idx], flags=cv2.IMREAD_COLOR))
        source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
        source_img[0:5, :] = 0.0
        source_img[:, 0:5] = 0.0
        source_img[source_img.shape[0] - 5:, :] = 0.0
        source_img[:, source_img.shape[1] - 5:] = 0.0
        source_img = (source_img - 127.5) / 127.5

        points = np.fromfile(self.point_cloud[idx], sep=' ')
        points = points.reshape((-1, 4))
        points = points[:90000, :3]
        if points.shape[0] < 90000:
            points = np.vstack((points, np.zeros((90000 - points.shape[0], 3))))
        ones_col = np.ones(shape=(points.shape[0], 1))
        points = np.hstack((points, ones_col)).astype(np.float32)

        points_in_cam_axis = np.matmul(cf.KITTI_Info["R_rect_00"], (np.matmul(cf.KITTI_Info["velo_to_cam"], points.T)))
        transformed_points = np.matmul(random_transform, points_in_cam_axis)
        points_2d = np.matmul(cf.KITTI_Info["K"],
                              np.matmul(cf.KITTI_Info["cam_02_transform"], transformed_points)[:-1, :])

        Z = points_2d[2, :]
        x = (points_2d[0, :] / Z).T
        y = (points_2d[1, :] / Z).T

        x = np.clip(x, 0.0, cf.KITTI_Info["WIDTH"] - 1).astype(np.int)
        y = np.clip(y, 0.0, cf.KITTI_Info["HEIGHT"] - 1).astype(np.int)

        Z_Index = np.where(Z > 0)[0]
        source_map = np.zeros((cf.KITTI_Info["HEIGHT"], cf.KITTI_Info["WIDTH"]))
        source_map[y[Z_Index], x[Z_Index]] = Z[Z_Index]
        source_map = np.repeat(np.expand_dims(source_map, axis=2), 3, axis=2)
        source_map[0:5, :] = 0.0
        source_map[:, 0:5] = 0.0
        source_map[source_map.shape[0] - 5:, :] = 0.0
        source_map[:, source_map.shape[1] - 5:] = 0.0
        source_map = (source_map - 40.0) / 40.0
        source_map = np.float32(source_map)

        GT_RTMatrix = np.matmul(cf.KITTI_Info["cam_02_transform"], np.matmul(cf.KITTI_Info["R_rect_00"],
                                                                             cf.KITTI_Info["velo_to_cam"]))
        # points_2d = np.matmul(cf.KITTI_Info["K"], np.matmul(GT_RTMatrix, points.T)[:-1, :])
        # Z = points_2d[2, :]
        # x = (points_2d[0, :] / Z).T
        # y = (points_2d[1, :] / Z).T
        #
        # x = np.clip(x, 0.0, cf.KITTI_Info["WIDTH"] - 1).astype(np.int)
        # y = np.clip(y, 0.0, cf.KITTI_Info["HEIGHT"] - 1).astype(np.int)
        #
        # Z_Index = np.where(Z > 0)[0]
        # source_map = np.zeros((cf.KITTI_Info["HEIGHT"], cf.KITTI_Info["WIDTH"]))
        # source_map[y[Z_Index], x[Z_Index]] = Z[Z_Index]
        #
        # smc.imsave("depth_map_target.png", source_map)

        transformed = np.linalg.inv(random_transform)  # Ground Truth RT Matrix
        rotation, translation = convert_RTMatrix_to_6DoF(transformed)

        data = {'source_depth_map': self.transform(source_map), 'source_image': self.transform(source_img),
                'point_cloud': self.transform(points), 'rotation_vector': rotation, 'translation_vector': translation,
                'transform_matrix': transformed}

        return data


def get_loader(dataset, batch_size, shuffle=True, num_worker=0):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_worker,
        pin_memory=True,
        sampler=None
    )
    return dataloader
