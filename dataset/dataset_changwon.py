import numpy as np

from utils.mathutils_func import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
import cv2
import imageio as smc


K = np.array([384.8557, 0, 328.4401,
              0, 345.4014, 245.6107,
              0, 0, 1]).reshape(3, 3)


class CalibDNNDataset_Changwon(Dataset):
    def __init__(self, path, training=True):
        """
        Args:
            training (bool) : training 인지 validation 인지 여부 확
        """
        self.dataset = np.loadtxt(path, dtype = str)
        self.training = training
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.data_num = len(self.dataset) * 0.8

        if self.training:
            self.dataset = self.dataset[:int(self.data_num)]
        else:
            self.dataset = self.dataset[int(self.data_num):]

        self.source_depth_map = self.dataset[:, 0]
        self.target_depth_map = self.dataset[:, 1]
        self.source_image = self.dataset[:, 2]
        self.target_image = self.dataset[:, 3]
        self.point_cloud = self.dataset[:, 4]
        self.transforms = np.float32(self.dataset[:, 5:])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        self.point_cloud_list = np.zeros((0, 4), dtype=np.float32)
        if torch.is_tensor(idx):
            idx = idx.tolist()

        source_map = np.float32(cv2.imread(self.source_depth_map[idx], flags=cv2.IMREAD_GRAYSCALE))
        source_map = np.repeat(np.expand_dims(source_map, axis=2), 3, axis=2)
        source_map[0:5, :] = 0.0
        source_map[:, 0:5] = 0.0
        source_map[source_map.shape[0] - 5:, :] = 0.0
        source_map[:, source_map.shape[1] - 5:] = 0.0
        source_map = (source_map - 127.5) / 127.5

        target_map = np.float32(cv2.imread(self.target_depth_map[idx], flags=cv2.IMREAD_GRAYSCALE))
        target_map[0:5, :] = 0.0
        target_map[:, 0:5] = 0.0
        target_map[target_map.shape[0] - 5:, :] = 0.0
        target_map[:, target_map.shape[1] - 5:] = 0.0
        target_map = (target_map - 127.5) / 127.5

        source_img = np.float32(cv2.imread(self.source_image[idx], flags=cv2.IMREAD_COLOR))
        source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
        source_img[0:5, :] = 0.0
        source_img[:, 0:5] = 0.0
        source_img[source_img.shape[0] - 5:, :] = 0.0
        source_img[:, source_img.shape[1] - 5:] = 0.0
        source_img = (source_img - 127.5) / 127.5

        target_img = np.float32(cv2.imread(self.target_image[idx], flags=cv2.IMREAD_COLOR))
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
        target_img[0:5, :] = 0.0
        target_img[:, 0:5] = 0.0
        target_img[target_img.shape[0] - 5:, :] = 0.0
        target_img[:, target_img.shape[1] - 5:] = 0.0
        target_img = (target_img - 127.5) / 127.5

        points = self.point_cloud[idx]

        points = np.fromfile(self.point_cloud[idx], sep=' ')
        points = points.reshape((-1, 4))
        points = points[:230400, :3]
        if points.shape[0] < 230400:
            points = np.vstack((points, np.zeros((230400-points.shape[0], 3))))
        ones_col = np.ones(shape=(points.shape[0], 1))
        points = np.hstack((points, ones_col)).astype(np.float32)

        ground_truth_matrix = self.transforms[idx].reshape(4, 4)     # Ground Truth RT Matrix
        K_inverse = np.linalg.inv(K)
        transformed_matrix = np.matmul(K_inverse, ground_truth_matrix[:-1])
        transformed_matrix = np.vstack((transformed_matrix, [0.0, 0.0, 0.0, 1.0]))
        rotation, translation = convert_RTMatrix_to_6DoF(transformed_matrix)

        data = {'source_depth_map': self.transform(source_map), 'target_depth_map': self.transform(target_map),
                'source_image': self.transform(source_img), 'target_image': self.transform(target_img),
                'point_cloud': self.transform(points), 'rotation_vector': rotation, 'translation_vector': translation,
                'transform_matrix': ground_truth_matrix}

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



