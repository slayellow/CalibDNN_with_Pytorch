from utils.mathutils_func import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import imageio as smc
import torch
import cv2
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
        if torch.is_tensor(idx):
            idx = idx.tolist()

        max_rot = self.max_rotation
        max_tr = self.max_translation

        omega_x = np.random.uniform(-max_rot, max_rot)
        omega_y = np.random.uniform(-max_rot, max_rot)
        omega_z = np.random.uniform(-max_rot, max_rot)
        tr_x = np.random.uniform(-max_tr, max_tr)
        tr_y = np.random.uniform(-max_tr, max_tr)
        tr_z = np.random.uniform(-max_tr, max_tr)

        r_org = mathutils.Euler((omega_x * (3.141592 / 180.0), omega_y * (3.141592 / 180.0), omega_z * (3.141592 / 180.0)))
        t_org = mathutils.Vector((tr_x, tr_y, tr_z))

        R = r_org.to_matrix()
        R.resize_4x4()
        T = mathutils.Matrix.Translation(t_org)
        RT = T @ R

        random_transform = np.array(RT)

        source_img = np.float32(cv2.imread(self.source_image[idx], flags=cv2.IMREAD_COLOR))
        source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)

        # 2021.12.29. Add
        image_to_tensor = transforms.ToTensor()
        image_normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if self.training:
            color_transform = transforms.ColorJitter(0.1, 0.1, 0.1)
            source_img = color_transform(source_img)
        source_img = image_to_tensor(source_img)
        source_img = image_normalization(source_img)

        points = np.fromfile(self.point_cloud[idx], sep=' ')
        points = points.reshape((-1, 4))

        valid_indices = points[:, 0] < -3.
        valid_indices = valid_indices | (points[:, 0] > 3.)
        valid_indices = valid_indices | (points[:, 1] < -3.)
        valid_indices = valid_indices | (points[:, 1] > 3.)
        points = points[valid_indices]

        # points = points[:90000, :3]
        # if points.shape[0] < 90000:
        #     points = np.vstack((points, np.zeros((90000 - points.shape[0], 3))))
        ones_col = np.ones(shape=(points.shape[0], 1))
        points = np.hstack((points, ones_col)).astype(np.float32)

        points_in_cam_axis = np.matmul(cf.KITTI_Info["R_rect_00"], (np.matmul(cf.KITTI_Info["velo_to_cam"], points.T)))
        transformed_points = np.matmul(random_transform, points_in_cam_axis)
        points_2d = np.matmul(cf.KITTI_Info["K"],
                              np.matmul(cf.KITTI_Info["cam_02_transform"], transformed_points)[:-1, :])

        Z = points_2d[2, :]
        x = (points_2d[0, :] / Z + 1e-10).T
        y = (points_2d[1, :] / Z + 1e-10).T
        mask = (x > 0) & (x < cf.KITTI_Info["WIDTH"]) & (y > 0) & (y < cf.KITTI_Info["HEIGHT"]) & (Z > 0)
        x = x[mask]
        y = y[mask]
        Z = Z[mask]
        source_map = np.zeros((cf.KITTI_Info["HEIGHT"], cf.KITTI_Info["WIDTH"]))
        source_map[y, x] = Z
        # source_map = np.repeat(np.expand_dims(source_map, axis=2), 3, axis=2)
        source_map = (source_map - 40.0) / 40.0
        source_map = np.float32(source_map)

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
        smc.imsave("depth_map.png", source_map)

        transformed = np.linalg.inv(random_transform)  # Ground Truth RT Matrix
        rotation, translation = convert_RTMatrix_to_6DoF(transformed)

        data = {'source_depth_map': self.transform(source_map), 'source_image': source_img,
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
