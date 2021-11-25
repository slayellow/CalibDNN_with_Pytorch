import torch
from torch import nn as nn
import utils.config as cf
from utils.mathutils_func import *
import time


class TotalLoss(nn.Module):
    def __init__(self, rotation_weight=1.0, translation_weight=2.0, depth_map_loss_weight=1.0,
                 point_cloud_loss_weight=0.5):
        super(TotalLoss, self).__init__()
        self.rotation_weight = rotation_weight
        self.translation_weight = translation_weight
        self.depth_map_loss_weight = depth_map_loss_weight
        self.translation_loss = nn.MSELoss(reduction='none')
        self.rotation_loss = nn.MSELoss(reduction='none')
        self.point_cloud_loss_weight = point_cloud_loss_weight
        self.loss = {}

    def forward(self, point_clouds, gt_translation_vector, gt_rotation_vector,
                predicted_translation_vector, predicted_rotation_vector, gt_rt_matrix, k_matrix, gt_depth_map):

        # Transformation Loss
        loss_translation = self.translation_loss(predicted_translation_vector, gt_translation_vector).sum(1).mean()
        loss_rotation = self.rotation_loss(predicted_rotation_vector, gt_rotation_vector).sum(1).mean()
        transformation_loss = (loss_translation * self.translation_weight) + (loss_rotation * self.rotation_weight)

        # Depth Map Loss
        depth_map_loss = torch.tensor([0.0], dtype=torch.float32).to(predicted_rotation_vector.device)

        # PointCloud Loss
        point_clouds_loss = torch.tensor([0.0], dtype=torch.float32).to(predicted_rotation_vector.device)
        for i in range(len(point_clouds)):
            point_cloud_gt = torch.mm(gt_rt_matrix[i], point_clouds[i][0].t()).to(predicted_rotation_vector.device)

            theta = torch.sqrt(predicted_rotation_vector[i][0] ** 2 +
                               predicted_rotation_vector[i][1] ** 2 +
                               predicted_rotation_vector[i][2] ** 2).to(predicted_rotation_vector.device)
            omega_cross = torch.tensor([[0.0, -predicted_rotation_vector[i][2],
                                         predicted_rotation_vector[i][1]],
                                        [predicted_rotation_vector[i][2], 0.0,
                                         -predicted_rotation_vector[i][0]],
                                        [-predicted_rotation_vector[i][1],
                                         predicted_rotation_vector[i][0], 0.0]]).to(predicted_rotation_vector.device)
            A = torch.sin(theta) / theta
            B = (1.0 - torch.cos(theta)) / (theta ** 2)
            R = torch.eye(3, 3).to(predicted_rotation_vector.device) + A * omega_cross + B * \
                torch.mm(omega_cross, omega_cross).to(predicted_rotation_vector.device)
            T = torch.tensor([[predicted_translation_vector[i][0]],
                              [predicted_translation_vector[i][1]],
                              [predicted_translation_vector[i][2]]]).to(predicted_rotation_vector.device)
            predicted_rt_matrix = torch.vstack((torch.hstack((R, T)),
                                                torch.tensor([[0.0, 0.0, 0.0, 1.0]]).to(
                                                    predicted_rotation_vector.device)))

            point_cloud_predicted = torch.mm(predicted_rt_matrix, point_clouds[i][0].t())

            error = (point_cloud_predicted - point_cloud_gt).norm(dim=0)
            error.clamp(100.)
            point_clouds_loss += error.mean()

            points_2d_predicted = torch.mm(k_matrix, point_cloud_predicted[:-1, :])
            Z = points_2d_predicted[2, :]
            x = (points_2d_predicted[0, :] / Z).t()
            y = (points_2d_predicted[1, :] / Z).t()

            x = torch.clamp(x, 0.0, cf.camera_info["WIDTH"] - 1).to(torch.long)
            y = torch.clamp(y, 0.0, cf.camera_info['HEIGHT'] - 1).to(torch.long)

            # High Speed ( 2021. 11. 25. )
            Z_Index = torch.where(Z > 0)
            predicted_depth_map = torch.zeros_like(gt_depth_map[i][0])
            predicted_depth_map[y[Z_Index], x[Z_Index]] = Z[Z_Index]

            # Low Speed ( Modefied Need )
            # predicted_depth_map = torch.zeros_like(gt_depth_map[i][0])
            # for x_idx, y_idx, z_idx in zip(x, y, Z):
            #     if z_idx > 0:
            #         predicted_depth_map[int(y_idx), int(x_idx)] = z_idx

            depth_error = (predicted_depth_map - gt_depth_map[i][0]).norm(dim=0)
            depth_error.clamp(100.)
            depth_map_loss += depth_error.mean()
        total_loss = (1 - self.point_cloud_loss_weight) * transformation_loss + \
                     self.depth_map_loss_weight * (depth_map_loss / gt_translation_vector.shape[0]) + \
                     self.point_cloud_loss_weight * (point_clouds_loss / gt_translation_vector.shape[0])
        self.loss['total_loss'] = total_loss
        self.loss['transformation_loss'] = transformation_loss
        self.loss['depth_map_loss'] = depth_map_loss / gt_translation_vector.shape[0]
        self.loss['point_clouds_loss'] = point_clouds_loss / gt_translation_vector.shape[0]

        return self.loss
