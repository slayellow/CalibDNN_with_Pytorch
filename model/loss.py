import torch
from torch import nn as nn
import utils.config as cf
from model.quaternion_loss import *
from utils.mathutils_func import *
import time


class TotalLoss(nn.Module):
    def __init__(self, rotation_weight=1.0, translation_weight=2.0, depth_map_loss_weight=1.0,
                 point_cloud_loss_weight=0.5):
        super(TotalLoss, self).__init__()
        self.rotation_weight = rotation_weight
        self.translation_weight = translation_weight
        self.depth_map_loss_weight = depth_map_loss_weight
        self.translation_loss = nn.MSELoss()
        self.point_cloud_loss_weight = point_cloud_loss_weight
        self.depth_map_loss = nn.MSELoss()
        self.loss = {}

    def forward(self, point_clouds, gt_translation_vector, gt_rotation_vector,
                predicted_translation_vector, predicted_rotation_vector, gt_rt_matrix, k_matrix, gt_depth_map):

        # Transformation Loss
        loss_translation = self.translation_loss(predicted_translation_vector, gt_translation_vector)
        loss_rotation = quaternion_distance(predicted_rotation_vector, gt_rotation_vector,
                                            predicted_rotation_vector.device).mean()
        transformation_loss = (loss_translation * self.translation_weight) + (loss_rotation * self.rotation_weight)

        # Depth Map Loss
        depth_map_loss = torch.tensor([0.0], dtype=torch.float32).to(predicted_rotation_vector.device)

        # PointCloud Loss
        point_clouds_loss = torch.tensor([0.0], dtype=torch.float32).to(predicted_rotation_vector.device)
        for i in range(len(point_clouds)):
            point_cloud_gt = torch.mm(gt_rt_matrix[i], point_clouds[i][0].t()).to(predicted_rotation_vector.device)

            R_predicted = quat2mat(predicted_rotation_vector[i])
            T_predicted = tvector2mat(predicted_translation_vector[i])
            RT_predicted = torch.mm(T_predicted, R_predicted)

            point_cloud_out = torch.mm(RT_predicted, point_clouds[i][0].t())
            error = (point_cloud_out - point_cloud_gt).norm(dim=0)
            error.clamp(100.)
            point_clouds_loss += error.mean()

            points_2d_predicted = torch.mm(k_matrix, point_cloud_out[:-1, :])
            Z = points_2d_predicted[2, :]
            x = (points_2d_predicted[0, :] / Z).t()
            y = (points_2d_predicted[1, :] / Z).t()

            x = torch.clamp(x, 0.0, cf.camera_info["WIDTH"] - 1).to(torch.long)
            y = torch.clamp(y, 0.0, cf.camera_info['HEIGHT'] - 1).to(torch.long)

            # High Speed ( 2021. 11. 25. )
            Z_Index = torch.where(Z > 0)
            predicted_depth_map = torch.zeros_like(gt_depth_map[i][0])
            predicted_depth_map[y[Z_Index], x[Z_Index]] = Z[Z_Index]
            predicted_depth_map[0:5, :] = 0.0
            predicted_depth_map[:, 0:5] = 0.0
            predicted_depth_map[predicted_depth_map.shape[0] - 5:, :] = 0.0
            predicted_depth_map[:, predicted_depth_map.shape[1] - 5:] = 0.0
            predicted_depth_map = (predicted_depth_map - 40.0) / 40.0

            depth_error = self.depth_map_loss(predicted_depth_map, gt_depth_map[i][0])
            depth_map_loss += depth_error
        total_loss = (1 - self.point_cloud_loss_weight) * transformation_loss + \
                     self.depth_map_loss_weight * (depth_map_loss / gt_translation_vector.shape[0]) + \
                     self.point_cloud_loss_weight * (point_clouds_loss / gt_translation_vector.shape[0])
        self.loss['total_loss'] = total_loss
        self.loss['transformation_loss'] = transformation_loss
        self.loss['depth_map_loss'] = depth_map_loss / gt_translation_vector.shape[0]
        self.loss['point_clouds_loss'] = point_clouds_loss / gt_translation_vector.shape[0]

        return self.loss
