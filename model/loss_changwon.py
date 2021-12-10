import torch
from torch import nn as nn
import utils.config as cf
from model.quaternion_loss import *
from utils.mathutils_func import *
import time


class TotalLoss(nn.Module):
    def __init__(self, rotation_weight=1.0, translation_weight=2.0, point_cloud_loss_weight=0.5):
        super(TotalLoss, self).__init__()
        self.rotation_weight = rotation_weight
        self.translation_weight = translation_weight
        self.translation_loss = nn.MSELoss()
        self.point_cloud_loss_weight = point_cloud_loss_weight
        self.loss = {}

    def forward(self, point_clouds, gt_translation_vector, gt_rotation_vector,
                predicted_translation_vector, predicted_rotation_vector, gt_rt_matrix):

        # Transformation Loss
        loss_translation = self.translation_loss(predicted_translation_vector, gt_translation_vector)
        loss_rotation = quaternion_distance(predicted_rotation_vector, gt_rotation_vector,
                                            predicted_rotation_vector.device).mean()
        transformation_loss = (loss_translation * self.translation_weight) + (loss_rotation * self.rotation_weight)

        # PointCloud Loss
        point_clouds_loss = torch.tensor([0.0], dtype=torch.float32).to(predicted_rotation_vector.device)
        for i in range(len(point_clouds)):
            point_cloud_gt = point_clouds[i][0].to(predicted_rotation_vector.device)
            point_cloud_out = point_clouds[i][0].clone()

            R_predicted = quat2mat(predicted_rotation_vector[i])
            T_predicted = tvector2mat(predicted_translation_vector[i])
            RT_predicted = torch.mm(T_predicted, R_predicted)

            RT_Total = torch.mm(gt_rt_matrix[i].inverse(), RT_predicted)

            point_cloud_out = torch.mm(point_cloud_out, RT_Total)
            error = (point_cloud_out - point_cloud_gt).norm(dim=0)
            error.clamp(100.)
            point_clouds_loss += error.mean()

        total_loss = (1 - self.point_cloud_loss_weight) * transformation_loss + \
                     self.point_cloud_loss_weight * (point_clouds_loss / gt_translation_vector.shape[0])
        self.loss['total_loss'] = total_loss
        self.loss['transformation_loss'] = transformation_loss
        self.loss['point_clouds_loss'] = point_clouds_loss / gt_translation_vector.shape[0]

        return self.loss
