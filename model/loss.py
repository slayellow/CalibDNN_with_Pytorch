import torch
import torchgeometry as tgm
from torch import nn as nn
import utils.config as cf
from utils.mathutils_func import *
import imageio as smc


class TotalLoss(nn.Module):
    def __init__(self, alpha):
        super(TotalLoss, self).__init__()
        self.alpha = alpha
        self.translation_loss = nn.MSELoss()
        self.rotation_loss = nn.MSELoss()
        self.depth_map_loss = nn.MSELoss()
        self.loss = {}

    def forward(self, point_clouds, gt_translation_vector, gt_rotation_vector,
                rtvec, gt_rt_matrix, transformation_weight, point_cloud_weight):
        # Transformation Loss
        loss_translation = self.translation_loss(rtvec[:, 3:], gt_translation_vector)
        loss_rotation = self.rotation_loss(rtvec[:, :3], gt_rotation_vector)
        transformation_loss = loss_translation + (loss_rotation * self.alpha)

        RT_predicted = tgm.rtvec_to_pose(rtvec)
        # PointCloud Loss
        point_clouds_loss = torch.tensor([0.0], dtype=torch.float32).to(rtvec.device)

        for i in range(len(point_clouds)):
            point_cloud_gt = point_clouds[i].to(rtvec.device)
            point_cloud_out = point_clouds[i].clone()

            RT = RT_predicted[i]

            RT_Total = torch.mm(gt_rt_matrix[i].inverse(), RT)

            point_cloud_out = torch.mm(point_cloud_out, RT_Total)

            error = (point_cloud_out - point_cloud_gt).norm(dim=0)
            error.clamp(100.)
            point_clouds_loss += error.mean()

        total_loss = transformation_loss + transformation_weight * (point_clouds_loss / gt_translation_vector.shape[0])
        self.loss['total_loss'] = total_loss
        self.loss['transformation_loss'] = transformation_loss
        self.loss['point_clouds_loss'] = point_clouds_loss / gt_translation_vector.shape[0]

        return self.loss
