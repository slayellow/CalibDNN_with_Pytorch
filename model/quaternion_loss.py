import numpy as np
import torch


def quatmultiply(q, r, device='cpu'):
    if isinstance(q, torch.Tensor):
        t = torch.zeros(q.shape[0], 4, device=device)
    elif isinstance(q, np.ndarray):
        t = np.zeros((q.shape[0], 4))
    else:
        raise TypeError('Type not supported')
    t[:, 0] = r[:, 0] * q[:, 0] - r[:, 1] * q[:, 1] - r[:, 2] * q[:, 2] - r[:, 3] * q[:, 3]
    t[:, 1] = r[:, 0] * q[:, 1] + r[:, 1] * q[:, 0] - r[:, 2] * q[:, 3] + r[:, 3] * q[:, 2]
    t[:, 2] = r[:, 0] * q[:, 2] + r[:, 1] * q[:, 3] + r[:, 2] * q[:, 0] - r[:, 3] * q[:, 1]
    t[:, 3] = r[:, 0] * q[:, 3] - r[:, 1] * q[:, 2] + r[:, 2] * q[:, 1] + r[:, 3] * q[:, 0]
    return t


def quatinv(q):
    if isinstance(q, torch.Tensor):
        t = q.clone()
    elif isinstance(q, np.ndarray):
        t = q.copy()
    else:
        raise TypeError('Type not supported')
    t *= -1
    t[:, 0] *= -1
    return t


def quaternion_distance(q, r, device):
    t = quatmultiply(q, quatinv(r), device)
    return 2 * torch.atan2(torch.norm(t[:, 1:], dim=1), torch.abs(t[:, 0]))


def quaternion_distance_numpy(q, r):
    t = quatmultiply(q, quatinv(r), 'cpu')
    return 2 * np.arctan2(np.linalg.norm(t[:, 1:], axis=1), np.abs(t[:, 0]))

