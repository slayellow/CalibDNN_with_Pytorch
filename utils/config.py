import numpy as np


dataprocessing = dict(
    dataset_path = "/data/2011_09_26/",
    dataset_changwon_path = "/data/2021_12_02/train/"
)


paths = dict(
    dataset_changwon_path = "/home/HONG/CalibDNN_with_Pytorch/dataset/Changwon/parsed_set_changwon.txt",
    dataset_path = "/home/HONG/CalibDNN_with_Pytorch/dataset/parsed_set.txt",
    pretrained_path = "/home/HONG/PretrainedParameter/CalibDNN/",
    training_img_result_path = "/home/HONG/CalibDNN_Result/training",
    validation_img_result_path = "/home/HONG/CalibDNN_Result/valid",
    inference_img_result_path = "/home/HONG/CalibDNN_Result/inference"
)

# 카메라 관련 파라메타

camera_info = dict(
    WIDTH = 1242,
    HEIGHT = 375,
    WIDTH_CHANGWON = 640,
    HEIGHT_CHANGWON = 480,

    fx = 7.215377e+02,
    fy = 7.215377e+02,
    cx = 6.095593e+02,
    cy = 1.728540e+02,

    cam_transform_02 = np.array([1.0, 0.0, 0.0, (-4.485728e+01)/7.215377e+02,
                                 0.0, 1.0, 0.0, (-2.163791e-01)/7.215377e+02,
                                 0.0, 0.0, 1.0, (-2.745884e-03),
                                 0.0, 0.0, 0.0, 1.0]).reshape(4, 4),

    cam_transform_02_inv = np.array([1.0, 0.0, 0.0, (4.485728e+01)/7.215377e+02,
                                     0.0, 1.0, 0.0, (2.163791e-01)/7.215377e+02,
                                     0.0, 0.0, 1.0, (2.745884e-03),
                                     0.0, 0.0, 0.0, 1.0]).reshape(4, 4)
)

fx_scaled = 2 * camera_info['fx'] / np.float32(camera_info['WIDTH'])
fy_scaled = 2 * camera_info['fy'] / np.float32(camera_info['HEIGHT'])
cx_scaled = -1 * 2 * (camera_info['cx'] - 1.0) / np.float32(camera_info['WIDTH'])
cy_scaled = -1 * 2 * (camera_info['cy'] - 1.0) / np.float32(camera_info['HEIGHT'])

camera_intrinsic_parameter = np.array([[fx_scaled, 0.0, cx_scaled],
                                       [0.0, fy_scaled, cy_scaled],
                                       [0.0, 0.0, 1.0]], dtype= np.float32)

K = np.array([7.215377e+02, 0.000000e+00, 6.095593e+02, 0.000000e+00, 7.215377e+02, 1.728540e+02, 0.000000e+00, 0.000000e+00, 1.000000e+00]).reshape(3,3)
K_changwon = np.array([384.8557, 0, 328.4401,
              0, 345.4014, 245.6107,
              0, 0, 1]).reshape(3, 3)

# 네트워크 구성 관련 파라메타

network_info = dict(
    rotation_weight=1.0,
    translation_weight=2.0,
    depth_map_loss_weight=1.0,
    point_cloud_loss_weight=0.5,
    batch_size = 24,                        # batch_size take during training
    epochs = 60,                            # total number of epoch
    learning_rate = 3e-4,                   # learining rate
    beta1 = 0.9,                            # momentum term for Adam Optimizer
    freq_print = 10,
    num_worker = 4
)
