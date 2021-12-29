import numpy as np


dataprocessing = dict(
    dataset_path = "/data/2011_09_26/",
    dataset_changwon_path = "/data/2021_12_02/train/"
)


paths = dict(
    dataset_changwon_path = "/home/HONG/CalibDNN_with_Pytorch/dataset/Changwon/parsed_set_changwon.txt",
    dataset_path = "/home/HONG/CalibDNN_with_Pytorch/dataset/KITTI/parsed_set.txt",
    pretrained_path = "/home/HONG/PretrainedParameter/CalibDNN/",
    training_img_result_path = "/home/HONG/CalibDNN_Result/training",
    validation_img_result_path = "/home/HONG/CalibDNN_Result/valid",
    inference_img_result_path = "/home/HONG/CalibDNN_Result/inference"
)

# 카메라 관련 파라메타
KITTI_velo_to_cam_R = np.array(
    [7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04, -9.998902e-01, 9.998621e-01, 7.523790e-03,
     1.480755e-02]).reshape(3, 3)
KITTI_velo_to_cam_T = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01]).reshape(3, 1)

KITTI_Info = dict(
    WIDTH = 1242,
    HEIGHT = 375,

    fx = 7.215377e+02,
    fy = 7.215377e+02,
    cx = 6.095593e+02,
    cy = 1.728540e+02,

    K = np.array([7.215377e+02, 0.000000e+00, 6.095593e+02,
              0.000000e+00, 7.215377e+02, 1.728540e+02,
              0.000000e+00, 0.000000e+00, 1.000000e+00]).reshape(3,3),


    velo_to_cam = np.vstack((np.hstack((KITTI_velo_to_cam_R, KITTI_velo_to_cam_T)), np.array([[0,0,0,1]]))),

    R_rect_00 =  np.array([9.999239e-01, 9.837760e-03, -7.445048e-03, 0.0,
                      -9.869795e-03, 9.999421e-01, -4.278459e-03, 0.0,
                       7.402527e-03, 4.351614e-03, 9.999631e-01,  0.0,
                       0.0,          0.0,          0.0,           1.0]).reshape(4,4),

    cam_02_transform = np.array([1.0, 0.0, 0.0, 4.485728e+01/7.215377e+02,
                             0.0, 1.0, 0.0, 2.163791e-01/7.215377e+02,
                             0.0, 0.0, 1.0, 2.745884e-03,
                             0.0, 0.0, 0.0, 1.0]).reshape(4,4),
    save_checkpoint_name = "CalibDNN_KITTI_ROT1.0_TR0.1"

)

Changwon_Info = dict(
    WIDTH_CHANGWON = 640,
    HEIGHT_CHANGWON = 480,
    K_changwon = np.array([384.8557, 0, 328.4401,
              0, 345.4014, 245.6107,
              0, 0, 1]).reshape(3, 3),

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

network_info = dict(
    rotation_weight=1.0,
    translation_weight=2.0,
    point_cloud_loss_weight=0.5,
    rotation_range = 20.0,             # dataset random transformation rotation range ( 20.0, 10.0, 5.0, 2.0, 1.0 )
    translation_range = 2.0,           # dataset random transformation translation range ( 2.0, 1.0, 0.5, 0.2, 0.1 )
    batch_size = 2,                        # batch_size take during training
    epochs = 200,                            # total number of epoch
    learning_rate = 4e-4,                   # learining rate        1e-4
    beta1 = 0.9,                            # momentum term for Adam Optimizer
    freq_print = 10,
    num_worker = 0,
    learning_scheduler=[50, 100, 150]
)

inference_info = dict(
    weights = ['/home/HONG/PretrainedParameter/CalibDNN/CalibDNN_KITTI_ROT20_TR2.pth',
               '/home/HONG/PretrainedParameter/CalibDNN/CalibDNN_KITTI_ROT10_TR1.pth',
               '/home/HONG/PretrainedParameter/CalibDNN/CalibDNN_KITTI_ROT5.0_TR0.5.pth',
               '/home/HONG/PretrainedParameter/CalibDNN/CalibDNN_KITTI_ROT2.0_TR0.2.pth',
               '/home/HONG/PretrainedParameter/CalibDNN/CalibDNN_KITTI_ROT1.0_TR0.1.pth'],
    freq_print = 10,
    num_worker = 4,
    batch_size = 1,
    rotation_range=20.0,  # dataset random transformation rotation range ( 20.0, 10.0, 5.0, 2.0, 1.0 )
    translation_range=2.0,  # dataset random transformation translation range ( 2.0, 1.0, 0.5, 0.2, 0.1 )
)