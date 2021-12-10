import numpy as np
import mathutils
import numpy as np
import os
import glob
import argparse
from natsort import natsorted as ns
from skimage import io
import imageio as smc

K = np.array([384.8557, 0, 328.4401,
              0, 345.4014, 245.6107,
              0, 0, 1]).reshape(3, 3)

# GT Matrix ( unit : mm ) -> convert meter [* 1000]
# ExtrinsicParameter = np.array([372.987160, 339.659994, -38.710366, -82675.118467,
#                                -11.669059, 208.264253, -368.939818, -171438.409027,
#                                -0.035360, 0.994001, -0.103494, -179.826270,
#                                0.0, 0.0, 0.0, 1.0]).reshape(4, 4)

ExtrinsicParameter_meter = np.array([372.987160, 339.659994, -38.710366, -82.675118467,
                               -11.669059, 208.264253, -368.939818, -171.438409027,
                               -0.035360, 0.994001, -0.103494, -0.179826270,
                                     0.0, 0.0, 0.0, 1.0]).reshape(4, 4)
IMG_HT = 480
IMG_WDT = 640

dataset_path = '/data/2021_12_02/train/'

for path, dirs, files in os.walk(dataset_path):
    for foldername in dirs:
        if not os.path.exists(dataset_path + foldername + "/depth_maps_3"):
            os.makedirs(dataset_path + foldername + "/depth_maps_3")
        if not os.path.exists(dataset_path + foldername + "/depth_maps_transformed_3"):
            os.makedirs(dataset_path + foldername + "/depth_maps_transformed_3")
        if not os.path.exists(dataset_path + foldername + "/target_images_3"):
            os.makedirs(dataset_path + foldername + "/target_images_3")

        point_files = ns(glob.glob(dataset_path + foldername + "/lidars/*.txt"))
        imgs_files = ns(glob.glob(dataset_path + foldername + "/images/*.png"))

        angle_limit = 0.34722965035593395/1.25
        tr_limit = 0.34722965035593395/1.25

        angle_list = np.zeros((1,16), dtype = np.float32)

        for img_name, cloud_name in zip(imgs_files, point_files):

            print(img_name, cloud_name)
            imagefile = img_name.split('/')[6]

            omega_x = angle_limit*np.random.random_sample() - (angle_limit/2.0)
            omega_y = angle_limit*np.random.random_sample() - (angle_limit/2.0)
            omega_z = angle_limit*np.random.random_sample() - (angle_limit/2.0)
            tr_x = tr_limit*np.random.random_sample() - (tr_limit/2.0)
            tr_y = tr_limit*np.random.random_sample() - (tr_limit/2.0)
            tr_z = tr_limit*np.random.random_sample() - (tr_limit/2.0)

            r_org = mathutils.Euler((omega_x, omega_y, omega_z))
            t_org = mathutils.Vector((tr_x, tr_y, tr_z))

            R = r_org.to_matrix()
            R.resize_4x4()
            T = mathutils.Matrix.Translation(t_org)
            RT = T @ R
            random_transform = np.array(RT)

            to_write_tr = np.expand_dims(np.ndarray.flatten(random_transform), 0)
            angle_list = np.vstack((angle_list, to_write_tr))

            points = np.loadtxt(cloud_name)
            points = points[:, :3] * 0.01
            ones_col = np.ones(shape=(points.shape[0], 1))
            points = np.hstack((points,ones_col))
            current_img = smc.imread(img_name)

            img = smc.imread(img_name)
            img_ht = img.shape[0]
            img_wdt = img.shape[1]

            transformed_points = np.matmul(ExtrinsicParameter_meter, np.matmul(random_transform, points.T))
            points_2d = transformed_points

            Z = points_2d[2,:]
            x = (points_2d[0,:]/Z).T
            y = (points_2d[1,:]/Z).T

            x = np.clip(x, 0.0, img_wdt - 1)
            y = np.clip(y, 0.0, img_ht - 1)

            reprojected_img = np.zeros_like(img)
            for x_idx, y_idx,z_idx in zip(x,y,Z):
                if(z_idx>0):
                    reprojected_img[int(y_idx), int(x_idx)] = z_idx

            smc.imsave(dataset_path + foldername + "/depth_maps_transformed_3" + "/" + imagefile, reprojected_img)

            points_2d = np.matmul(ExtrinsicParameter_meter, points.T)

            Z = points_2d[2,:]
            x = (points_2d[0,:]/Z).T
            y = (points_2d[1,:]/Z).T

            x = np.clip(x, 0.0, img_wdt - 1)
            y = np.clip(y, 0.0, img_ht - 1)

            reprojected_img = np.zeros_like(img)
            for x_idx, y_idx,z_idx in zip(x,y,Z):
                if(z_idx > 0):
                    reprojected_img[int(y_idx), int(x_idx)] = z_idx
            pooled_img = reprojected_img

            reconstructed_img = current_img*(pooled_img>0.)
            smc.imsave(dataset_path + foldername + "/depth_maps_3" + "/" + imagefile, pooled_img)
            smc.imsave(dataset_path + foldername + "/target_images_3" + "/" + imagefile, reconstructed_img)

        np.savetxt(dataset_path + foldername + "/angle_list_3.txt", angle_list[1:], fmt = "%.4f")