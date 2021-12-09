import numpy as np
import scipy.misc as smc
from natsort import natsorted as ns
import glob, os
import argparse

dataset_path = '/data/2021_12_02/train/'

folder_names = ns(glob.glob(dataset_path +"*" + os.path.sep))

dataset_array = np.zeros(dtype = str, shape = (1,21))
dataset_array_2 = np.zeros(dtype = str, shape = (1,21))
dataset_array_3 = np.zeros(dtype= str, shape = (1,21))

for fn in folder_names:
    print (fn)
    file_names_source = ns(glob.glob(fn + "depth_maps_transformed/*.png"))
    file_names_target = ns(glob.glob(fn + "depth_maps/*.png"))
    img_source = ns(glob.glob(fn + "images/*.png"))
    img_target = ns(glob.glob(fn + "images/*.png"))
    transforms_list = np.loadtxt(fn + "angle_list.txt", dtype = str)
    pointcloud_list = ns(glob.glob(fn + "lidars/*.txt"))

    file_names_source = np.array(file_names_source, dtype=str).reshape(-1,1)
    file_names_target = np.array(file_names_target, dtype=str).reshape(-1,1)
    img_source = np.array(img_source, dtype=str).reshape(-1,1)
    img_target = np.array(img_target, dtype=str).reshape(-1,1)
    pointcloud_list = np.array(pointcloud_list, dtype=str).reshape(-1, 1)

    dataset = np.hstack((file_names_source, file_names_target, img_source, img_target, pointcloud_list, transforms_list))
    print(dataset.shape)

    dataset_array = np.vstack((dataset_array, dataset))

    #######################################################################################

    file_names_source_2 = ns(glob.glob(fn + "depth_maps_transformed_2/*.png"))
    file_names_target_2 = ns(glob.glob(fn + "depth_maps_2/*.png"))
    transforms_list_2 = np.loadtxt(fn + "angle_list_2.txt", dtype = str)
    file_names_source_2 = np.array(file_names_source_2, dtype=str).reshape(-1,1)
    file_names_target_2 = np.array(file_names_target_2, dtype=str).reshape(-1,1)

    dataset_2 = np.hstack((file_names_source_2, file_names_target_2, img_source, img_target, pointcloud_list, transforms_list_2))
    print(dataset_2.shape)

    dataset_array_2 = np.vstack((dataset_array_2, dataset_2))

    #######################################################################################

    file_names_source_3 = ns(glob.glob(fn + "depth_maps_transformed_3/*.png"))
    file_names_target_3 = ns(glob.glob(fn + "depth_maps_3/*.png"))

    transforms_list_3 = np.loadtxt(fn + "angle_list_3.txt", dtype = str)

    file_names_source_3 = np.array(file_names_source_3, dtype=str).reshape(-1,1)
    file_names_target_3 = np.array(file_names_target_3, dtype=str).reshape(-1,1)

    dataset_3 = np.hstack((file_names_source_3, file_names_target_3, img_source, img_target, pointcloud_list, transforms_list_3))
    print(dataset_3.shape)

    dataset_array_3 = np.vstack((dataset_array_3, dataset_3))

dataset_array = dataset_array[1:]
dataset_array_2 = dataset_array_2[1:]
dataset_array_3 = dataset_array_3[1:]

final_array = np.vstack((dataset_array, dataset_array_2, dataset_array_3))

np.random.shuffle(final_array)
np.savetxt("parsed_set_changwon.txt", final_array, fmt = "%s", delimiter=' ')