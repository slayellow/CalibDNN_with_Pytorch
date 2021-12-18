import numpy as np
import scipy.misc as smc
from natsort import natsorted as ns
import glob, os
import argparse


# parser = argparse.ArgumentParser(description="Create Lidar Dataset Parser file")
# parser.add_argument("path", help = "path_to_folder", type = str)
# args = parser.parse_args()
#
# dataset_path = args.path
dataset_path = '/data/2011_09_26/'


#Picking up all sync folders
folder_names = ns(glob.glob(dataset_path +"*_sync" + os.path.sep))

dataset_array = np.zeros(dtype = str, shape = (1,2))

for fn in folder_names:
    print (fn)
    img_source = ns(glob.glob(fn + "image_02/data/*.png"))
    pointcloud_list = []
    with open(fn + "pointcloud_list.txt") as file:
        for line in file:
            line = line.strip()
            pointcloud_list.append(line)

    img_source = np.array(img_source, dtype=str).reshape(-1,1)
    pointcloud_list = np.array(pointcloud_list, dtype=str).reshape(-1, 1)

    dataset = np.hstack((img_source, pointcloud_list))
    print(dataset.shape)

    dataset_array = np.vstack((dataset_array, dataset))

dataset_array = dataset_array[1:]

np.random.shuffle(dataset_array)
np.savetxt("parsed_set.txt", dataset_array, fmt = "%s", delimiter=' ')
