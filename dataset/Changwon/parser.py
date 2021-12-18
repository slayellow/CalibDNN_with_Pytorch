import numpy as np
import scipy.misc as smc
from natsort import natsorted as ns
import glob, os
import argparse

dataset_path = '/data/2021_12_02/train/'

folder_names = ns(glob.glob(dataset_path +"*" + os.path.sep))

dataset_array = np.zeros(dtype = str, shape = (1,2))
dataset_array_2 = np.zeros(dtype = str, shape = (1,2))
dataset_array_3 = np.zeros(dtype= str, shape = (1,2))

for fn in folder_names:
    print (fn)
    img_source = ns(glob.glob(fn + "images/*.png"))
    pointcloud_list = ns(glob.glob(fn + "lidars/*.txt"))

    img_source = np.array(img_source, dtype=str).reshape(-1,1)
    pointcloud_list = np.array(pointcloud_list, dtype=str).reshape(-1, 1)

    dataset = np.hstack((img_source, pointcloud_list))
    print(dataset.shape)

    dataset_array = np.vstack((dataset_array, dataset))

dataset_array = dataset_array[1:]

np.random.shuffle(dataset_array)
np.savetxt("parsed_set_changwon.txt", dataset_array, fmt = "%s", delimiter=' ')
