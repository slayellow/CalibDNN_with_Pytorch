import numpy as np
import os
from os.path import getsize
import glob
from natsort import natsorted as ns
from PIL import Image
import struct

IMG_HT = 480
IMG_WDT = 640

dataset_path = '/data/2021_12_02/'

navigation_datalist = ns(glob.glob(dataset_path + "*.navigation"))
navigation_datalist = [file.split('/')[3] for file in navigation_datalist]
navigation_datalist = [file.split('.')[0] for file in navigation_datalist]

lidar_packet_size = 1843873
image_packet_size = 921763

for filename in navigation_datalist:
    print("=============================================================================")
    print(filename + " Folder Created!!")
    if not os.path.exists(dataset_path + filename + "/lidars"):
        os.makedirs(dataset_path + filename + "/lidars")
    if not os.path.exists(dataset_path + filename + "/images"):
        os.makedirs(dataset_path + filename + "/images")

    images_folder = dataset_path + filename + "/images"
    lidars_folder = dataset_path + filename + "/lidars"

    lidar_file_path = dataset_path + filename + ".velodyne128xyz"
    image_file_path = dataset_path + filename + ".triplecameraccdcenter"

    lidar_file = open(lidar_file_path, 'rb')
    image_file = open(image_file_path, 'rb')

    lidar_file_size = getsize(lidar_file_path)
    image_file_size = getsize(image_file_path)

    fileIndex = 0
    image_count = 0
    lidar_count = 0

    while True:
        if lidar_count == lidar_file_size:
            break
        xyz_file = open(lidars_folder + "/" + filename + "{0:08d}".format(fileIndex) + ".txt", "w")
        # Lidar File Read
        byte = lidar_file.read(8)
        lidar_synctime = struct.unpack('d', byte)
        byte = lidar_file.read(8)
        lidar_navi_timestamp = struct.unpack('n', byte)
        lidar_file.seek(657, 1)
        lidar_count = lidar_count + 673
        while lidar_count % lidar_packet_size != 0:
            byte = lidar_file.read(2)
            x = struct.unpack('h', byte)[0]
            byte = lidar_file.read(2)
            y = struct.unpack('h', byte)[0]
            byte = lidar_file.read(2)
            z = struct.unpack('h', byte)[0]
            byte = lidar_file.read(1)
            layer_index = struct.unpack('B', byte)[0]
            byte = lidar_file.read(1)
            intensity = struct.unpack('B', byte)[0]
            if x == 0 and y == 0 and z == 0:
                lidar_count = lidar_count + 8
            else:
                xyzi = xyz_file.write("%d %d %d %d\n" % (x, y, z, intensity))
                lidar_count = lidar_count + 8
        xyz_file.close()

        while True:
            if image_count == image_file_size:
                break
            # Image File Read
            byte = image_file.read(8)
            synctime = struct.unpack('d', byte)
            byte = image_file.read(8)
            image_navi_timestamp = struct.unpack('n', byte)
            if image_navi_timestamp > lidar_navi_timestamp:
                image_file.seek(147, 1)
                image_count = image_count + 163
                byte = image_file.read(921600)
                image_count = image_count + 921600
                img = Image.frombytes('RGB', (IMG_WDT, IMG_HT), byte)
                image = np.array(img)
                red, green, blue = image.T
                image = np.array([blue, green, red])
                image = image.transpose()
                image = Image.fromarray(image)
                image.save(images_folder + "/" + filename + "{0:08d}".format(fileIndex) + ".png")
                fileIndex = fileIndex + 1
                print("LIDAR[", str(lidar_count), ":", str(lidar_file_size), "] :", str(lidar_navi_timestamp),
                      "IMAGE[", str(image_count), ":", str(image_file_size), "] :", str(image_navi_timestamp))
                break
            else:
                image_file.seek(image_packet_size - 16, 1)
                image_count = image_count + image_packet_size
    lidar_file.close()
    image_file.close()
    print("=============================================================================")
