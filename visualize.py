import matplotlib.pyplot as plt
import imageio as smc
import numpy as np
import cv2



changwon_depth_map_path = "/media/ros/Data2/2021_12_02/train/2021-01-19_10-13-27/depth_maps/2021-01-19_10-13-2700000000.png"
changwon_transformed_image_path = "/media/ros/Data2/2021_12_02/train/2021-01-19_10-13-27/depth_maps_transformed/2021-01-19_10-13-2700000000.png"
kitti_transformed_image_path = "/media/ros/Data2/2011_09_26/2011_09_26_drive_0001_sync/depth_maps_transformed/0000000000.png"
kitti_depth_map_path = "/media/ros/Data2/2011_09_26/2011_09_26_drive_0001_sync/depth_maps/0000000000.png"


kitti_transformed_image = np.float32(cv2.imread(kitti_transformed_image_path, flags=cv2.IMREAD_GRAYSCALE))

plt.imshow(kitti_transformed_image)
plt.savefig("/media/ros/Data2/ppt/kitti_transformed_image.png")


kitti_depth_map = np.float32(cv2.imread(kitti_depth_map_path, flags=cv2.IMREAD_GRAYSCALE))

plt.imshow(kitti_depth_map)
plt.savefig("/media/ros/Data2/ppt/kitti_depth_map.png")


kitti_transformed_image = np.float32(cv2.imread(changwon_transformed_image_path, flags=cv2.IMREAD_GRAYSCALE))

plt.imshow(kitti_transformed_image)
plt.savefig("/media/ros/Data2/ppt/changwon_transformed_image.png")


kitti_depth_map = np.float32(cv2.imread(changwon_depth_map_path, flags=cv2.IMREAD_GRAYSCALE))

plt.imshow(kitti_depth_map)
plt.savefig("/media/ros/Data2/ppt/changwon_depth_map.png")

