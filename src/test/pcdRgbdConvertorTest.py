import os

import cv2
import numpy as np

from src.loaders.config import IclNuim
from src.utils.point_cloud import rgb_and_depth_to_pcd_custom, pcd_to_rgb_and_depth_custom


def test_rgbd_pcd_rgbd_convertion():
    depth_frame_path = os.path.join("data", "depth.png")
    rgb_frame_path = os.path.join("data", "rgb.png")
    depth_image = cv2.imread(depth_frame_path, cv2.IMREAD_ANYDEPTH)
    rgb_image = cv2.imread(rgb_frame_path)
    camera_intrinsics = IclNuim.get_cam_intrinsic(depth_image.shape)
    initial_pcd_transform = IclNuim.get_initial_pcd_transform()

    pcd = rgb_and_depth_to_pcd_custom(rgb_image, depth_image, camera_intrinsics, initial_pcd_transform)
    rgb_image_restored, depth_image_restored = pcd_to_rgb_and_depth_custom(
        pcd,
        camera_intrinsics,
        initial_pcd_transform
    )

    cv2.imwrite("rgbRes.png", rgb_image_restored)
    cv2.imwrite("depthRes.png", depth_image_restored)

    np.testing.assert_almost_equal(rgb_image, rgb_image_restored)
    np.testing.assert_almost_equal(depth_image, depth_image_restored)
