import open3d as o3d
import numpy as np

from src.loaders.depth_image.CameraIntrinsics import CameraIntrinsics
from src.utils.colors import normalize_color_arr


def rgb_and_depth_to_pcd_custom(rgb_image, depth_image, camera_intrinsics: CameraIntrinsics, initial_pcd_transform):
    pcd = depth_to_pcd_custom(depth_image, camera_intrinsics, initial_pcd_transform)
    pcd = load_rgb_colors_to_pcd(rgb_image, pcd)

    return pcd


def depth_to_pcd_custom(depth_image, camera_intrinsics: CameraIntrinsics, initial_pcd_transform):
    points = np.zeros((camera_intrinsics.width * camera_intrinsics.height, 3))

    column_indices = np.tile(np.arange(camera_intrinsics.width), (camera_intrinsics.height, 1)).flatten()
    row_indices = np.transpose(np.tile(np.arange(camera_intrinsics.height), (camera_intrinsics.width, 1))).flatten()

    points[:, 2] = depth_image.flatten() / camera_intrinsics.factor
    points[:, 0] = (column_indices - camera_intrinsics.cx) * points[:, 2] / camera_intrinsics.fx
    points[:, 1] = (row_indices - camera_intrinsics.cy) * points[:, 2] / camera_intrinsics.fy

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.transform(initial_pcd_transform)

    return pcd


def load_rgb_colors_to_pcd(rgb_image, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    colors = normalize_color_arr(rgb_image.reshape(rgb_image.shape[0] * rgb_image.shape[1], 3))
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def pcd_to_rgb_and_depth_custom(
        pcd: o3d.geometry.PointCloud,
        camera_intrinsics: CameraIntrinsics,
        initial_pcd_transform
):
    inverted_transform = np.linalg.inv(initial_pcd_transform)
    pcd.transform(inverted_transform)
    colors = np.asarray(pcd.colors)
    points = np.asarray(pcd.points)

    # This works with knowledge that pcd is never permuted in all processing
    # and its order is the same as after rgb_and_depth_to_pcd_custom --- so we can simply reshape to images
    rgb_image = (colors.reshape((camera_intrinsics.height, camera_intrinsics.width, 3)) * 255).astype(dtype=np.uint8)
    depth_image = points[:, 2].reshape((camera_intrinsics.height, camera_intrinsics.width)) * camera_intrinsics.factor
    depth_image = depth_image.astype(dtype=np.uint16)

    return rgb_image, depth_image
