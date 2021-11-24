import open3d as o3d
import numpy as np

from src.loaders.config import CameraIntrinsics
from src.utils.colors import normalize_color


def rgbd_to_pcd(rgbd_image, camera_intrinsics: CameraIntrinsics, initial_pcd_transform):
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        camera_intrinsics.open3dIntrinsics
    )
    pcd.transform(initial_pcd_transform)
    return pcd


def depth_to_pcd(depth_image, camera_intrinsics: CameraIntrinsics, initial_pcd_transform):
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        o3d.geometry.Image(depth_image),
        camera_intrinsics.open3dIntrinsics,
        depth_scale=5000.0,
        depth_trunc=1000.0
    )
    pcd.transform(initial_pcd_transform)
    return pcd


def rgb_and_depth_to_pcd_custom(rgb_image, depth_image, camera_intrinsics: CameraIntrinsics, initial_pcd_transform):
    factor = 5000  # for the 16-bit PNG files
    colors = np.zeros((camera_intrinsics.width * camera_intrinsics.height, 3))
    points = np.zeros((camera_intrinsics.width * camera_intrinsics.height, 3))
    for u in range(0, camera_intrinsics.width):
        for v in range(0, camera_intrinsics.height):
            number = v * camera_intrinsics.width + u
            colors[number] = np.asarray(normalize_color(rgb_image[v, u]))
            points[number, 2] = depth_image[v, u] / factor
            points[number, 0] = (u - camera_intrinsics.cx) * points[number, 2] / camera_intrinsics.fx
            points[number, 1] = (v - camera_intrinsics.cy) * points[number, 2] / camera_intrinsics.fy
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.transform(initial_pcd_transform)
    return pcd


def merge_pcd(pcd_left, pcd_right):
    pcd_left_points = np.asarray(pcd_left.points)
    pcd_right_points = np.asarray(pcd_right.points)
    pcd_res_points = np.concatenate((pcd_left_points, pcd_right_points), axis=0)
    pcd_left_colors = np.asarray(pcd_left.colors)
    pcd_right_colors = np.asarray(pcd_right.colors)
    pcd_res_colors = np.concatenate((pcd_left_colors, pcd_right_colors), axis=0)
    pcd_res = o3d.geometry.PointCloud()
    pcd_res.points = o3d.utility.Vector3dVector(pcd_res_points)
    pcd_res.colors = o3d.utility.Vector3dVector(pcd_res_colors)

    return pcd_res
