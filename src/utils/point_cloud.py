import open3d as o3d
import numpy as np


def rgbd_to_pcd(rgbd_image, camera_intrinsics, initial_pcd_transform):
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        camera_intrinsics
    )
    pcd.transform(initial_pcd_transform)
    return pcd


def depth_to_pcd(depth_image, camera_intrinsics, initial_pcd_transform):
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth_image,
        camera_intrinsics,
        depth_scale=5000.0,
        depth_trunc=1000.0
    )
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
