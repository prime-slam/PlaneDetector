import open3d as o3d
import numpy as np

from src.model.SegmentedPointCloud import SegmentedPointCloud
from src.loaders.config import CameraIntrinsics
from src.utils.colors import normalize_color, denormalize_color


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


# TODO: vectorize and create segmentedpoint cloud  instead of o3d pcd
def rgb_and_depth_to_pcd_custom(rgb_image, depth_image, camera_intrinsics: CameraIntrinsics, initial_pcd_transform):
    factor = 5000  # for the 16-bit PNG files
    colors = np.zeros((camera_intrinsics.width * camera_intrinsics.height, 3))
    points = np.zeros((camera_intrinsics.width * camera_intrinsics.height, 3))
    for u in range(0, camera_intrinsics.width):
        for v in range(0, camera_intrinsics.height):
            number = v * camera_intrinsics.width + u
            colors[number] = normalize_color(rgb_image[v, u])
            points[number, 2] = depth_image[v, u] / factor
            points[number, 0] = (u - camera_intrinsics.cx) * points[number, 2] / camera_intrinsics.fx
            points[number, 1] = (v - camera_intrinsics.cy) * points[number, 2] / camera_intrinsics.fy
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.transform(initial_pcd_transform)
    return pcd


# TODO: vectorize and create segmented point cloud  nstead of o3d pcd
def pcd_to_rgb_and_depth_custom(pcd: o3d.geometry.PointCloud, camera_intrinsics: CameraIntrinsics, initial_pcd_transform):
    rgb_image = np.zeros((camera_intrinsics.height, camera_intrinsics.width, 3), dtype=np.uint8)
    depth_image = np.zeros((camera_intrinsics.height, camera_intrinsics.width), dtype=np.uint16)
    factor = 5000  # for the 16-bit PNG files
    inverted_transform = np.linalg.inv(initial_pcd_transform)
    pcd.transform(inverted_transform)
    colors = np.asarray(pcd.colors)
    points = np.asarray(pcd.points)

    for index, point in enumerate(points):
        u = round(point[0] * camera_intrinsics.fx / point[2] + camera_intrinsics.cx)
        v = round(point[1] * camera_intrinsics.fy / point[2] + camera_intrinsics.cy)

        rgb_image[v, u] = denormalize_color(colors[index])
        depth_image[v, u] = point[2] * factor

    return rgb_image, depth_image
