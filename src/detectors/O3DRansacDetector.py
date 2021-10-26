import open3d as o3d

from src.utils.colors import get_random_normalized_color
from src.utils.point_cloud import merge_pcd


def detect_plane(pcd):
    _, inliers = pcd.segment_plane(
        distance_threshold=0.01,
        ransac_n=3,
        num_iterations=1000
    )
    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    return inlier_cloud, outlier_cloud


def detect_planes(pcd, num_planes=5):
    outlier_pcd = pcd
    result = o3d.geometry.PointCloud()

    for _ in range(num_planes):
        inlier_pcd, outlier_pcd = detect_plane(outlier_pcd)
        color = get_random_normalized_color()
        inlier_pcd.paint_uniform_color(color)
        result = merge_pcd(result, inlier_pcd)

    black_color = [0., 0., 0.]
    outlier_pcd.paint_uniform_color(black_color)
    return merge_pcd(result, outlier_pcd)
