import open3d as o3d

from src.SegmentedPlane import SegmentedPlane
from src.SegmentedPointCloud import SegmentedPointCloud


def detect_plane(pcd):
    _, inliers = pcd.segment_plane(
        distance_threshold=0.000001,
        ransac_n=3,
        num_iterations=1000
    )
    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    return inlier_cloud, outlier_cloud


def detect_planes(pcd: o3d.geometry.PointCloud, num_planes=5) -> SegmentedPointCloud:
    outlier_pcd = pcd
    detected_planes = []

    for _ in range(num_planes):
        inlier_pcd, outlier_pcd = detect_plane(outlier_pcd)
        detected_planes.append(SegmentedPlane(inlier_pcd))

    return SegmentedPointCloud(detected_planes, outlier_pcd)
