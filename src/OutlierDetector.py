import open3d as o3d

from src.SegmentedPointCloud import SegmentedPointCloud, SegmentedPlane
from src.detectors.O3DRansacDetector import detect_plane
from src.utils.point_cloud import merge_pcd


def remove_planes_outliers(segmented_pcd: SegmentedPointCloud) -> SegmentedPointCloud:
    updated_planes = []
    updated_unsegmented_pcd = segmented_pcd.unsegmented_cloud
    for plane in segmented_pcd.planes:
        inlier_pcd, outlier_pcd = detect_plane(plane.pcd)
        updated_unsegmented_pcd = merge_pcd(updated_unsegmented_pcd, outlier_pcd)
        updated_planes.append(SegmentedPlane(inlier_pcd))

    return SegmentedPointCloud(updated_planes, updated_unsegmented_pcd)
