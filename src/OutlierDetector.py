from src.model.SegmentedPointCloud import SegmentedPointCloud, SegmentedPlane
from src.detectors.O3DRansacDetector import detect_plane


def remove_planes_outliers(segmented_pcd: SegmentedPointCloud) -> SegmentedPointCloud:
    updated_planes = []
    updated_unsegmented_pcd = segmented_pcd.unsegmented_cloud
    for plane in segmented_pcd.planes:
        inlier_pcd, outlier_pcd = detect_plane(plane.pcd)
        updated_unsegmented_pcd += outlier_pcd
        updated_planes.append(SegmentedPlane(inlier_pcd, plane.track_id, plane.color))

    return SegmentedPointCloud(updated_planes, updated_unsegmented_pcd)
