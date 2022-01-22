import numpy as np

from src.detectors.O3DRansacDetector import O3DRansacDetector
from src.model.SegmentedPointCloud import SegmentedPointCloud, SegmentedPlane


def remove_planes_outliers(segmented_pcd: SegmentedPointCloud) -> SegmentedPointCloud:
    updated_planes = []
    unsegmented_indices_list = []
    if segmented_pcd.unsegmented_cloud_indices is not None:
        unsegmented_indices_list.append(segmented_pcd.unsegmented_cloud_indices)
    for plane in segmented_pcd.planes:
        if plane.pcd_indices.size < 3:
            updated_planes.append(SegmentedPlane(plane.pcd_indices, plane.track_id, plane.color))
            continue
        plane_pcd = segmented_pcd.pcd.select_by_index(plane.pcd_indices)
        inliers, outliers = O3DRansacDetector.detect_plane(plane_pcd)  # indices in the small pcd of only one plane
        inlier_indices = plane.pcd_indices[inliers]
        outlier_indices = plane.pcd_indices[outliers]
        unsegmented_indices_list.append(outlier_indices)
        updated_planes.append(SegmentedPlane(inlier_indices, plane.track_id, plane.color))

    updated_unsegmented_indices = np.concatenate(unsegmented_indices_list)

    return SegmentedPointCloud(
        segmented_pcd.pcd,
        updated_planes,
        unsegmented_cloud_indices=updated_unsegmented_indices,
        zero_depth_cloud_indices=segmented_pcd.zero_depth_cloud_indices,
        structured_shape=segmented_pcd.structured_shape
    )
