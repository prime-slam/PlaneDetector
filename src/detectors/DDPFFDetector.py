import numpy as np
import open3d as o3d

from src.model.SegmentedPlane import SegmentedPlane
from src.model.SegmentedPointCloud import SegmentedPointCloud


def detect_planes(segmented_pcd: SegmentedPointCloud) -> SegmentedPointCloud:
    pcd = segmented_pcd.pcd
    with open("planes.txt", 'r') as planes_input:
        all_plane_indices = set()
        planes = []
        for line in planes_input:
            plane_indices = np.asarray([int(index) for index in line.split(" ")[:-1]])
            all_plane_indices.update(plane_indices)
            planes.append(SegmentedPlane(plane_indices, SegmentedPlane.NO_TRACK))

        outlier_pcd_indices = np.setdiff1d(
            np.arange(np.asarray(pcd.points).shape[0]),
            np.concatenate(list(all_plane_indices))
        )

        return SegmentedPointCloud(
            pcd,
            planes,
            outlier_pcd_indices,
            structured_shape=segmented_pcd.structured_shape
        )
