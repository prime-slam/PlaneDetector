import numpy as np
import open3d as o3d

from src.SegmentedPlane import SegmentedPlane
from src.SegmentedPointCloud import SegmentedPointCloud


def detect_planes(pcd: o3d.geometry.PointCloud) -> SegmentedPointCloud:
    with open("planes.txt", 'r') as planes_input:
        all_plane_indices = set()
        planes = []
        for line in planes_input:
            indices = [int(index) for index in line.split(" ")[:-1]]
            all_plane_indices.update(indices)
            plane_pcd = pcd.select_by_index(np.asarray(indices))
            planes.append(SegmentedPlane(plane_pcd))

        outlier_pcd = pcd.select_by_index(np.asarray(list(all_plane_indices)), invert=True)

        return SegmentedPointCloud(planes, outlier_pcd)
