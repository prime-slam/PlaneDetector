import numpy as np
import open3d as o3d

from src.model.SegmentedPlane import SegmentedPlane
from src.utils.colors import UNSEGMENTED_PCD_COLOR_NORMALISED


class SegmentedPointCloud:

    def __init__(
            self,
            pcd: o3d.geometry.PointCloud,
            planes: [SegmentedPlane] = None,
            zero_depth_cloud_indices: np.array = np.asarray([], dtype=np.int64),
            unsegmented_cloud_indices: np.array = np.asarray([], dtype=np.int64),
            structured_shape: tuple = None
    ):
        if pcd is None:
            pcd = o3d.geometry.PointCloud()
        # Without this colors vector will be just empty
        pcd.paint_uniform_color(UNSEGMENTED_PCD_COLOR_NORMALISED)
        self.pcd = pcd

        if planes is None:
            planes = []
        self.planes = planes
        self.zero_depth_cloud_indices = zero_depth_cloud_indices
        self.unsegmented_cloud_indices = unsegmented_cloud_indices
        self.structured_shape = structured_shape

    def __repr__(self):
        return "Cloud: {{planes: {0}, unsegmented_cloud_indices: {1}, zero_depth_cloud_indices: {2}, pcd: {3}}}".format(
            self.planes,
            self.unsegmented_cloud_indices,
            self.zero_depth_cloud_indices,
            self.pcd
        )

    def filter_planes(self, filter_func):
        filtered_planes = []
        for plane in self.planes:
            if filter_func(plane):
                filtered_planes.append(plane)
        self.planes = filtered_planes

    def get_raw_pcd(self):
        result_pcd = o3d.geometry.PointCloud()
        result_pcd.points = self.pcd.points

        return result_pcd

    def get_color_pcd_for_visualization(self):
        colors = np.asarray(self.pcd.colors)
        colors[self.unsegmented_cloud_indices] = UNSEGMENTED_PCD_COLOR_NORMALISED
        for plane in self.planes:
            color = plane.normalized_color
            colors[plane.pcd_indices] = color

        self.pcd.colors = o3d.utility.Vector3dVector(colors)

        return self.pcd
