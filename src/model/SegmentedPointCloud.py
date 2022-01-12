import numpy as np
import open3d as o3d

from src.model.SegmentedPlane import SegmentedPlane


class SegmentedPointCloud:

    UNSEGMENTED_PCD_COLOR = [0, 0, 0]  # [0.5, 0.5, 0.5]

    def __init__(
            self,
            pcd: o3d.geometry.PointCloud,
            planes: [SegmentedPlane] = None,
            unsegmented_cloud_indices: np.array = None,
    ):
        if pcd is None:
            pcd = o3d.geometry.PointCloud()
        # Without this colors vector will be just empty
        pcd.paint_uniform_color(SegmentedPointCloud.UNSEGMENTED_PCD_COLOR)
        self.pcd = pcd

        if planes is None:
            planes = []
        self.planes = planes
        if unsegmented_cloud_indices is None:
            unsegmented_cloud_indices = np.asarray([], dtype=np.int64)
        self.unsegmented_cloud_indices = unsegmented_cloud_indices

    def __repr__(self):
        return "Cloud: {{planes: {0}, unsegmented_cloud_indices: {1}, pcd: {2}}}".format(
            self.planes,
            self.unsegmented_cloud_indices,
            self.pcd
        )

    def get_color_pcd_for_visualization(self):
        colors = np.asarray(self.pcd.colors)
        colors[self.unsegmented_cloud_indices] = SegmentedPointCloud.UNSEGMENTED_PCD_COLOR
        for plane in self.planes:
            color = plane.normalized_color
            colors[plane.pcd_indices] = color

        self.pcd.colors = o3d.utility.Vector3dVector(colors)

        return self.pcd
