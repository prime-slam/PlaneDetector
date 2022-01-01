import open3d as o3d

from src.model.SegmentedPlane import SegmentedPlane


class SegmentedPointCloud:

    def __init__(self, planes: [SegmentedPlane] = None, unsegmented_cloud: o3d.geometry.PointCloud = None):
        if planes is None:
            planes = []
        self.planes = planes
        if unsegmented_cloud is None:
            unsegmented_cloud = o3d.geometry.PointCloud()
        self.unsegmented_cloud = unsegmented_cloud
        self.unsegmented_cloud.paint_uniform_color([0.5, 0.5, 0.5])

    def __repr__(self):
        return "Cloud: {{planes: {0}, unsegmented_cloud: {1}}}".format(self.planes, self.unsegmented_cloud)

    def get_color_pcd_for_visualization(self):
        res = self.unsegmented_cloud
        for plane in self.planes:
            plane.pcd.paint_uniform_color(plane.normalized_color)
            res += plane.pcd

        return res
