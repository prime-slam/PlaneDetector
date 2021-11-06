import open3d as o3d

from src.utils.colors import get_random_color, normalize_color


class SegmentedPlane:
    def __init__(self, pcd: o3d.geometry.PointCloud):
        self.pcd = pcd
        self.color = get_random_color()
        self.normalized_color = normalize_color(self.color)
        pcd.paint_uniform_color(self.normalized_color)

    def __repr__(self):
        return "Plane: {{pcd: {0}, color: {1}}}".format(self.pcd, self.color)
