import open3d as o3d


class PointCloudPrinter:
    def __init__(self, pcd: o3d.geometry.PointCloud):
        self.pcd = pcd

    def save_to_pcd(self, filename):
        assert filename[-4:] == ".pcd"
        o3d.io.write_point_cloud(filename, self.pcd, write_ascii=True)

    def save_to_ply(self, filename):
        assert filename[-4:] == ".ply"
        o3d.io.write_point_cloud(filename, self.pcd)
