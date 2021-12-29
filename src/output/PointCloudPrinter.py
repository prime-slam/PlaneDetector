import numpy as np
import open3d as o3d


class PointCloudPrinter:
    def __init__(self, pcd: o3d.geometry.PointCloud):
        self.pcd = pcd

    def save_to_pcd(self, filename):
        assert filename[-4:] == ".pcd"
        o3d.io.write_point_cloud(filename, self.pcd)

    def save_to_ply(self, filename):
        assert filename[-4:] == ".ply"

        points = np.asarray(self.pcd.points)
        colors = np.asarray(self.pcd.colors)

        formatted_points = []

        for point, color in zip(points, colors):
            formatted_points.append(
                "%f %f %f %d %d %d 0\n" % (point[0], point[1], point[2], color[0], color[1], color[2])
            )

        with open(filename, "w") as output:
            output.write(
                '''ply
format ascii 1.0
element vertex %d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property uchar alpha
end_header
%s
                ''' % (len(points), "".join(formatted_points)))
