import os

import numpy as np
import open3d as o3d

from src.loaders.BaseLoader import BaseLoader
from src.model.SegmentedPointCloud import SegmentedPointCloud


class KittiLoader(BaseLoader):
    def __init__(self, path):
        super().__init__(path)
        cloud_filenames = os.listdir(path)
        self.clouds = [os.path.join(path, filename) for filename in cloud_filenames]

    def get_frame_count(self) -> int:
        return len(self.clouds)

    def read_pcd(self, frame_num) -> SegmentedPointCloud:
        cloud_path = self.clouds[frame_num]
        pcd_points = np.fromfile(cloud_path, dtype=np.float32).reshape(-1, 4)
        pcd = o3d.geometry.PointCloud()
        # data contains [x, y, z, reflectance] for each point -- we skip the last one
        pcd.points = o3d.utility.Vector3dVector(pcd_points[:, :3])
        cloud_size = pcd_points.shape[0]

        return SegmentedPointCloud(
            pcd=pcd,
            unsegmented_cloud_indices=np.arange(cloud_size)
        )
