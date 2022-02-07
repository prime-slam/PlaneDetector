import os

import numpy as np
import open3d as o3d

from src.loaders.BaseLoader import BaseLoader
from src.model.SegmentedPointCloud import SegmentedPointCloud


class O3DLoader(BaseLoader):
    def __init__(self, path):
        super().__init__(path)
        cloud_filenames = os.listdir(path)
        self.clouds = [os.path.join(path, filename) for filename in cloud_filenames]

    def get_frame_count(self) -> int:
        return len(self.clouds)

    def read_pcd(self, frame_num) -> SegmentedPointCloud:
        cloud_path = self.clouds[frame_num]
        pcd = o3d.io.read_point_cloud(cloud_path)
        cloud_size = np.asarray(pcd.points).shape[0]

        return SegmentedPointCloud(
            pcd=pcd, unsegmented_cloud_indices=np.arange(cloud_size)
        )
