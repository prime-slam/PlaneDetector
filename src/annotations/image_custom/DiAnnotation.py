import os

import cv2
import open3d as o3d

from src.utils.colors import normalize_color_arr


class DiAnnotation:
    def __init__(self, path: str):
        self.path = path
        annotations = [os.path.join(path, filename) for filename in os.listdir(path)]
        self.annotations = sorted(annotations, key=lambda x: int(x))

    def annotate(self, pcd: o3d.geometry.PointCloud, frame_num: int) -> o3d.geometry.PointCloud:
        annotation = self.annotations[frame_num]
        annotation_rgb = cv2.imread(annotation)
        colors = normalize_color_arr(annotation_rgb.reshape((annotation_rgb.shape[0] * annotation_rgb.shape[1], 3)))

        pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd
