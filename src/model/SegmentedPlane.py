import numpy as np
import open3d as o3d

from src.utils.colors import get_random_color, normalize_color


class SegmentedPlane:

    NO_TRACK = -1

    def __init__(self, pcd_indices: np.array, track_id: int, color=None):
        self.pcd_indices = pcd_indices
        self.track_id = track_id
        if color is None:
            self.color = get_random_color()
        else:
            self.color = color
        self.normalized_color = normalize_color(self.color)
        # pcd.paint_uniform_color(self.normalized_color)

    def __repr__(self):
        return "Plane: {{pcd_indices: {0}, color: {1}, track_id: {2}}}".format(self.pcd_indices, self.color, self.track_id)
