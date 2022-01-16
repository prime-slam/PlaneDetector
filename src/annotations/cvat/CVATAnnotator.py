import numpy as np

from src.annotations.BaseAnnotator import BaseAnnotator
from src.annotations.cvat.CVATAnnotation import CVATAnnotation
from src.model.SegmentedPlane import SegmentedPlane
from src.model.SegmentedPointCloud import SegmentedPointCloud
from src.utils.annotations import draw_polygones
from src.utils.colors import color_to_string, color_from_string, denormalize_color
from src.utils.point_cloud import load_rgb_colors_to_pcd


next_track_id = 0
color_to_track = {}


class CVATAnnotator(BaseAnnotator):
    def __init__(self, path, start_frame_num: int):
        super().__init__(path, start_frame_num)
        self.annotation = CVATAnnotation(path, start_frame_num)

    def annotate(self, segmented_pcd: SegmentedPointCloud, frame_num: int) -> SegmentedPointCloud:
        if segmented_pcd.structured_shape is None:
            raise Exception("CVAT annotation works only for structured pcd")

        all_planes = self.annotation.get_all_planes_for_frame(frame_num)
        annotated_rgb = draw_polygones(all_planes, segmented_pcd.structured_shape)

        colored_pcd = load_rgb_colors_to_pcd(annotated_rgb, segmented_pcd.pcd)

        black_color = np.array([0., 0., 0.])
        black_color_str = color_to_string(black_color)
        planes_indexes = self.__group_pcd_indexes_by_color(colored_pcd)

        planes = []
        global next_track_id
        unsegmented_cloud_indices = None
        for color_str, indexes in planes_indexes.items():
            if color_str == black_color_str:
                unsegmented_cloud_indices = indexes
            else:
                if color_str in color_to_track:
                    track_id = color_to_track[color_str]
                else:
                    track_id = next_track_id
                    color_to_track[color_str] = track_id
                    next_track_id += 1

                planes.append(
                    SegmentedPlane(
                        indexes,
                        track_id,
                        denormalize_color(color_from_string(color_str))
                    )
                )

        return SegmentedPointCloud(
            colored_pcd,
            planes,
            unsegmented_cloud_indices,
            structured_shape=segmented_pcd.structured_shape
        )

    def __group_pcd_indexes_by_color(self, pcd):
        result = {}
        colors = np.asarray(pcd.colors)
        unique_colors = np.unique(colors, axis=0)
        for color in unique_colors:
            # remember that np.where returns tuple --- we have to extract array from it
            result[color_to_string(color)] = np.where(np.all(colors == color, axis=1))[0]

        return result
