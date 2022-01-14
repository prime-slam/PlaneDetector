import numpy as np

from src.annotations.BaseAnnotator import BaseAnnotator
from src.annotations.cvat.CVATAnnotation import CVATAnnotation
from src.loaders.depth_image.CameraIntrinsics import CameraIntrinsics
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
            color_decoded = color_str.decode('UTF-8')
            if color_decoded == black_color_str:
                unsegmented_cloud_indices = indexes
            else:
                if color_decoded in color_to_track:
                    track_id = color_to_track[color_decoded]
                else:
                    track_id = next_track_id
                    color_to_track[color_decoded] = track_id
                    next_track_id += 1

                planes.append(
                    SegmentedPlane(
                        indexes,
                        track_id,
                        color_from_string(color_str)
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
        colors_string = np.fromiter((color_to_string(denormalize_color(color)) for color in colors), dtype='|S256')
        unique_colors = np.unique(colors_string)
        for color in unique_colors:
            # remember that np.where returns tuple --- we have to extract array from it
            result[color] = np.where(colors_string == color)[0]

        return result
