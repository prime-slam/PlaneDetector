import cv2
import numpy as np

from src.annotations.BaseAnnotator import BaseAnnotator
from src.annotations.cvat.CVATAnnotation import CVATAnnotation
from src.model.SegmentedPlane import SegmentedPlane
from src.model.SegmentedPointCloud import SegmentedPointCloud
from src.utils.annotations import draw_polygones
from src.utils.colors import color_to_string, color_from_string, denormalize_color
from src.utils.point_cloud import load_rgb_colors_to_pcd


class CVATAnnotator(BaseAnnotator):
    def __init__(self, path, start_frame_num: int):
        super().__init__(path, start_frame_num)
        self.annotation = CVATAnnotation(path, start_frame_num)

    def annotate(self, segmented_pcd: SegmentedPointCloud, frame_num: int) -> SegmentedPointCloud:
        if segmented_pcd.structured_shape is None:
            raise Exception("CVAT annotation works only for structured pcd")

        all_planes = self.annotation.get_all_planes_for_frame(frame_num)
        all_planes = sorted(all_planes, key=lambda x: x.z)
        annotated_rgb = draw_polygones(all_planes, segmented_pcd.structured_shape)
        # cv2.imwrite("{}_annot.png".format(frame_num), annotated_rgb)

        colored_pcd = load_rgb_colors_to_pcd(annotated_rgb, segmented_pcd.pcd)

        black_color = np.array([0., 0., 0.])
        black_color_str = color_to_string(black_color)
        planes_indices = self.__group_pcd_indexes_by_color(colored_pcd)

        planes = []
        unsegmented_cloud_indices = None
        for color_str, indices in planes_indices.items():
            if color_str == black_color_str:
                unsegmented_cloud_indices = indices
            else:
                denorm_color_str = color_to_string(denormalize_color(color_from_string(color_str)).astype(dtype=int))
                track_id = self.annotation.color_to_track[denorm_color_str]

                not_zero_indices = np.setdiff1d(
                    indices,
                    segmented_pcd.zero_depth_cloud_indices
                )
                zero_indices = np.setdiff1d(
                    indices,
                    not_zero_indices
                )

                planes.append(
                    SegmentedPlane(
                        not_zero_indices,
                        zero_indices,
                        track_id,
                        denormalize_color(color_from_string(color_str))
                    )
                )

        return SegmentedPointCloud(
            colored_pcd,
            planes,
            unsegmented_cloud_indices=unsegmented_cloud_indices,
            zero_depth_cloud_indices=segmented_pcd.zero_depth_cloud_indices,
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
