import numpy as np

from src.loaders.depth_image.CameraIntrinsics import CameraIntrinsics
from src.model.SegmentedPlane import SegmentedPlane
from src.model.SegmentedPointCloud import SegmentedPointCloud
from src.utils.annotations import draw_polygones
from src.utils.colors import color_to_string, color_from_string, denormalize_color
from src.utils.point_cloud import load_rgb_colors_to_pcd


def group_pcd_indexes_by_color(pcd):
    result = {}
    colors = np.asarray(pcd.colors)
    colors_string = np.fromiter((color_to_string(denormalize_color(color)) for color in colors), dtype='|S256')
    unique_colors = np.unique(colors_string)
    for color in unique_colors:
        # remember that np.where returns tuple --- we have to extract array from it
        result[color] = np.where(colors_string == color)[0]

    return result


next_track_id = 0
color_to_track = {}


def segment_pcd_by_annotations(
        segmented_pcd: SegmentedPointCloud,
        cam_intrinsic: CameraIntrinsics,
        annotation,
        frame_number
) -> SegmentedPointCloud:
    all_planes = annotation.get_all_planes_for_frame(frame_number)
    image_shape = (cam_intrinsic.height, cam_intrinsic.width)
    annotated_rgb = draw_polygones(all_planes, image_shape)

    colored_pcd = load_rgb_colors_to_pcd(annotated_rgb, segmented_pcd.pcd)

    black_color = np.array([0., 0., 0.])
    black_color_str = color_to_string(black_color)
    planes_indexes = group_pcd_indexes_by_color(colored_pcd)

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

    return SegmentedPointCloud(colored_pcd, planes, unsegmented_cloud_indices)
