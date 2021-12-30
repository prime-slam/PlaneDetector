import open3d as o3d
import numpy as np

from src.SegmentedPlane import SegmentedPlane
from src.SegmentedPointCloud import SegmentedPointCloud
from src.output.PointCloudPrinter import PointCloudPrinter
from src.utils.annotations import draw_polygones
from src.utils.colors import color_to_string
from src.utils.point_cloud import rgbd_to_pcd, rgb_and_depth_to_pcd_custom


def group_pcd_indexes_by_color(pcd):
    result = {}
    colors = np.asarray(pcd.colors)
    colors_string = np.fromiter((color_to_string(color) for color in colors), dtype='|S256')
    unique_colors = np.unique(colors_string)
    for color in unique_colors:
        # remember that np.where returns tuple --- we have to extract array from it
        result[color] = np.where(colors_string == color)[0]

    return result


def segment_pcd_from_depth_by_annotations(
        depth_image,
        cam_intrinsic,
        initial_pcd_transform,
        annotation,
        frame_number
) -> SegmentedPointCloud:
    all_planes = annotation.get_all_planes_for_frame(frame_number)
    image_shape = depth_image.shape
    annotated_rgb = draw_polygones(all_planes, image_shape)

    pcd = rgb_and_depth_to_pcd_custom(
        annotated_rgb,
        depth_image,
        cam_intrinsic,
        initial_pcd_transform
    )
    # rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    #     o3d.geometry.Image(annotated_rgb),
    #     o3d.geometry.Image(depth_image),
    #     depth_scale=5000.0,
    #     depth_trunc=1000.0,
    #     convert_rgb_to_intensity=False
    # )
    # pcd = rgbd_to_pcd(rgbd_image, cam_intrinsic, initial_pcd_transform)

    black_color = np.array([0., 0., 0.])
    black_color_str = color_to_string(black_color)
    planes_indexes = group_pcd_indexes_by_color(pcd)

    planes = []
    next_track_id = 0
    track_colors = {}
    unsegmented_cloud = None
    for color_str, indexes in planes_indexes.items():
        extracted_pcd = pcd.select_by_index(indexes)
        color_decoded = color_str.decode('UTF-8')
        if color_decoded == black_color_str:
            unsegmented_cloud = extracted_pcd
        else:
            if color_decoded in track_colors:
                track_id = track_colors[color_decoded]
            else:
                track_id = next_track_id
                track_colors[color_decoded] = track_id
                next_track_id += 1

            planes.append(SegmentedPlane(extracted_pcd, track_id))

    return SegmentedPointCloud(planes, unsegmented_cloud)
