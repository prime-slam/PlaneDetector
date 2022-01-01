import os

import cv2

from scripts.rgbd_annotations.parser import create_input_parser
from src.FrameProcessor import process_frame
from src.model.CVATAnnotation import CVATAnnotation
from src.model.SegmentedPointCloud import SegmentedPointCloud
from src.assosiators.NaiveIoUAssociator import associate_segmented_point_clouds
from src.loaders.config import CameraIntrinsics
from src.parser import loaders
from src.utils.point_cloud import pcd_to_rgb_and_depth_custom


def update_track_indices(pcd: SegmentedPointCloud, matches: dict):
    for plane in pcd.planes:
        plane.track_id = matches[plane.track_id]


def save_frame(
        pcd: SegmentedPointCloud,
        frame_num: int,
        output_path: str,
        camera_intrinsics: CameraIntrinsics,
        initial_pcd_transform
):
    image_path = os.path.join(output_path, "{}.png".format(frame_num))
    rgb_image, _ = pcd_to_rgb_and_depth_custom(
        pcd.get_color_pcd_for_visualization(),
        camera_intrinsics,
        initial_pcd_transform
    )
    cv2.imwrite(image_path, rgb_image)


if __name__ == "__main__":
    parser = create_input_parser()
    args = parser.parse_args()
    path_to_dataset = args.dataset_path
    output_path = args.output_path
    start_depth_frame_num = args.frame_num
    loader_name = args.loader
    annotations_path = args.annotations_path

    loader = loaders[loader_name](path_to_dataset)
    depth_image = loader.read_depth_image(0)
    cam_intrinsic = loader.config.get_cam_intrinsic(depth_image.shape)
    initial_pcd_transform = loader.config.get_initial_pcd_transform()

    annotations = []
    annotations_ranges = []

    for file in os.listdir(annotations_path):
        filepath = os.path.join(annotations_path, file)
        annotation = CVATAnnotation(filepath)
        annotations.append(annotation)

        # If have previous range then fix it to exclude this range
        if len(annotations_ranges) > 0:
            last_annotation_range = annotations_ranges[-1]
            annotations_ranges[-1] = (last_annotation_range[0], min(last_annotation_range[1], annotation.max_frame_id - 1))

        annotations_ranges.append((annotation.min_frame_id, annotation.max_frame_id))

    track_indices_matches = None
    previous_pcd = None

    for frame_num in range(start_depth_frame_num, len(loader.depth_images)):
        annotation_index = None
        for index, annotation_range in enumerate(annotations_ranges):
            min_frame, max_frame = annotation_range
            if min_frame <= frame_num <= max_frame:
                annotation_index = index
                break

        if annotation_index is None:
            continue

        result_pcd, _ = process_frame(
            loader,
            frame_num,
            annotations[annotation_index],
            args.filter_annotation_outliers,
            algo=None,
            metric=None
        )

        # First frame of not first annotation have to be associated with previous pcd
        if frame_num == annotations_ranges[annotation_index][0] and previous_pcd is not None:
            track_indices_matches = associate_segmented_point_clouds(previous_pcd, result_pcd)
        elif track_indices_matches is not None:
            update_track_indices(result_pcd, track_indices_matches)

        previous_pcd = result_pcd

        save_frame(result_pcd, frame_num, output_path, cam_intrinsic, initial_pcd_transform)
        print("Done!")
        break
