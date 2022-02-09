import os

import cv2
import numpy as np
import open3d as o3d

from scripts.rgbd_annotations.parser import create_input_parser
from src.FrameProcessor import process_frame
from src.annotations.cvat.CVATAnnotator import CVATAnnotator
from src.loaders.depth_image.CameraIntrinsics import CameraIntrinsics
from src.model.SegmentedPlane import SegmentedPlane
from src.model.SegmentedPointCloud import SegmentedPointCloud
from src.assosiators.NaiveIoUAssociator import associate_segmented_point_clouds
from src.parser import loaders
from src.utils.point_cloud import pcd_to_rgb_and_depth_custom


last_used_track_id = -1

last_associate_matches: dict = None


def print_segment_tracks(original_track_to_unified: dict, prev_segment: int):
    print("{")
    for original_track in original_track_to_unified.keys():
        match_str = "   {}:".format(original_track + 1)
        if last_associate_matches is not None and original_track in last_associate_matches:
            match_str += " ({0}, {1})".format(prev_segment + 1, last_associate_matches[original_track] + 1)
        match_str += ","
        print(match_str)
    print("},")


def update_track_indices(pcd: SegmentedPointCloud, matches: dict, resolve_history_for_part: dict, is_first_part: bool):
    global last_used_track_id
    for plane in pcd.planes:
        if plane.track_id not in matches:
            if is_first_part:
                matches[plane.track_id] = plane.track_id
                resolve_history_for_part[plane.track_id] = plane.track_id
                last_used_track_id = max(last_used_track_id, plane.track_id)
            else:
                matches[plane.track_id] = last_used_track_id + 1
                resolve_history_for_part[plane.track_id] = last_used_track_id + 1
                last_used_track_id += 1

        plane.track_id = matches[plane.track_id]


def resolve_tracks(predefined_track_indices_matches: list, current_annot_segment: int, resolve_history: [dict]) -> dict:
    new_track_matches = {}
    for original_track, match_pair in predefined_track_indices_matches[current_annot_segment - 1].items():
        reference_annot_segment, reference_track = match_pair

        # Fix enumeration from 1 which was used for labeler and cvat ui
        reference_annot_segment -= 1
        reference_track -= 1
        original_track -= 1

        unified_track = resolve_history[reference_annot_segment][reference_track]
        new_track_matches[original_track] = unified_track
        resolve_history[current_annot_segment][original_track] = unified_track

    return new_track_matches


def build_track_matches(previous_pcd, result_pcd: SegmentedPointCloud):
    global last_used_track_id
    global last_associate_matches
    new_track_matches = {}
    for plane in result_pcd.planes:
        new_track_matches[plane.track_id] = last_used_track_id + 1
        plane.track_id = last_used_track_id + 1
        last_used_track_id += 1

    associate_matches = associate_segmented_point_clouds(previous_pcd, result_pcd)
    last_associate_matches = {}
    for original_track, unified_track in new_track_matches.items():
        if unified_track in associate_matches:
            new_track_matches[original_track] = associate_matches[unified_track]
            last_associate_matches[original_track] = associate_matches[unified_track]

    return new_track_matches


def update_planes_colors(result_pcd: SegmentedPointCloud, track_to_color: dict) -> dict:
    for plane in result_pcd.planes:
        if plane.track_id in track_to_color:
            plane.set_color(track_to_color[plane.track_id])
        else:
            track_to_color[plane.track_id] = plane.color

    return track_to_color


def save_frame(
        pcd: SegmentedPointCloud,
        filename: str,
        output_path: str,
        camera_intrinsics: CameraIntrinsics,
        initial_pcd_transform
):
    image_path = os.path.join(output_path, "{}.png".format(filename))
    rgb_image, _ = pcd_to_rgb_and_depth_custom(
        pcd.get_color_pcd_for_visualization(),
        camera_intrinsics,
        initial_pcd_transform
    )
    cv2.imwrite(image_path, rgb_image)


def pick_and_print_point(pcd: o3d.geometry.PointCloud):
    pts = np.asarray(pcd.points)
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    picked = vis.get_picked_points()
    print(pts[picked[0]])
    print(pts[picked[1]])


if __name__ == "__main__":
    parser = create_input_parser()
    args = parser.parse_args()
    path_to_dataset = args.dataset_path
    output_path = args.output_path
    start_depth_frame_num = args.frame_num
    loader_name = args.loader
    annotations_path = args.annotations_path

    loader = loaders[loader_name](path_to_dataset)
    cam_intrinsic = loader.config.get_cam_intrinsic()
    initial_pcd_transform = loader.config.get_initial_pcd_transform()

    annotators = []
    annotations_ranges = []

    annotation_files = os.listdir(annotations_path)
    annotation_files = sorted(annotation_files, key=lambda x: int(x.split("-")[0]))
    for file in annotation_files:
        filepath = os.path.join(annotations_path, file)
        start_frame_annot = int(file.split("-")[0])
        annotator = CVATAnnotator(filepath, start_frame_annot)
        annotators.append(annotator)

        # If have previous range then fix it to exclude this range
        if len(annotations_ranges) > 0:
            last_annotation_range = annotations_ranges[-1]
            annotations_ranges[-1] = (
                last_annotation_range[0],
                min(last_annotation_range[1], annotator.annotation.get_min_frame_id() - 1)
            )

        annotations_ranges.append((annotator.annotation.get_min_frame_id(), annotator.annotation.get_max_frame_id()))

    resolve_history = [{} for _ in annotations_ranges]
    original_track_to_unified = {}
    annotation_index = None
    # predefined_track_indices_matches = [
    #     {
    #         35: (1, 34),
    #         2: (1, 31),
    #         1: (1, 1),
    #         33: (1, 9)
    #     },  # for the 2-1 match
    #     {
    #         1: (2, 33),
    #         2: (2, 29),
    #         3: (1, 34)
    #     }   # for the 3-2 match --- track in 3d peace : (matched piece, track in matched piece)
    #     # if the track is new in this part, than just remove it from list
    # ]
    predefined_track_indices_matches = None
    previous_pcd = None

    track_to_color = {}

    for frame_num in range(start_depth_frame_num, loader.get_frame_count()):
        annotation_frame_num = loader.depth_to_rgb_index[frame_num]
        annotation_index = None
        for index, annotation_range in enumerate(annotations_ranges):
            min_frame, max_frame = annotation_range
            if min_frame <= annotation_frame_num <= max_frame:
                annotation_index = index
                break

        if annotation_index is None:
            continue

        result_pcd, _ = process_frame(
            loader,
            frame_num,
            annotators[annotation_index],
            args.filter_annotation_outliers,
            detector=None,
            benchmark=None
        )

        is_first_part = annotation_index == 0

        # First frame of not first annotation have to be associated with previous pcd
        if annotation_frame_num == annotations_ranges[annotation_index][0] and previous_pcd is not None:
            if predefined_track_indices_matches is None:
                prev_annot_index = annotation_index - 1
                if prev_annot_index != 0:
                    print_segment_tracks(original_track_to_unified, prev_annot_index)
                original_track_to_unified = build_track_matches(previous_pcd, result_pcd)
            else:
                original_track_to_unified = resolve_tracks(
                    predefined_track_indices_matches,
                    annotation_index,
                    resolve_history
                )
                update_track_indices(
                    result_pcd,
                    original_track_to_unified,
                    resolve_history[annotation_index],
                    is_first_part=False
                )
        else:
            update_track_indices(
                result_pcd,
                original_track_to_unified,
                resolve_history[annotation_index],
                is_first_part=is_first_part
            )

        track_to_color = update_planes_colors(result_pcd, track_to_color)

        previous_pcd = result_pcd

        output_filename = os.path.split(loader.depth_images[frame_num])[-1]
        output_filename = ".".join(output_filename.split(".")[:-1])
        # pick_and_print_point(result_pcd.get_color_pcd_for_visualization())
        # PointCloudPrinter(result_pcd.get_color_pcd_for_visualization()).save_to_pcd(output_filename + ".pcd")
        # cv2.imwrite(output_filename + "rgb.png", cv2.imread(loader.rgb_images[loader.depth_to_rgb_index[frame_num]]))
        # cv2.imwrite(output_filename + "depth.png", cv2.imread(loader.depth_images[frame_num]))

        def filter_tum_planes(plane: SegmentedPlane) -> bool:
            result_points = np.asarray(result_pcd.pcd.points)
            plane_points = result_points[plane.pcd_indices]
            if plane.pcd_indices.size == 0:
                return False
            distances_from_cam = np.sqrt(np.sum(plane_points ** 2, axis=-1))
            mean_distance = np.mean(distances_from_cam)
            # 5 for TUM pioneer, 4 for TUM desk, long office
            is_zero_dominate = plane.zero_depth_pcd_indices.size / 4 > plane.pcd_indices.size
            # print("Distance: {0}. Size of zero: {1}. Size of plane: {2}".format(
            #     mean_distance,
            #     plane.zero_depth_pcd_indices.size,
            #     plane.pcd_indices.size
            # ))

            # 3 for TUM pioneer, 3.5 for TUM desk,long_office
            return mean_distance < 3.5 and not is_zero_dominate

        result_pcd.filter_planes(filter_tum_planes)
        save_frame(result_pcd, output_filename, output_path, cam_intrinsic, initial_pcd_transform)
        # if frame_num == 3:
        #     break

    print_segment_tracks(original_track_to_unified, annotation_index)
