import cv2
import numpy as np
import open3d as o3d
import OutlierDetector
import tkinter as tk

from CVATAnnotation import CVATAnnotation
from src.detectors import AnnotationsDetector
from src.output.PointCloudPrinter import PointCloudPrinter
from src.parser import create_input_parser, loaders, algos, metrics
from src.utils.point_cloud import depth_to_pcd


def load_annotations(
        loader,
        depth_frame_num,
        annotation,
        depth_image,
        cam_intrinsic,
        initial_pcd_transform,
        filter_outliers
):
    frame_number = loader.depth_to_rgb_index[depth_frame_num]
    result_pcd = AnnotationsDetector.segment_pcd_from_depth_by_annotations(
        depth_image,
        cam_intrinsic,
        initial_pcd_transform,
        annotation,
        frame_number
    )
    if filter_outliers:
        result_pcd = OutlierDetector.remove_planes_outliers(result_pcd)

    return result_pcd


def pick_and_print_point(pcd: o3d.geometry.PointCloud):
    pts = np.asarray(pcd.points)
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    picked = vis.get_picked_points()
    print(pts[picked[0]])


def process_frame(depth_frame_num: int, args):
    depth_image = loader.read_depth_image(depth_frame_num)
    result_pcd = None
    detected_pcd = None
    image_shape = depth_image.shape
    cam_intrinsic = loader.config.get_cam_intrinsic(image_shape)
    initial_pcd_transform = loader.config.get_initial_pcd_transform()

    if args.annotations_path is not None:
        result_pcd = load_annotations(
            loader,
            depth_frame_num,
            annotation,
            depth_image,
            cam_intrinsic,
            initial_pcd_transform,
            args.filter_annotation_outliers
        )
        # PointCloudPrinter(result_pcd.get_color_pcd_for_visualization()).save_to_ply("result.ply")
        # PointCloudPrinter(result_pcd.get_color_pcd_for_visualization()).save_to_pcd("result.pcd")

    if args.algo is not None:
        pcd = depth_to_pcd(depth_image, cam_intrinsic, initial_pcd_transform)
        detector = algos[args.algo]
        detected_pcd = detector.detect_planes(pcd)

    if args.annotations_path is not None and args.algo is not None and len(args.metric) > 0:
        for metric_name in args.metric:
            benchmark = metrics[metric_name]()
            benchmark_result = benchmark.execute(detected_pcd, result_pcd)
            print(benchmark_result)

    return result_pcd, detected_pcd


if __name__ == '__main__':
    parser = create_input_parser()
    args = parser.parse_args()
    path_to_dataset = args.dataset_path
    depth_frame_num = args.frame_num
    loader_name = args.loader

    loader = loaders[loader_name](path_to_dataset)
    annotation = CVATAnnotation(args.annotations_path)
    # visualized_pcd = o3d.geometry.PointCloud()
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # vis.add_geometry(visualized_pcd)
    while depth_frame_num < len(loader.depth_images):
        result_pcd, detected_pcd = process_frame(depth_frame_num, args)
        result_for_visualization = result_pcd.get_color_pcd_for_visualization()
        o3d.visualization.draw_geometries([result_for_visualization])
        # visualized_pcd.points = result_for_visualization.points
        # visualized_pcd.colors = result_for_visualization.colors
        # vis.add_geometry(visualized_pcd)
        # vis.run()
        # vis.poll_events()
        # vis.update_renderer()
        # input()
        depth_frame_num += 1

    # vis.destroy_window()
