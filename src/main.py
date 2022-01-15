import numpy as np
import open3d as o3d

from src.FrameProcessor import process_frame
from src.metrics.CompositeBenchmark import CompositeBenchmark
from src.parser import create_input_parser, loaders, annotators, algos, metrics


def pick_and_print_point(pcd: o3d.geometry.PointCloud):
    pts = np.asarray(pcd.points)
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    picked = vis.get_picked_points()
    print(pts[picked[0]])


if __name__ == '__main__':
    parser = create_input_parser()
    args = parser.parse_args()
    path_to_dataset = args.dataset_path
    frame_num = args.frame_num
    loader_name = args.loader

    loader = loaders[loader_name](path_to_dataset)

    hasAnnotations = args.annotations_path is not None and args.annotator is not None
    if hasAnnotations:
        annotator = annotators[args.annotator](args.annotations_path, args.annotations_start_frame)
    else:
        annotator = None

    if args.algo is not None:
        detector = algos[args.algo]()
    else:
        detector = None

    if args.metric is not None:
        benchmarks = []
        for metric_name in args.metric:
            benchmarks.append(metrics[metric_name]())
        benchmark = CompositeBenchmark(benchmarks)
    else:
        benchmark = None

    while frame_num < loader.get_frame_count():
        result_pcd, detected_pcd = process_frame(
            loader,
            frame_num,
            annotator,
            args.filter_annotation_outliers,
            detector,
            benchmark
        )
        if result_pcd is not None:
            result_for_visualization = result_pcd.get_color_pcd_for_visualization()
            o3d.visualization.draw_geometries([result_for_visualization])
        if detected_pcd is not None:
            result_for_visualization = detected_pcd.get_color_pcd_for_visualization()
            o3d.visualization.draw_geometries([result_for_visualization])

        frame_num += 1
