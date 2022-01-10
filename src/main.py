import numpy as np
import open3d as o3d

from src.FrameProcessor import process_frame
from src.model.CVATAnnotation import CVATAnnotation
from src.parser import create_input_parser, loaders


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
    depth_frame_num = args.frame_num
    loader_name = args.loader

    loader = loaders[loader_name](path_to_dataset)

    if args.annotations_path is not None:
        annotation = CVATAnnotation(args.annotations_path, args.annotations_start_frame)
    else:
        annotation = None
    # visualized_pcd = o3d.geometry.PointCloud()
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # vis.add_geometry(visualized_pcd)
    while depth_frame_num < len(loader.depth_images):
        result_pcd, detected_pcd = process_frame(
            loader,
            depth_frame_num,
            annotation,
            args.filter_annotation_outliers,
            args.algo,
            args.metric
        )
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
