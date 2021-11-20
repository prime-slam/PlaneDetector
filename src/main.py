import argparse

import numpy as np
import open3d as o3d
import OutlierDetector

from CVATAnnotation import CVATAnnotation
from src.config import Tum, NclNuim
from src.detectors import AnnotationsDetector, O3DRansacDetector
from src.loaders.tum import TumDataset
from src.metrics.multi_value.MultiValueBenchmark import MultiValueBenchmark
from src.metrics.one_value.DiceBenchmark import DiceBenchmark
from src.metrics.one_value.IoUBenchmark import IoUBenchmark
from src.utils.point_cloud import depth_to_pcd


loaders = {
    'tum': TumDataset
}


def create_input_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'loader',
        type=str,
        choices=['tum', 'icl', 'custom'],
        help='Name of loader for dataset'
    )
    parser.add_argument(
        'dataset_path',
        type=str,
        help='Path to dataset'
    )
    parser.add_argument(
        'frame_num',
        type=int,
        default=0,
        help='Depth frame number in dataset'
    )
    parser.add_argument(
        '--annotations_path',
        type=str,
        help='Path to annotations.xml file in "CVAT for video" format'
    )
    parser.add_argument(
        '--annotation_frame_number',
        type=int,
        default=0,
        help='Number of frame in annotations.xml which will be used for annotations extraction. By default first '
             'frame is used '
    )
    parser.add_argument(
        '--annotation_filter_outliers',
        action='store_true',
        help='Specify if you want to remove outliers of you annotated planes with RANSAC'
    )

    return parser


if __name__ == '__main__':
    parser = create_input_parser()
    args = parser.parse_args()
    path_to_dataset = args.dataset_path
    depth_frame_num = args.frame_num
    loader_name = args.loader

    loader = loaders[loader_name](path_to_dataset)
    depth_image_path = loader.depth_images[depth_frame_num]

    depth_image = o3d.io.read_image(depth_image_path)
    result_pcd = None
    image_shape = np.asarray(depth_image).shape
    # Taken from https://www.doc.ic.ac.uk/~ahanda/VaFRIC/codes.html
    cam_intrinsic = NclNuim.get_cam_intrinsic(image_shape)

    path_to_annotations = args.annotations_path
    if path_to_annotations is not None:
        if loader_name == 'custom':
            frame_number = args.annotation_frame_number
        else:
            frame_number = loader.depth_to_rgb_index[depth_frame_num]
        annotation = CVATAnnotation(path_to_annotations)
        result_pcd = AnnotationsDetector.segment_pcd_from_depth_by_annotations(
            depth_image,
            cam_intrinsic,
            annotation,
            frame_number
        )
        if args.annotation_filter_outliers:
            result_pcd = OutlierDetector.remove_planes_outliers(result_pcd)

        pcd = depth_to_pcd(depth_image, cam_intrinsic)
        # detected_pcd = O3DRansacDetector.detect_planes(pcd)
        #
        # iou_benchmark_result = IoUBenchmark().execute(detected_pcd, result_pcd)
        # dice_benchmark_result = DiceBenchmark().execute(detected_pcd, result_pcd)
        # multi_value_benchmark_result = MultiValueBenchmark().execute(detected_pcd, result_pcd)
        #
        # print(iou_benchmark_result)
        # print(dice_benchmark_result)
        # print(multi_value_benchmark_result)
    else:
        pcd = depth_to_pcd(depth_image, cam_intrinsic)
        result_pcd = O3DRansacDetector.detect_planes(pcd)

    if result_pcd is None:
        print("Nothing to visualize!")
    else:
        # o3d.visualization.draw_geometries([detected_pcd.get_color_pcd_for_visualization()])
        o3d.io.write_point_cloud("result.pcd", result_pcd.get_color_pcd_for_visualization())
        o3d.visualization.draw_geometries([result_pcd.get_color_pcd_for_visualization()])
