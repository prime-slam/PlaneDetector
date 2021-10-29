import argparse

import numpy as np
import open3d as o3d
import OutlierDetector

from CVATAnnotation import CVATAnnotation
from src.detectors import AnnotationsDetector, O3DRansacDetector
from src.utils.point_cloud import depth_to_pcd


def create_input_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'depth_path',
        type=str,
        help='Path to depth image'
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
    path_to_depth = args.depth_path

    depth_image = o3d.io.read_image(path_to_depth)
    result_pcd = None
    image_shape = np.asarray(depth_image).shape
    # Taken from https://www.doc.ic.ac.uk/~ahanda/VaFRIC/codes.html
    cam_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=image_shape[1],
        height=image_shape[0],
        fx=481.20,  # X-axis focal length
        fy=-480.00,  # Y-axis focal length
        cx=319.50,  # X-axis principle point
        cy=239.50,  # Y-axis principle point
    )

    path_to_annotations = args.annotations_path
    if path_to_annotations is not None:
        frame_number = args.annotation_frame_number
        annotation = CVATAnnotation(path_to_annotations)
        result_pcd = AnnotationsDetector.segment_pcd_from_depth_by_annotations(
            depth_image,
            cam_intrinsic,
            annotation,
            frame_number
        )
        if args.annotation_filter_outliers:
            result_pcd = OutlierDetector.remove_planes_outliers(result_pcd)
    else:
        pcd = depth_to_pcd(depth_image, cam_intrinsic)
        result_pcd = O3DRansacDetector.detect_planes(pcd)

    if result_pcd is None:
        print("Nothing to visualize!")
    else:
        o3d.visualization.draw_geometries([result_pcd.get_color_pcd_for_visualization()])
