import argparse
import open3d as o3d
import numpy as np
import OutlierDetector

from CVATAnnotation import CVATAnnotation
from src.detectors import O3DRansacDetector
from src.utils.annotations import draw_polygones
from src.utils.point_cloud import rgbd_to_pcd, depth_to_pcd


def create_input_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'depth_path',
        type=str,
        help='Path to depth image'
    )
    parser.add_argument(
        'annotations_path',
        type=str,
        help='Path to annotations.xml file in CVAT for video format'
    )
    parser.add_argument(
        '--annotation_frame_number',
        type=int,
        default=0,
        help='Number of frame in annotations.xml which will be used for annotations extraction. By default first '
             'frame is used '
    )

    return parser


if __name__ == '__main__':
    parser = create_input_parser()
    args = parser.parse_args()
    path_to_depth = args.depth_path
    path_to_annotations = args.annotations_path
    frame_number = args.annotation_frame_number

    depth_image = o3d.io.read_image(path_to_depth)
    image_shape = np.asarray(depth_image).shape

    annotation = CVATAnnotation(path_to_annotations)
    all_planes = annotation.get_all_planes_for_frame(frame_number)
    annotated_rgb = draw_polygones(all_planes, image_shape)
    color_image = o3d.geometry.Image(annotated_rgb)

    # Taken from https://www.doc.ic.ac.uk/~ahanda/VaFRIC/codes.html
    cam_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=image_shape[1],
        height=image_shape[0],
        fx=481.20,  # X-axis focal length
        fy=-480.00,  # Y-axis focal length
        cx=319.50,  # X-axis principle point
        cy=239.50,  # Y-axis principle point
    )

    # rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    #     color_image,
    #     depth_image,
    #     depth_scale=5000.0,
    #     depth_trunc=1000.0,
    #     convert_rgb_to_intensity=False
    # )
    # pcd = rgbd_to_pcd(rgbd_image, cam_intrinsic)
    # pcd_with_outliers = OutlierDetector.remove_planes_outliers(pcd)
    # o3d.visualization.draw_geometries([pcd_with_outliers])

    pcd = depth_to_pcd(depth_image, cam_intrinsic)
    detected_pcd = O3DRansacDetector.detect_planes(pcd)
    o3d.visualization.draw_geometries([detected_pcd])
