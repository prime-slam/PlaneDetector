import sys
import open3d as o3d
import cv2
import numpy as np
import OutlierDetector

from CVATAnnotation import CVATAnnotation


def depth_to_pcd(rgbd_image, camera_intrinsics):
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        camera_intrinsics
    )
    pcd.transform([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    return pcd


def draw_polygone(image, plane):
    contours = np.array(plane.points)
    cv2.fillPoly(image, pts=np.int32([contours]), color=plane.color)
    return image


def draw_polygones(planes):
    image = np.zeros((480, 640, 3), np.uint8)
    for plane in planes:
        draw_polygone(image, plane)
    cv2.imwrite("out.png", image)
    return image


if __name__ == '__main__':
    path_to_depth = sys.argv[1]
    path_to_annotations = sys.argv[2]
    annotations = CVATAnnotation(path_to_annotations)
    all_planes = annotations.get_all_planes_for_frame(0)
    annotated_rgb = draw_polygones(all_planes)
    depth_image = o3d.io.read_image(path_to_depth)
    cam_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=640,
        height=480,
        fx=481.20,  # X-axis focal length
        fy=-480.00,  # Y-axis focal length
        cx=319.50,  # X-axis principle point
        cy=239.50,  # Y-axis principle point
    )
    color_image = o3d.io.read_image("out.png")
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_image,
        depth_image,
        depth_scale=5000.0,
        depth_trunc=1000.0,
        convert_rgb_to_intensity=False
    )
    pcd = depth_to_pcd(rgbd_image, cam_intrinsic)
    pcd_with_outliers = OutlierDetector.remove_planes_outliers(pcd)
    o3d.visualization.draw_geometries([pcd_with_outliers])
