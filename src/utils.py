import open3d as o3d
import numpy as np
import cv2


def rgbd_to_pcd(rgbd_image, camera_intrinsics):
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


def draw_polygones(planes, image_shape):
    image = np.zeros((image_shape[0], image_shape[1], 3), np.uint8)
    for plane in planes:
        draw_polygone(image, plane)

    return image