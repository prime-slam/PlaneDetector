import numpy as np
import cv2


def draw_polygone(image, plane):
    contours = np.array(plane.points).astype(dtype=np.float)
    color_tuple = tuple([int(part) for part in plane.color])
    cv2.fillPoly(image, pts=np.int32([contours]), color=color_tuple)
    return image


def draw_polygones(planes, image_shape):
    image = np.zeros((image_shape[0], image_shape[1], 3), np.uint8)
    for plane in planes:
        draw_polygone(image, plane)

    return image
