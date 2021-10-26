import open3d as o3d
import numpy as np

from src.detectors.O3DRansacDetector import detect_plane
from src.utils.colors import color_to_string, color_from_string
from src.utils.point_cloud import merge_pcd


def group_pcd_indexes_by_color(pcd):
    result = {}
    colors = np.asarray(pcd.colors)
    colors_string = np.fromiter((color_to_string(color) for color in colors), dtype='|S256')
    unique_colors = np.unique(colors_string)
    for color in unique_colors:
        # remember that np.where returns tuple --- we have to extract array from it
        result[color] = np.where(colors_string == color)[0]

    return result


def remove_planes_outliers(pcd):
    black_color = np.array([0., 0., 0.])
    black_color_str = color_to_string(black_color)
    planes_indexes = group_pcd_indexes_by_color(pcd)
    result_pcd = o3d.geometry.PointCloud()
    for color_str, indexes in planes_indexes.items():
        color = color_from_string(color_str)

        extracted_pcd = pcd.select_by_index(indexes)
        if color_str.decode('UTF-8') != black_color_str:
            inlier_pcd, outlier_pcd = detect_plane(extracted_pcd)
            inlier_pcd.paint_uniform_color(color)
            outlier_pcd.paint_uniform_color(black_color)
            plane_with_outliers_pcd = merge_pcd(inlier_pcd, outlier_pcd)
        else:
            extracted_pcd.paint_uniform_color(black_color)
            plane_with_outliers_pcd = extracted_pcd

        result_pcd = merge_pcd(result_pcd, plane_with_outliers_pcd)

    return result_pcd
