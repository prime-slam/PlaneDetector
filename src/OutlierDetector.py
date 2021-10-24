import open3d as o3d
import numpy as np


def color_to_string(color_arr):
    # np.array2string is critically slow
    return "{0},{1},{2}".format(color_arr[0], color_arr[1], color_arr[2])


def color_from_string(color_str):
    return np.fromiter((float(channel_str) for channel_str in color_str.decode('UTF-8').split(',')), dtype=np.float64)


def group_pcd_indexes_by_color(pcd):
    result = {}
    colors = np.asarray(pcd.colors)
    colors_string = np.fromiter((color_to_string(color) for color in colors), dtype='|S256')
    unique_colors = np.unique(colors_string)
    for color in unique_colors:
        result[color] = np.where(colors_string == color)

    return result


def extract_pcd_by_indexes(source_pcd, indexes):
    source_points = np.asarray(source_pcd.points)
    dest_points = source_points[indexes]
    dest_pcd = o3d.geometry.PointCloud()
    dest_pcd.points = o3d.utility.Vector3dVector(dest_points)

    return dest_pcd


def detect_plane(pcd):
    _, inliers = pcd.segment_plane(
        distance_threshold=0.01,
        ransac_n=3,
        num_iterations=1000
    )
    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    return inlier_cloud, outlier_cloud


def merge_pcd(pcd_left, pcd_right):
    pcd_left_points = np.asarray(pcd_left.points)
    pcd_right_points = np.asarray(pcd_right.points)
    pcd_res_points = np.concatenate((pcd_left_points, pcd_right_points), axis=0)
    pcd_left_colors = np.asarray(pcd_left.colors)
    pcd_right_colors = np.asarray(pcd_right.colors)
    pcd_res_colors = np.concatenate((pcd_left_colors, pcd_right_colors), axis=0)
    pcd_res = o3d.geometry.PointCloud()
    pcd_res.points = o3d.utility.Vector3dVector(pcd_res_points)
    pcd_res.colors = o3d.utility.Vector3dVector(pcd_res_colors)

    return pcd_res


def remove_planes_outliers(pcd):
    black_color = np.array([0., 0., 0.])
    black_color_str = color_to_string(black_color)
    planes_indexes = group_pcd_indexes_by_color(pcd)
    result_pcd = o3d.geometry.PointCloud()
    for color_str, indexes in planes_indexes.items():
        color = color_from_string(color_str)

        extracted_pcd = extract_pcd_by_indexes(pcd, indexes)
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
