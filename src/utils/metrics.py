import numpy as np
import open3d as o3d

from src.SegmentedPlane import SegmentedPlane


def planes_intersection_pcd(plane_left: SegmentedPlane, plane_right: SegmentedPlane) -> o3d.geometry.PointCloud:
    pcd_left_points = np.asarray(plane_left.pcd.points)
    pcd_right_points = np.asarray(plane_right.pcd.points)

    nrows, ncols = pcd_left_points.shape
    dtype = {'names': ['f{}'.format(i) for i in range(ncols)],
             'formats': ncols * [pcd_left_points.dtype]}

    intersection_points = np.intersect1d(pcd_left_points.view(dtype), pcd_right_points.view(dtype))

    intersection_points = intersection_points.view(pcd_left_points.dtype).reshape(-1, ncols)
    res = o3d.geometry.PointCloud()
    res.points = o3d.utility.Vector3dVector(intersection_points)

    return res


def pcd_delta(pcd_left: o3d.geometry.PointCloud, pcd_right: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    pcd_right_points = np.asarray(pcd_right.points)
    pcd_left_points = np.asarray(pcd_left.points)

    if pcd_right_points.size == 0:
        return pcd_left
    elif pcd_left_points.size == 0:
        return o3d.geometry.PointCloud()

    nrows, ncols = pcd_left_points.shape
    dtype = {'names': ['f{}'.format(i) for i in range(ncols)],
             'formats': ncols * [pcd_left_points.dtype]}

    delta_indices = ~np.in1d(pcd_left_points.view(dtype), pcd_right_points.view(dtype))

    pcd_left_minus_right = pcd_left_points.view(dtype)[delta_indices].view(pcd_left_points.dtype).reshape(-1, ncols)
    res = o3d.geometry.PointCloud()
    res.points = o3d.utility.Vector3dVector(pcd_left_minus_right)

    return res


def planes_union_pcd(plane_left: SegmentedPlane, plane_right: SegmentedPlane) -> o3d.geometry.PointCloud:
    intersection = planes_intersection_pcd(plane_left, plane_right)
    pcd_only_left = pcd_delta(plane_left.pcd, intersection)
    pcd_only_left_points = np.asarray(pcd_only_left.points)

    union_points = np.concatenate((pcd_only_left_points, plane_right.pcd.points), axis=0)
    res = o3d.geometry.PointCloud()
    res.points = o3d.utility.Vector3dVector(union_points)

    return res
