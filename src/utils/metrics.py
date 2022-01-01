import numpy as np
import open3d as o3d

from src.model.SegmentedPlane import SegmentedPlane


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


def planes_union_pcd(
        plane_left: SegmentedPlane,
        plane_right: SegmentedPlane,
        intersection_pcd: o3d.geometry.PointCloud = None
) -> o3d.geometry.PointCloud:
    if intersection_pcd is None:
        intersection_pcd = planes_intersection_pcd(plane_left, plane_right)

    pcd_only_left = pcd_delta(plane_left.pcd, intersection_pcd)
    pcd_only_left_points = np.asarray(pcd_only_left.points)

    union_points = np.concatenate((pcd_only_left_points, plane_right.pcd.points), axis=0)
    res = o3d.geometry.PointCloud()
    res.points = o3d.utility.Vector3dVector(union_points)

    return res


def are_nearly_overlapped(plane_predicted: SegmentedPlane, plane_gt: SegmentedPlane, required_overlap: float):
    """
    Calculate if planes are overlapped enough (80%) to be used for PP-PR metric
    :param required_overlap: overlap threshold which will b checked to say that planes overlaps
    :param plane_predicted: predicted segmentation
    :param plane_gt: ground truth segmentation
    :return: true if planes are overlapping by 80% or more, false otherwise
    """
    intersection = planes_intersection_pcd(plane_predicted, plane_gt)
    intersection_size = np.asarray(intersection.points).size
    gt_size = np.asarray(plane_gt.pcd.points).size
    predicted_size = np.asarray(plane_predicted.pcd.points).size

    return intersection_size / predicted_size >= required_overlap and intersection_size / gt_size >= required_overlap

