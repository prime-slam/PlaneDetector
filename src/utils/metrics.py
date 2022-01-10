import numpy as np
import open3d as o3d

from src.model.SegmentedPlane import SegmentedPlane


def planes_intersection_indices(plane_left: SegmentedPlane, plane_right: SegmentedPlane) -> np.array:
    pcd_left_indices = plane_left.pcd_indices
    pcd_right_indices = plane_right.pcd_indices
    intersection_indices = np.intersect1d(pcd_left_indices, pcd_right_indices)

    return intersection_indices


def planes_union_indices(
        plane_left: SegmentedPlane,
        plane_right: SegmentedPlane,
        intersection_indices: np.array = None
) -> np.array:
    if intersection_indices is None:
        intersection_indices = planes_intersection_indices(plane_left, plane_right)

    only_left_indices = np.setdiff1d(
        plane_left.pcd_indices,
        intersection_indices
    )

    return np.concatenate((only_left_indices, plane_right.pcd_indices))


def are_nearly_overlapped(plane_predicted: SegmentedPlane, plane_gt: SegmentedPlane, required_overlap: float):
    """
    Calculate if planes are overlapped enough (80%) to be used for PP-PR metric
    :param required_overlap: overlap threshold which will b checked to say that planes overlaps
    :param plane_predicted: predicted segmentation
    :param plane_gt: ground truth segmentation
    :return: true if planes are overlapping by 80% or more, false otherwise
    """
    intersection = planes_intersection_indices(plane_predicted, plane_gt)
    intersection_size = np.asarray(intersection.points).shape[0]
    gt_size = plane_gt.pcd_indices.size
    predicted_size = plane_predicted.pcd_indices.size

    return intersection_size / predicted_size >= required_overlap and intersection_size / gt_size >= required_overlap

