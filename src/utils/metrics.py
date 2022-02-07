import numpy as np
import open3d as o3d

from src.model.SegmentedPlane import SegmentedPlane


def get_dictionary_indices_of_current_label(indices_array: np.ndarray) -> dict:
    dictionary = {}

    for index, value in enumerate(indices_array):
        if value in dictionary:
            dictionary[value] = np.append(dictionary[value], index)
        else:
            dictionary[value] = np.array(index)

    return dictionary


def planes_intersection_indices(plane_left: np.ndarray, plane_right: np.ndarray) -> np.array:
    return np.intersect1d(plane_left, plane_right)


def planes_union_indices(
        plane_left: np.ndarray,
        plane_right: np.ndarray,
) -> np.array:
    return np.union1d(plane_left, plane_right)


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
