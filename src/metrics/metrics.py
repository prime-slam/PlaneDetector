import numpy as np

from src.SegmentedPlane import SegmentedPlane
from src.utils.metrics import planes_intersection_pcd, planes_union_pcd


def iou(plane_left: SegmentedPlane, plane_right: SegmentedPlane):
    intersection = planes_intersection_pcd(plane_left, plane_right)
    union = planes_union_pcd(plane_left, plane_right)
    return np.asarray(intersection.points).size / np.asarray(union.points).size
