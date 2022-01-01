import numpy as np

from src.model.SegmentedPlane import SegmentedPlane
from src.metrics.one_value.OneValueBenchmark import OneValueBenchmark
from src.utils.metrics import planes_union_pcd, planes_intersection_pcd


def iou(plane_predicted: SegmentedPlane, plane_gt: SegmentedPlane):
    """
    Calculates Intersection-over-Union metric
    :param plane_predicted: predicted segmentation
    :param plane_gt: ground truth segmentation
    :return: IoU value
    """
    intersection = planes_intersection_pcd(plane_predicted, plane_gt)
    union = planes_union_pcd(plane_predicted, plane_gt, intersection)
    return np.asarray(intersection.points).size / np.asarray(union.points).size


class IoUBenchmark(OneValueBenchmark):
    def get_metric_name(self):
        return "IoU"

    def calculate_metric(self, plane_predicted: SegmentedPlane, plane_gt: SegmentedPlane) -> float:
        return iou(plane_predicted, plane_gt)
