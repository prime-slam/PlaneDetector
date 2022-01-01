import numpy as np

from src.model.SegmentedPlane import SegmentedPlane
from src.metrics.one_value.OneValueBenchmark import OneValueBenchmark
from src.utils.metrics import planes_intersection_pcd


def dice(plane_predicted: SegmentedPlane, plane_gt: SegmentedPlane):
    """
    Calculates DICE metric
    :param plane_predicted: predicted segmentation
    :param plane_gt: ground truth segmentation
    :return: DICE value
    """
    intersection = planes_intersection_pcd(plane_predicted, plane_gt)
    intersection_size = np.asarray(intersection.points).size
    gt_size = np.asarray(plane_gt.pcd.points).size
    predicted_size = np.asarray(plane_predicted.pcd.points).size

    return 2 * intersection_size / (predicted_size + gt_size)


class DiceBenchmark(OneValueBenchmark):
    def get_metric_name(self):
        return "Dice"

    def calculate_metric(self, plane_predicted: SegmentedPlane, plane_gt: SegmentedPlane) -> float:
        return dice(plane_predicted, plane_gt)
