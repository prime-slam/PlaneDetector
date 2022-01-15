from src.model.SegmentedPlane import SegmentedPlane
from src.metrics.one_value.OneValueBenchmark import OneValueBenchmark
from src.utils.metrics import planes_union_indices, planes_intersection_indices


def iou(plane_predicted: SegmentedPlane, plane_gt: SegmentedPlane):
    """
    Calculates Intersection-over-Union metric
    :param plane_predicted: predicted segmentation
    :param plane_gt: ground truth segmentation
    :return: IoU value
    """
    intersection = planes_intersection_indices(plane_predicted, plane_gt)
    union = planes_union_indices(plane_predicted, plane_gt, intersection)
    return intersection.size / union.size


class IoUBenchmark(OneValueBenchmark):
    def _get_metric_name(self):
        return "IoU"

    def _calculate_metric(self, plane_predicted: SegmentedPlane, plane_gt: SegmentedPlane) -> float:
        return iou(plane_predicted, plane_gt)
