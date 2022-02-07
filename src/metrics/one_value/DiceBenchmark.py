from src.model.SegmentedPlane import SegmentedPlane
from src.metrics.one_value.OneValueBenchmark import OneValueBenchmark
from src.utils.metrics import planes_intersection_indices


def dice(plane_predicted: SegmentedPlane, plane_gt: SegmentedPlane):
    """
    Calculates DICE metric
    :param plane_predicted: predicted segmentation
    :param plane_gt: ground truth segmentation
    :return: DICE value
    """
    intersection = planes_intersection_indices(plane_predicted, plane_gt)
    intersection_size = intersection.size
    gt_size = plane_gt.pcd_indices.size
    predicted_size = plane_predicted.pcd_indices.size

    return 2 * intersection_size / (predicted_size + gt_size)


class DiceBenchmark(OneValueBenchmark):
    def _get_metric_name(self):
        return "Dice"

    def _calculate_metric(
        self, plane_predicted: SegmentedPlane, plane_gt: SegmentedPlane
    ) -> float:
        return dice(plane_predicted, plane_gt)
