import numpy as np

from src.metrics.one_value.OneValueBenchmark import OneValueBenchmark
from src.model.SegmentedPlane import SegmentedPlane
from src.utils.metrics import (
    planes_union_indices,
    planes_intersection_indices,
    get_dictionary_indices_of_current_label,
)


def iou(plane_predicted: np.ndarray, plane_gt: np.ndarray) -> np.float64:
    """
    Calculates Intersection-over-Union metric
    :param plane_predicted: predicted segmentation
    :param plane_gt: ground truth segmentation
    :return: IoU value
    """
    intersection = planes_intersection_indices(plane_predicted, plane_gt)
    union = planes_union_indices(plane_predicted, plane_gt)
    return intersection.size / union.size


def iou_mean(
    point_cloud_predicted: np.ndarray, point_cloud_gt: np.ndarray
) -> np.float64:
    plane_predicted_dict = get_dictionary_indices_of_current_label(
        point_cloud_predicted
    )
    plane_gt_dic = get_dictionary_indices_of_current_label(point_cloud_gt)

    iou_mean_array = np.empty((1, 0), np.float64)
    for key, indices in enumerate(point_cloud_predicted):
        if key in plane_gt_dic:
            iou_mean_array = np.append(
                iou_mean_array, iou(plane_predicted_dict[key], plane_gt_dic[key])
            )

    if iou_mean_array.size == 0:
        return 0
    return iou_mean_array.mean()


class IoUBenchmark(OneValueBenchmark):
    def _calculate_metric(
        self, plane_predicted: SegmentedPlane, plane_gt: SegmentedPlane
    ) -> float:
        pass

    def _get_metric_name(self):
        return "IoU"

    def calculate_cumulative_metric(
        self,
        point_cloud: np.array,
        point_cloud_predicted: np.array,
        point_cloud_gt: np.array,
    ) -> np.float64:
        return iou_mean(point_cloud_predicted, point_cloud_gt)
