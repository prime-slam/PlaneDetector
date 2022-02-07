from abc import abstractmethod, ABC

import numpy as np

from src.metrics.BaseBenchmark import BaseBenchmark
from src.model.SegmentedPlane import SegmentedPlane
from src.model.SegmentedPointCloud import SegmentedPointCloud


class OneValueBenchmarkResult:
    def __init__(
        self,
        not_predicted: list,
        predicted_ghost: list,
        predicted: dict,
        metric_name: str,
    ):
        self.missed = not_predicted
        self.predicted_ghost = predicted_ghost
        self.predicted = predicted
        self.metric_name = metric_name

    def __str__(self):
        return (
            f"Results of '{self.metric_name}' metric\n"
            f"Predicted: {self.predicted}\n"
            f"Missed: {self.missed}\n"
            f"Predicted ghosts: {self.predicted_ghost}\n"
            f"Average metric: {self.average_metric}"
        )

    @property
    def average_metric(self):
        return np.average(np.array(list(self.predicted.values()))[:, 1].astype(float))


class OneValueBenchmark(BaseBenchmark):
    @property
    def metric_name(self):
        return self._get_metric_name()

    @abstractmethod
    def _get_metric_name(self):
        pass

    @abstractmethod
    def _calculate_metric(
        self, plane_predicted: SegmentedPlane, plane_gt: SegmentedPlane
    ) -> float:
        pass

    @abstractmethod
    def calculate_cumulative_metric(
        self,
        point_cloud: np.array,
        point_cloud_predicted: np.array,
        point_cloud_gt: np.array,
    ):
        pass

    def execute(
        self, cloud_predicted: SegmentedPointCloud, cloud_gt: SegmentedPointCloud
    ):
        not_predicted = []
        predicted_ghost = []
        predicted = {}

        for predicted_plane in cloud_predicted.planes:
            max_metric = 0
            max_metric_gt_plane = None
            for gt_plane in cloud_gt.planes:
                metric_value = self._calculate_metric(predicted_plane, gt_plane)
                if metric_value > max_metric:
                    max_metric_gt_plane = gt_plane
                    max_metric = metric_value

            if max_metric < 0.1:
                predicted_ghost.append(predicted_plane)
            else:
                predicted[max_metric_gt_plane] = (predicted_plane, max_metric)

        for gt_plane in cloud_gt.planes:
            if gt_plane not in predicted.keys():
                not_predicted.append(gt_plane)

        return OneValueBenchmarkResult(
            not_predicted, predicted_ghost, predicted, self.metric_name
        )
