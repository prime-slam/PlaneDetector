from abc import ABC, abstractmethod

from src.model.SegmentedPointCloud import SegmentedPointCloud


class BaseBenchmark(ABC):
    @abstractmethod
    def execute(
        self, cloud_predicted: SegmentedPointCloud, cloud_gt: SegmentedPointCloud
    ):
        pass
