from abc import ABC, abstractmethod

from src.model.SegmentedPointCloud import SegmentedPointCloud


class BaseDetector(ABC):
    @abstractmethod
    def detect_planes(self, segmented_pcd: SegmentedPointCloud) -> SegmentedPointCloud:
        pass
