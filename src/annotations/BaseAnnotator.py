from abc import ABC, abstractmethod

from src.model.SegmentedPointCloud import SegmentedPointCloud


class BaseAnnotator(ABC):
    def __init__(self, path):
        self.path = path

    @abstractmethod
    def annotate(self, pcd: SegmentedPointCloud, frame_num: int) -> SegmentedPointCloud:
        pass
