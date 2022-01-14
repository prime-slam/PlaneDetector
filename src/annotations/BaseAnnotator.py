from abc import ABC, abstractmethod

from src.model.SegmentedPointCloud import SegmentedPointCloud


class BaseAnnotator(ABC):
    def __init__(self, path, start_frame_num: int):
        self.path = path
        self.start_frame_num = start_frame_num

    @abstractmethod
    def annotate(self, pcd: SegmentedPointCloud, frame_num: int) -> SegmentedPointCloud:
        pass
