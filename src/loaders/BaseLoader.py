from abc import ABC, abstractmethod

from src.model.SegmentedPointCloud import SegmentedPointCloud


class BaseLoader(ABC):
    def __init__(self, path):
        self.path = path

    @abstractmethod
    def read_pcd(self, frame_num) -> SegmentedPointCloud:
        pass

    @abstractmethod
    def get_frame_count(self) -> int:
        pass
