from abc import ABC, abstractmethod

import numpy as np

from src.loaders.depth_image.CameraIntrinsics import CameraIntrinsics


class CameraConfig(ABC):
    @abstractmethod
    def get_cam_intrinsic(self, image_shape) -> CameraIntrinsics:
        pass

    @abstractmethod
    def get_initial_pcd_transform(self) -> np.array:
        pass
