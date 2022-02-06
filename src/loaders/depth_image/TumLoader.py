import os

from src.loaders.depth_image.CameraConfig import CameraConfig
from src.loaders.depth_image.CameraIntrinsics import CameraIntrinsics
from src.loaders.depth_image.ImageLoader import ImageLoader


class TumLoader(ImageLoader):
    def __init__(self, path):
        super().__init__(path)

    def _provide_config(self) -> CameraConfig:
        return self.TumCameraConfig()

    def _provide_rgb_and_depth_path(self, path):
        depth_path = os.path.join(path, "depth")
        rgb_path = os.path.join(path, "rgb")

        return rgb_path, depth_path

    @staticmethod
    def __filenames_sorted_mapper(filename):
        return int(filename.split(".")[0])

    def _provide_filenames(self, rgb_path, depth_path) -> (list, list):
        rgb_filenames = os.listdir(rgb_path)
        depth_filenames = os.listdir(depth_path)

        rgb_filenames = sorted(rgb_filenames, key=TumLoader.__filenames_sorted_mapper)
        depth_filenames = sorted(depth_filenames, key=TumLoader.__filenames_sorted_mapper)

        return rgb_filenames, depth_filenames

    def _match_rgb_with_depth(self, rgb_filenames, depth_filenames) -> list:
        depth_to_rgb_index = []
        rgb_index = 0
        depth_index = 0
        prev_delta = float('inf')
        while depth_index < len(depth_filenames) and rgb_index < len(rgb_filenames):
            rgb_timestamp = float(rgb_filenames[rgb_index][:-4])
            depth_timestamp = float(depth_filenames[depth_index][:-4])
            delta = abs(depth_timestamp - rgb_timestamp)

            if rgb_timestamp < depth_timestamp:
                prev_delta = delta
                rgb_index += 1
                continue

            if prev_delta < delta:
                depth_to_rgb_index.append(rgb_index - 1)
            else:
                depth_to_rgb_index.append(rgb_index)

            depth_index += 1

        # Fix case when the last timestamp was for depth img
        while depth_index < len(depth_filenames):
            depth_to_rgb_index.append(rgb_index - 1)
            depth_index += 1

        return depth_to_rgb_index

    class TumCameraConfig(CameraConfig):
        def get_cam_intrinsic(self, image_shape=(480, 640)) -> CameraIntrinsics:
            return CameraIntrinsics(
                width=image_shape[1],
                height=image_shape[0],
                fx=591.1,  # X-axis focal length
                fy=590.1,  # Y-axis focal length
                cx=331.0,  # X-axis principle point
                cy=234.0,  # Y-axis principle point
                factor=5000  # for the 16-bit PNG files
            )

        def get_initial_pcd_transform(self):
            return [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
