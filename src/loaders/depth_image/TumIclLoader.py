from src.loaders.depth_image.CameraConfig import CameraConfig
from src.loaders.depth_image.CameraIntrinsics import CameraIntrinsics
from src.loaders.depth_image.TumLoader import TumLoader


class TumIclLoader(TumLoader):
    def provide_config(self) -> CameraConfig:
        return self.TumIclCameraConfig()

    def _get_filenames(self, rgb_path, depth_path):
        rgb_filenames, depth_filenames = super()._get_filenames(
            rgb_path,
            depth_path
        )
        normalized_rgb_filenames = TumIclLoader.__normalize_filenames(rgb_filenames)
        normalized_depth_filenames = TumIclLoader.__normalize_filenames(depth_filenames)

        return normalized_rgb_filenames, normalized_depth_filenames

    @staticmethod
    def __normalize_filenames(filenames):
        max_filename_len = max([len(filename[:-4]) for filename in filenames])
        normalized_filenames = [
            "0" * (max_filename_len - len(filename[:-4])) + filename for filename in filenames
        ]

        return normalized_filenames

    class TumIclCameraConfig(CameraConfig):
        def get_cam_intrinsic(self, image_shape=(480, 640)) -> CameraIntrinsics:
            # Taken from https://www.doc.ic.ac.uk/~ahanda/VaFRIC/codes.html
            return CameraIntrinsics(
                width=image_shape[1],
                height=image_shape[0],
                fx=481.20,  # X-axis focal length
                fy=-480.00,  # Y-axis focal length
                cx=319.50,  # X-axis principle point
                cy=239.50,  # Y-axis principle point
                factor=5000  # for the 16-bit PNG files
            )

        def get_initial_pcd_transform(self):
            return [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
