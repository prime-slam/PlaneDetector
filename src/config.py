import open3d as o3d


class NclNuim:
    @staticmethod
    def get_cam_intrinsic(image_shape):
        return o3d.camera.PinholeCameraIntrinsic(
            width=image_shape[1],
            height=image_shape[0],
            fx=481.20,  # X-axis focal length
            fy=-480.00,  # Y-axis focal length
            cx=319.50,  # X-axis principle point
            cy=239.50,  # Y-axis principle point
        )


class Tum:
    @staticmethod
    def get_cam_intrinsic(image_shape):
        return o3d.camera.PinholeCameraIntrinsic(
            width=image_shape[1],
            height=image_shape[0],
            fx=591.1,  # X-axis focal length
            fy=590.1,  # Y-axis focal length
            cx=331.0,  # X-axis principle point
            cy=234.0,  # Y-axis principle point
        )
