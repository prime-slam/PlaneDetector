import open3d as o3d


class CameraIntrinsics:
    def __init__(self, width, height, fx, fy, cx, cy):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.open3dIntrinsics = o3d.camera.PinholeCameraIntrinsic(
            width=width,
            height=height,
            fx=fx,  # X-axis focal length
            fy=fy,  # Y-axis focal length
            cx=cx,  # X-axis principle point
            cy=cy  # Y-axis principle point
        )


class IclNuim:
    @staticmethod
    def get_cam_intrinsic(image_shape) -> CameraIntrinsics:
        # Taken from https://www.doc.ic.ac.uk/~ahanda/VaFRIC/codes.html
        return CameraIntrinsics(
            width=image_shape[1],
            height=image_shape[0],
            fx=481.20,  # X-axis focal length
            fy=-480.00,  # Y-axis focal length
            cx=319.50,  # X-axis principle point
            cy=239.50  # Y-axis principle point
        )

    @staticmethod
    def get_initial_pcd_transform():
        return [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]


class Tum:
    @staticmethod
    def get_cam_intrinsic(image_shape) -> CameraIntrinsics:
        return CameraIntrinsics(
            width=image_shape[1],
            height=image_shape[0],
            fx=591.1,  # X-axis focal length
            fy=590.1,  # Y-axis focal length
            cx=331.0,  # X-axis principle point
            cy=234.0  # Y-axis principle point
        )

    @staticmethod
    def get_initial_pcd_transform():
        return [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
