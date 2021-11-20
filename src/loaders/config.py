import open3d as o3d


class IclNuim:
    @staticmethod
    def get_cam_intrinsic(image_shape):
        # Taken from https://www.doc.ic.ac.uk/~ahanda/VaFRIC/codes.html
        return o3d.camera.PinholeCameraIntrinsic(
            width=image_shape[1],
            height=image_shape[0],
            fx=481.20,  # X-axis focal length
            fy=-480.00,  # Y-axis focal length
            cx=319.50,  # X-axis principle point
            cy=239.50,  # Y-axis principle point
        )

    @staticmethod
    def get_initial_pcd_transform():
        return [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]


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

    @staticmethod
    def get_initial_pcd_transform():
        return [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
