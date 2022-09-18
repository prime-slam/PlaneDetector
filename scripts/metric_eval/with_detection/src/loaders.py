import math
import os
from abc import abstractmethod

import cv2
import numpy as np


class CameraIntrinsics:
    def __init__(self, width: int, height: int, fx: float, fy: float, cx: float, cy: float, factor: int):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.factor = factor


def depth_to_pcd(depth_image: np.array, camera_intrinsics: CameraIntrinsics):
    image_height, image_width = depth_image.shape[:2]
    pcd_points = np.zeros((image_height * image_width, 3))

    column_indices = np.tile(np.arange(image_width), (image_height, 1)).flatten()
    row_indices = np.transpose(np.tile(np.arange(image_height), (image_width, 1))).flatten()

    pcd_points[:, 2] = depth_image.flatten() / camera_intrinsics.factor
    pcd_points[:, 0] = (column_indices - camera_intrinsics.cx) * pcd_points[:, 2] / camera_intrinsics.fx
    pcd_points[:, 1] = (row_indices - camera_intrinsics.cy) * pcd_points[:, 2] / camera_intrinsics.fy

    return pcd_points


class ImageLoader:
    def __init__(self, path):
        self.cam_intrinsics = self._provide_intrinsics()

        rgb_path, depth_path = self._provide_rgb_and_depth_path(path)

        rgb_filenames, depth_filenames = self._provide_filenames(rgb_path, depth_path)

        self.depth_images = [os.path.join(depth_path, filename) for filename in depth_filenames]
        self.rgb_images = [os.path.join(rgb_path, filename) for filename in rgb_filenames]
        self.depth_to_rgb_index = self._match_rgb_with_depth(rgb_filenames, depth_filenames)

    def get_frame_count(self) -> int:
        return len(self.depth_images)

    @abstractmethod
    def _provide_intrinsics(self) -> CameraIntrinsics:
        pass

    @abstractmethod
    def _provide_rgb_and_depth_path(self, path: str) -> (str, str):
        pass

    @abstractmethod
    def _match_rgb_with_depth(self, rgb_filenames, depth_filenames) -> list:
        pass

    @abstractmethod
    def _provide_filenames(self, rgb_path, depth_path) -> (list, list):
        pass

    def read_depth_image(self, frame_num) -> np.array:
        depth_frame_path = self.depth_images[frame_num]
        return cv2.imread(depth_frame_path, cv2.IMREAD_ANYDEPTH)

    def read_pcd(self, frame_num) -> np.array:
        depth_image = self.read_depth_image(frame_num)
        pcd = depth_to_pcd(depth_image, self.cam_intrinsics)

        return pcd


class TumLoader(ImageLoader):
    def __init__(self, path):
        super().__init__(path)

    def _provide_intrinsics(self) -> CameraIntrinsics:
        return CameraIntrinsics(
            width=640,
            height=480,
            fx=591.1,  # X-axis focal length
            fy=590.1,  # Y-axis focal length
            cx=331.0,  # X-axis principle point
            cy=234.0,  # Y-axis principle point
            factor=5000  # for the 16-bit PNG files
        )

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


class TumIclLoader(TumLoader):
    def _provide_intrinsics(self) -> CameraIntrinsics:
        return CameraIntrinsics(
            width=640,
            height=480,
            fx=481.20,  # X-axis focal length
            fy=-480.00,  # Y-axis focal length
            cx=319.50,  # X-axis principle point
            cy=239.50,  # Y-axis principle point
            factor=5000  # for the 16-bit PNG files
        )


class IclLoader(ImageLoader):
    def _provide_intrinsics(self) -> CameraIntrinsics:
        return CameraIntrinsics(
            width=640,
            height=480,
            cx=None,
            cy=None,
            fx=None,
            fy=None,
            factor=100  # from cm to m
        )

    def _provide_rgb_and_depth_path(self, path: str) -> (str, str):
        return path, path

    def _match_rgb_with_depth(self, rgb_filenames, depth_filenames) -> list:
        return list(range(len(rgb_filenames)))  # as rgb and depth are synchronized

    @staticmethod
    def __filenames_sorted_mapper(filename: str) -> int:
        return int(filename.split(".")[0].split("_")[-1])

    def _provide_filenames(self, rgb_path, depth_path) -> (list, list):
        path = depth_path  # as paths are equal
        filenames = os.listdir(path)
        rgb_filenames = (filter(lambda x: x.endswith(".png"), filenames))
        depth_filenames = (filter(lambda x: x.endswith(".depth"), filenames))

        rgb_filenames = sorted(rgb_filenames, key=IclLoader.__filenames_sorted_mapper)
        depth_filenames = sorted(depth_filenames, key=IclLoader.__filenames_sorted_mapper)

        return rgb_filenames, depth_filenames

    def __load_camera_params_from_file(self, frame_num) -> dict:
        result = {}
        params_path = self.depth_images[frame_num][:-5] + "txt"
        with open(params_path, 'r') as input_file:
            for line in input_file:
                field_name_start = 0
                field_name_end = line.find(" ")
                field_name = line[field_name_start:field_name_end]
                value_start = line.find("=") + 2  # skip space after '='
                if field_name == "cam_angle":
                    value_end = line.find(";")
                else:
                    value_end = line.find(";") - 1
                value = line[value_start:value_end]
                result[field_name] = value

            return result

    def __get_camera_params_for_frame(self, frame_num):
        # Adopted from https://www.doc.ic.ac.uk/~ahanda/VaFRIC/getcamK.m
        camera_params_raw = self.__load_camera_params_from_file(frame_num)
        cam_dir = np.fromstring(camera_params_raw["cam_dir"][1:-1], dtype=float, sep=',').T
        cam_right = np.fromstring(camera_params_raw["cam_right"][1:-1], dtype=float, sep=',').T
        cam_up = np.fromstring(camera_params_raw["cam_up"][1:-1], dtype=float, sep=',').T
        focal = np.linalg.norm(cam_dir)
        aspect = np.linalg.norm(cam_right) / np.linalg.norm(cam_up)
        angle = 2 * math.atan(np.linalg.norm(cam_right) / 2 / focal)

        width = self.cam_intrinsics.width
        height = self.cam_intrinsics.height
        psx = 2 * focal * math.tan(0.5 * angle) / width
        psy = 2 * focal * math.tan(0.5 * angle) / aspect / height

        psx = psx / focal
        psy = psy / focal

        o_x = (width + 1) * 0.5
        o_y = (height + 1) * 0.5

        fx = 1 / psx
        fy = -1 / psy
        cx = o_x
        cy = o_y

        return fx, fy, cx, cy

    def read_depth_image(self, frame_num) -> np.array:
        raise Exception("Not implemented")

    def read_pcd(self, frame_num) -> np.array:
        # Adopted from https://www.doc.ic.ac.uk/~ahanda/VaFRIC/compute3Dpositions.m
        depth_frame_path = self.depth_images[frame_num]
        fx, fy, cx, cy = self.__get_camera_params_for_frame(frame_num)

        width = self.cam_intrinsics.width
        height = self.cam_intrinsics.height
        x_matrix = np.tile(np.arange(width), (height, 1)).flatten()
        y_matrix = np.transpose(np.tile(np.arange(height), (width, 1))).flatten()
        x_modifier = (x_matrix - cx) / fx
        y_modifier = (y_matrix - cy) / fy

        points = np.zeros((width * height, 3))

        with open(depth_frame_path, 'r') as input_file:
            data = input_file.read()
            depth_data = np.asarray(list(
                map(lambda x: float(x), data.split(" ")[:height * width])
            ))

            scale = self.cam_intrinsics.factor
            points[:, 2] = depth_data / np.sqrt(x_modifier ** 2 + y_modifier ** 2 + 1) / scale
            points[:, 0] = x_modifier * points[:, 2]
            points[:, 1] = y_modifier * points[:, 2]

            return points
