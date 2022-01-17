import math
import os
import open3d as o3d

import numpy as np

from src.loaders.depth_image.CameraConfig import CameraConfig
from src.loaders.depth_image.ImageLoader import ImageLoader
from src.model.SegmentedPointCloud import SegmentedPointCloud


class IclLoader(ImageLoader):

    __IMAGE_WIDTH = 640
    __IMAGE_HEIGHT = 480

    def _provide_config(self) -> CameraConfig:
        return None

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

        width = self.__IMAGE_WIDTH
        height = self.__IMAGE_HEIGHT
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

    def read_pcd(self, frame_num) -> SegmentedPointCloud:
        # Adopted from https://www.doc.ic.ac.uk/~ahanda/VaFRIC/compute3Dpositions.m
        depth_frame_path = self.depth_images[frame_num]
        fx, fy, cx, cy = self.__get_camera_params_for_frame(frame_num)

        x_matrix = np.tile(np.arange(self.__IMAGE_WIDTH), (self.__IMAGE_HEIGHT, 1)).flatten()
        y_matrix = np.transpose(np.tile(np.arange(self.__IMAGE_HEIGHT), (self.__IMAGE_WIDTH, 1))).flatten()
        x_modifier = (x_matrix - cx) / fx
        y_modifier = (y_matrix - cy) / fy

        points = np.zeros((self.__IMAGE_WIDTH * self.__IMAGE_HEIGHT, 3))

        with open(depth_frame_path, 'r') as input_file:
            data = input_file.read()
            depth_data = np.asarray(list(
                map(lambda x: float(x), data.split(" ")[:self.__IMAGE_HEIGHT * self.__IMAGE_WIDTH])
            ))
            # depth_data = depth_data.reshape((480, 640))

            scale = 100  # from cm to m
            points[:, 2] = depth_data / np.sqrt(x_modifier ** 2 + y_modifier ** 2 + 1) / scale
            points[:, 0] = x_modifier * points[:, 2]
            points[:, 1] = y_modifier * points[:, 2]

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.transform([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

            return SegmentedPointCloud(
                pcd=pcd,
                unsegmented_cloud_indices=np.arange(points.shape[0]),
                structured_shape=(self.__IMAGE_HEIGHT, self.__IMAGE_WIDTH)
            )
