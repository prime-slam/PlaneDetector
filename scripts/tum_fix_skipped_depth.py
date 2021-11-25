import os
import sys

import cv2
import numpy as np


class TumDataset:
    def __init__(self, path):
        rgb_path, depth_path = self.provide_rgb_and_depth_path(path)

        rgb_filenames, depth_filenames = self.provide_filenames(rgb_path, depth_path)
        rgb_filenames.sort()
        depth_filenames.sort()

        self.depth_images = [os.path.join(depth_path, filename) for filename in depth_filenames]
        self.rgb_images = [os.path.join(rgb_path, filename) for filename in rgb_filenames]
        self.depth_to_rgb_index = []

        self.match_rgb_with_depth(rgb_filenames, depth_filenames)

    def provide_rgb_and_depth_path(self, path):
        depth_path = os.path.join(path, "depth")
        rgb_path = os.path.join(path, "rgb")

        return rgb_path, depth_path

    def provide_filenames(self, rgb_path, depth_path):
        rgb_filenames = os.listdir(rgb_path)
        depth_filenames = os.listdir(depth_path)

        return rgb_filenames, depth_filenames

    def read_depth_image(self, frame_num):
        depth_frame_path = self.depth_images[frame_num]
        return cv2.imread(depth_frame_path, cv2.IMREAD_ANYDEPTH)

    def match_rgb_with_depth(self, rgb_filenames, depth_filenames):
        rgb_index = 0
        depth_index = 0
        prev_delta = float('inf')
        while depth_index < len(depth_filenames) and rgb_index < len(rgb_filenames):
            rgb_timestamp = float(rgb_filenames[rgb_index][:-4])
            depth_timestamp = float(depth_filenames[depth_index][:-4])
            delta = abs(depth_timestamp - rgb_timestamp)

            if rgb_timestamp <= depth_timestamp:
                prev_delta = delta
                rgb_index += 1
                continue

            if prev_delta < delta:
                self.depth_to_rgb_index.append(rgb_index - 1)
            else:
                self.depth_to_rgb_index.append(rgb_index)

            depth_index += 1


class CameraIntrinsics:
    def __init__(self, width, height, fx, fy, cx, cy):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy


def get_cam_intrinsics(image_shape):
    return CameraIntrinsics(
            width=image_shape[1],
            height=image_shape[0],
            fx=591.1,  # X-axis focal length
            fy=590.1,  # Y-axis focal length
            cx=331.0,  # X-axis principle point
            cy=234.0  # Y-axis principle point
        )


if __name__ == "__main__":
    path = sys.argv[1]
    tum_loader = TumDataset(path)
    path_out = os.path.join(path, "out")
    if not os.path.exists(path_out):
        os.mkdir(path_out)

    i = 0
    for index, depth_path in enumerate(tum_loader.depth_images):
        rgb_index = tum_loader.depth_to_rgb_index[index]
        rgb_path = tum_loader.rgb_images[rgb_index]

        rgb_image = cv2.imread(rgb_path)
        depth_image = tum_loader.read_depth_image(index)
        if index == 0:
            camera_intrinsics = get_cam_intrinsics(depth_image.shape)
            column_indices = np.tile(np.arange(camera_intrinsics.width), (camera_intrinsics.height, 1)).flatten()
            row_indices = np.transpose(np.tile(np.arange(camera_intrinsics.height), (camera_intrinsics.width, 1))).flatten()

        rgb_image[depth_image == 0] = [0, 0, 0]

        factor = 5000
        points = np.zeros((camera_intrinsics.width * camera_intrinsics.height, 3))

        points[:, 2] = depth_image.flatten() / factor
        points[:, 0] = (column_indices - camera_intrinsics.cx) * points[:, 2] / camera_intrinsics.fx
        points[:, 1] = (row_indices - camera_intrinsics.cy) * points[:, 2] / camera_intrinsics.fy
        distance = np.sqrt(np.sum(np.square(points), axis=1)).reshape((480, 640))
        rgb_image[distance >= 5] = [0, 0, 0]

        cv2.imwrite(os.path.join(path_out, "{:05d}.png".format(i)), rgb_image)
        i += 1

