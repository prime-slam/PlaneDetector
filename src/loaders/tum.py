import os

from src.loaders.config import Tum


class TumDataset:
    def __init__(self, path):
        self.config = self.provide_config()

        rgb_path, depth_path = self.provide_rgb_and_depth_path(path)
        self.depth_images = [os.path.join(depth_path, filename) for filename in os.listdir(depth_path)]
        self.rgb_images = [os.path.join(rgb_path, filename) for filename in os.listdir(rgb_path)]
        self.depth_to_rgb_index = []

        rgb_filenames, depth_filenames = self.provide_filenames(rgb_path, depth_path)

        self.match_rgb_with_depth(rgb_filenames, depth_filenames)

    def provide_config(self):
        return Tum

    def provide_rgb_and_depth_path(self, path):
        depth_path = os.path.join(path, "depth")
        rgb_path = os.path.join(path, "rgb")

        return rgb_path, depth_path

    def provide_filenames(self, rgb_path, depth_path):
        rgb_filenames = os.listdir(rgb_path)
        depth_filenames = os.listdir(depth_path)

        return rgb_filenames, depth_filenames

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
