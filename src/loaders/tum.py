import os


class TumDataset:
    def __init__(self, path):
        self.path = path
        depth_path = os.path.join(path, "depth")
        self.depth_images = [os.path.join(depth_path, filename) for filename in os.listdir(depth_path)]
        rgb_path = os.path.join(path, "rgb")
        self.rgb_images = [os.path.join(rgb_path, filename) for filename in os.listdir(rgb_path)]
        self.depth_to_rgb_index = []

        rgb_filenames = os.listdir(rgb_path)
        depth_filenames = os.listdir(depth_path)
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
