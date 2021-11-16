import os
import sys

import cv2


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
        black_image = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)

        rgb_image[black_image == 0] = [0, 0, 0]
        cv2.imwrite(os.path.join(path_out, "{:05d}.png".format(i)), rgb_image)
        i += 1

