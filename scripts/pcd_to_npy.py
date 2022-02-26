import os
import sys

import numpy as np
import open3d as o3d

if __name__ == "__main__":
    input_path = sys.argv[1]
    for filename in os.listdir(input_path):
        file_path = os.path.join(input_path, filename)
        pcd = o3d.io.read_point_cloud(file_path)

        label_colors = np.array(pcd.colors)
        labels = np.zeros(label_colors.shape[0], dtype=int)
        unique_colors = np.unique(label_colors, axis=0)
        for index, color in enumerate(unique_colors):
            color_indices = np.where(np.all(label_colors == color, axis=-1))[0]
            if not np.array_equal(color, np.asarray([0., 0., 0.])):
                labels[color_indices] = index + 1

        np.save("{}.npy".format(file_path[:-4]), labels)
