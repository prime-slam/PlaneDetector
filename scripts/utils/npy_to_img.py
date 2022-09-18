import os
import sys

import cv2
import numpy as np

from src.utils.colors import denormalize_color_arr

if __name__ == "__main__":
    input_path = sys.argv[1]

    for folder in os.listdir(input_path):
        folder_path = os.path.join(input_path, folder)
        for filename in os.listdir(folder_path):
            if not filename.endswith('.npy'):
                continue
            file_path = os.path.join(folder_path, filename)
            labels = np.load(file_path).astype(dtype=int)
            labels_unique = np.unique(labels)
            colors = np.concatenate([np.asarray([[0, 0, 0]]), np.random.rand(np.max(labels_unique), 3)])
            colors_denorm = denormalize_color_arr(colors)
            label_colors = colors_denorm[labels]
            annot_img = label_colors.reshape((480, 640, 3))
            cv2.imwrite("{}.png".format(file_path[:-4]), annot_img)
