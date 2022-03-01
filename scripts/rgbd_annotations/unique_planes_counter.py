import os
import sys

import cv2
import numpy as np

if __name__ == "__main__":
    input_path = sys.argv[1]
    colors = []
    sum = 0
    cnt = 0
    for index, filename in enumerate(os.listdir(input_path)):
        if not filename.endswith(".npy"):
            continue
        filepath = os.path.join(input_path, filename)
        labels = np.load(filepath)
        unique_labels = np.unique(labels)
        colors.append(unique_labels)
        cnt += 1
        sum += unique_labels.shape[0]
        print("{} is ready".format(index))

    unique_planes = np.unique(np.concatenate(colors))
    print(unique_planes.shape[0])
    print("Avg: {}".format(sum / cnt))

