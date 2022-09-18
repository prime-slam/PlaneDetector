import os
import sys

import cv2
import numpy as np
import skimage

from src.utils.colors import color_to_string

if __name__ == "__main__":
    path_to_images = sys.argv[1]
    output_path = sys.argv[2]

    colors_to_process = [
        color_to_string([128, 121, 155]),
        color_to_string([140, 8, 226]),
        color_to_string([151, 2, 233]),
        color_to_string([228, 154, 237]),
        color_to_string([69, 164, 214]),
        color_to_string([9, 20, 126]),
        color_to_string([197, 167, 21]),
        color_to_string([122, 235, 7]),
        color_to_string([218, 241, 247]),
        color_to_string([80, 181, 86]),
        color_to_string([14, 60, 221]),
        color_to_string([212, 156, 166]),
        color_to_string([167, 13, 99]),
    ]
    for image_filename in os.listdir(path_to_images):
        image_path = os.path.join(path_to_images, image_filename)
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        unique_colors = np.unique(image.reshape(image.shape[0] * image.shape[1], 3), axis=0)

        # For each color find the largest connected component and then fix it, other components to the bin
        skipped_cc_indices = []
        image_with_max_cc = np.zeros_like(image)
        for color in unique_colors:
            # skip black color
            if np.all(color == [0, 0, 0]):
                continue

            # simply copy not processed colors
            if color_to_string(color) not in colors_to_process:
                saved_indices = np.where(np.all(image == color, axis=-1))
                image_with_max_cc[saved_indices[0], saved_indices[1]] = color
                continue

            mask = (image == color).all(axis=-1)

            max_cc_indices = (np.asarray([], dtype=int), np.asarray([], dtype=int))
            labels = skimage.measure.label(mask)
            labels = (labels + 1) * mask  # now only background is zero
            not_max_cc_indices = []
            for label_id in range(1, labels.max() + 1):
                tmp = labels == label_id
                label_indices = np.where(labels == label_id)
                if label_indices[0].size > max_cc_indices[0].size:
                    if max_cc_indices[0].size > 0:
                        not_max_cc_indices.append(max_cc_indices)
                    max_cc_indices = label_indices
                elif label_indices[0].size > 0:
                    not_max_cc_indices.append(label_indices)

            if len(not_max_cc_indices) > 0:
                skipped_cc_indices.append(not_max_cc_indices)
            image_with_max_cc[max_cc_indices[0], max_cc_indices[1]] = color

        # flatten_skipped_cc_indices = [indices for index_group in skipped_cc_indices for indices in index_group]
        # all_skipped_cc_indices = (
        #     np.concatenate([indices[0] for indices in flatten_skipped_cc_indices]),
        #     np.concatenate([indices[1] for indices in flatten_skipped_cc_indices])
        # )
        # image_with_max_cc[all_skipped_cc_indices] = [125, 125, 125]

        cv2.imwrite(os.path.join(output_path, image_filename),  cv2.cvtColor(image_with_max_cc, cv2.COLOR_RGB2BGR))

