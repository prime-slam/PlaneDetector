import os
import struct
import sys

import numpy as np


def read_labels(filename):
    """ read labels from given file. """
    contents = bytes()
    with open(filename, "rb") as f:
        f.seek(0, 2)  # move the cursor to the end of the file
        num_points = int(f.tell() / 4)
        f.seek(0, 0)
        contents = f.read()

    arr = [struct.unpack('<I', contents[4 * i:4 * i + 4])[0] for i in range(num_points)]

    return arr


PLANAR_IDS = [40, 48]


if __name__ == "__main__":
    input_path = sys.argv[1]
    segm_path = sys.argv[2]
    output_path = sys.argv[3]

    for labels_filename, segm_labels_filename in zip(os.listdir(input_path), os.listdir(segm_path)):
      labels_filepath = os.path.join(input_path, labels_filename)
      segm_filepath = os.path.join(segm_path, segm_labels_filename)
      labels = np.load(labels_filepath)
      segm = np.asarray(read_labels(segm_filepath))
      segm_is_planar = np.asarray([label in PLANAR_IDS for label in segm])
      labels[segm_is_planar] = segm[segm_is_planar]
      output_filepath = os.path.join(output_path, labels_filename)
      np.save(output_filepath, labels)
