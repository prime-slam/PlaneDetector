import os
import sys

import numpy as np
import open3d as o3d

if __name__ == "__main__":
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    input_filepaths = [os.path.join(input_path, filename) for filename in os.listdir(input_path) if filename.endswith(".pcd")]
    for filepath in input_filepaths:
        filename = os.path.split(filepath)[-1]
        pcd = o3d.io.read_point_cloud(filepath)
        points = np.asarray(pcd.points)
        points.tofile(os.path.join(output_path, "{}.bin".format(filename[:-4])))
        print("Done {}".format(filename))
