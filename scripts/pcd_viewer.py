import os
import sys

import open3d as o3d

if __name__ == "__main__":
    path_dir = sys.argv[1]
    if len(sys.argv) > 2:
        first_frame_num = int(sys.argv[2])
    else:
        first_frame_num = 0

    filenames = os.listdir(path_dir)
    filenames = sorted(filenames, key=lambda x: int(x[:-4]))
    for index, filename in enumerate(filenames):
        if index < first_frame_num:
            continue

        path = os.path.join(path_dir, filename)
        print("Shown {0}) {1}".format(index, filename))
        o3d.visualization.draw_geometries([o3d.io.read_point_cloud(path)])

