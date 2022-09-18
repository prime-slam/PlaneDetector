import os
import sys

import numpy as np
import open3d as o3d

if __name__ == "__main__":
    path_dir = sys.argv[1]
    if len(sys.argv) > 2:
        labels_dir = sys.argv[2]
    else:
        labels_dir = path_dir
    if len(sys.argv) > 3:
        first_frame_num = int(sys.argv[3])
    else:
        first_frame_num = 0

    filenames = list(filter(lambda x: x.endswith(".pcd"), os.listdir(path_dir)))
    label_filenames = list(filter(lambda x: x.endswith(".npy"), os.listdir(labels_dir)))
    filenames = sorted(filenames, key=lambda x: int(x[:-4]))
    label_filenames = sorted(label_filenames, key=lambda x: int(x[:-4]))
    for index, filename in enumerate(filenames):
        if index < first_frame_num:
            continue

        path = os.path.join(path_dir, filename)
        print("Shown {0}) {1}".format(index, filename))

        pcd = o3d.io.read_point_cloud(path)

        # Use this instead of open3d read if you need loading from KITTI format
        # pcd_points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(pcd_points[:, :3])

        labels = np.load(os.path.join(labels_dir, label_filenames[index]))
        colors = np.concatenate([np.asarray([[0, 0, 0]]), np.random.rand(np.max(labels), 3)])
        pcd.paint_uniform_color([0, 0, 0])
        pcd.colors = o3d.utility.Vector3dVector(colors[labels])

        o3d.visualization.draw_geometries([pcd])
