import os

import open3d as o3d

if __name__ == "__main__":
    path_dir = "C:\\Users\\dimaj\\Documents\\Github\\PlaneDetector\\scripts\\kitti\\debug"
    # path_dir = "C:\\Users\\dimaj\\Documents\\Github\\PlaneDetector\\scripts\\kitti\\debug_map"
    filenames = os.listdir(path_dir)
    # filenames = sorted(filenames, key=lambda x: int(x.split(".")[0]))
    for filename in filenames:
        path = os.path.join(path_dir, filename)
        print(filename)
        o3d.visualization.draw_geometries([o3d.io.read_point_cloud(path)])

