import os

import open3d as o3d

if __name__ == "__main__":
    path_dir = "C:\\Users\\dimaj\\Documents\\Github\\PlaneDetector\\scripts\\kitti\\debug_map_no_eye"
    filenames = sorted(os.listdir(path_dir), key=lambda x: int(x.split(".")[0]))
    for filename in filenames:
        path = os.path.join(path_dir, filename)
        print(filename)
        o3d.visualization.draw_geometries([o3d.io.read_point_cloud(path)])

# 73(too high points) -- before eye
# 215!!(too low points) - after eye