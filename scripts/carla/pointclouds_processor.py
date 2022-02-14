import argparse
import os.path

import numpy as np
import open3d as o3d
import ast

from pykdtree.kdtree import KDTree


def visualize_pcd_labels(pcd: o3d.geometry.PointCloud, labels: np.array, filename: str = None):
    colors = np.concatenate([np.asarray([[0, 0, 0]]), np.random.rand(np.max(labels), 3)])
    pcd_for_vis = o3d.geometry.PointCloud()
    pcd_for_vis.points = o3d.utility.Vector3dVector(np.asarray(pcd.points))
    pcd_for_vis.paint_uniform_color([0, 0, 0])
    pcd_for_vis.colors = o3d.utility.Vector3dVector(colors[labels])
    if filename is None:
        o3d.visualization.draw_geometries([pcd_for_vis])
    else:
        o3d.io.write_point_cloud(filename, pcd_for_vis)


def pcd_from_carla_line(line: str) -> o3d.geometry.PointCloud:
    points_packed = ast.literal_eval(line.split(",|,")[1])
    points = np.asarray([point_label_id[0] for point_label_id in points_packed])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    return pcd


def build_map(data_path) -> o3d.geometry.PointCloud:
    map_pcd = o3d.geometry.PointCloud()
    with open(data_path) as data_file:
        for index, line in enumerate(data_file):
            if index % 10 != 0:
                continue

            frame_pcd = pcd_from_carla_line(line)
            map_pcd += frame_pcd
            print("Cloud {} loaded!".format(index + 1))

        map_pcd = map_pcd.voxel_down_sample(0.2)

    return map_pcd


def annotate_map(map_pcd: o3d.geometry.PointCloud, annot_path: str) -> np.array:
    mesh_names = [filename[:-4] for filename in os.listdir(annot_path) if filename.endswith(".pcd")]
    map_tree = KDTree(np.asarray(map_pcd.points))
    map_points_count = np.asarray(map_pcd.points).shape[0]
    map_labels = np.zeros(map_points_count, dtype=int)
    max_used_label = -1
    for index, mesh_name in enumerate(mesh_names):
        mesh_pcd_filename = "{}.pcd".format(mesh_name)
        mesh_labels_filename = "{}.npy".format(mesh_name)
        mesh_pcd = o3d.io.read_point_cloud(os.path.join(annot_path, mesh_pcd_filename))
        mesh_labels = np.load(os.path.join(annot_path, mesh_labels_filename))
        mesh_labels += max_used_label + 1
        max_used_label = max(max_used_label, np.max(mesh_labels))

        mesh_points = np.asarray(mesh_pcd.points)
        # Swap y and z axis for UE coordinates and go from cm to metres
        mesh_points_z = mesh_points[:, 2].copy()
        mesh_points[:, 2] = mesh_points[:, 1]
        mesh_points[:, 1] = mesh_points_z
        mesh_points /= 100

        mesh_map_indices = map_tree.query(mesh_points, distance_upper_bound=0.5)[1]
        mesh_labels_not_null_indices = np.where(mesh_map_indices < map_points_count)[0]
        mesh_map_indices_not_null = mesh_map_indices[mesh_labels_not_null_indices]
        map_labels[mesh_map_indices_not_null] = mesh_labels[mesh_labels_not_null_indices]

        print("{0} is ready! ({1}/{2})".format(mesh_name, index + 1, len(mesh_names)))

    return map_labels


def annotate_frames(data_path: str, output_path: str, map_pcd: o3d.geometry.PointCloud, map_labels: np.array):
    map_kd_tree = KDTree(np.asarray(map_pcd.points))
    with open(data_path) as data_file:
        for index, line in enumerate(data_file):
            frame_pcd = pcd_from_carla_line(line)
            frame_labels = annotate_frame_with_map(frame_pcd, map_kd_tree, map_labels)
            o3d.io.write_point_cloud(os.path.join(output_path, "{:06d}.pcd".format(index)), frame_pcd)
            np.save(os.path.join(output_path, "{:06d}.npy".format(index)), frame_labels)


def annotate_frame_with_map(frame_pcd: o3d.geometry.PointCloud, map_kd_tree: KDTree, map_labels: np.array) -> np.array:
    frame_indices_in_map = map_kd_tree.query(np.asarray(frame_pcd.points), distance_upper_bound=0.5)[1]
    # points with no reference will be marked with len(mapped_frame_pcd) index, so add zero to this index
    map_labels = np.concatenate([map_labels, np.asarray([0])])
    return map_labels[frame_indices_in_map]


def process(data_path, annot_path, output_path):
    map_filename = "carla_map.pcd"
    annot_filename = "carla_map.npy"

    if os.path.exists(map_filename):
        map_pcd = o3d.io.read_point_cloud(map_filename)
    else:
        map_pcd = build_map(data_path)
        o3d.io.write_point_cloud(map_filename, map_pcd)

    map_pcd.paint_uniform_color([0, 0, 0])

    if os.path.exists(annot_filename):
        map_labels = np.load(annot_filename)
    else:
        map_labels = annotate_map(map_pcd, annot_path)
        np.save(annot_filename, map_labels)

    annotate_frames(data_path, output_path, map_pcd, map_labels)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        'data_path',
        help='path to measurements file'
    )
    argparser.add_argument(
        'annot_path_path',
        help='path to where to save new pcd'
    )
    argparser.add_argument(
        'output_path',
        help='path to where to save new pcd'
    )
    args = argparser.parse_args()

    process(args.data_path, args.annot_path_path, args.output_path)
