import os
import sys

import numpy as np
import open3d as o3d
from pykdtree.kdtree import KDTree

from scripts.kitti.map_builder import KittiLoader, load_poses, load_calib_matrix, SSEAnnotation, cloud_to_map


def build_map(parts_path) -> (o3d.geometry.PointCloud, np.array):
    # data_filenames = [
    #     os.path.join(parts_path, filename) for filename in os.listdir(parts_path) if filename.endswith(".pcd")
    # ]
    # max_used_plane_id = 0
    # map_pcd = o3d.geometry.PointCloud()
    # map_labels_list = []
    # labels_filenames = [filename + ".labels" for filename in data_filenames]
    # for data_filename, label_filename in zip(data_filenames, labels_filenames):
    #     data_pcd = o3d.io.read_point_cloud(data_filename)
    #     labels = SSEAnnotation(label_filename).load_labels()
    #     is_null_labels = labels != 0
    #     labels = (labels + max_used_plane_id) * is_null_labels
    #     max_used_plane_id = max(max_used_plane_id, np.max(labels))
    #     map_pcd += data_pcd
    #     map_labels_list.append(labels)
    #
    # return map_pcd, np.concatenate(map_labels_list)
    return o3d.io.read_point_cloud("map.pcd"), np.load("map.pcd.labels.npy")


def annotate_frame_with_map(
        frame_pcd: o3d.geometry.PointCloud,
        map_kd_tree: KDTree,
        map_labels: np.array,
        transform_matrix: np.array,
        calib_matrix: np.array
) -> np.array:
    mapped_frame_pcd = cloud_to_map(frame_pcd, transform_matrix, calib_matrix)
    frame_indices_in_map = map_kd_tree.query(np.asarray(mapped_frame_pcd.points))[1]
    # map_labels = np.concatenate([map_labels, np.asarray([0])])
    return map_labels[frame_indices_in_map], frame_indices_in_map


if __name__ == "__main__":
    data_path = sys.argv[1]
    map_parts_path = sys.argv[2]
    path_to_poses = sys.argv[3]
    path_to_calib = sys.argv[4]
    output_path = sys.argv[5]
    debug = True

    loader = KittiLoader(data_path)
    poses = load_poses(path_to_poses)
    calib_matrix = load_calib_matrix(path_to_calib)

    map_pcd, map_labels = build_map(map_parts_path)
    map_kd_tree = KDTree(np.asarray(map_pcd.points))

    if debug:
        control_frame_ids = np.concatenate([np.random.randint(low=0, high=250, size=3), np.asarray([31, 211])])

    for frame_id in range(loader.get_frame_count()):
        if frame_id != 31:
            continue
        frame_pcd = loader.read_pcd(frame_id)
        transform_matrix = poses[frame_id]
        frame_labels, frame_indices_in_map = annotate_frame_with_map(frame_pcd, map_kd_tree, map_labels, transform_matrix, calib_matrix)

        output_filename = "label-{:06d}.npy".format(frame_id)
        np.save(os.path.join(output_path, output_filename), frame_labels)

        if debug and frame_id in control_frame_ids:
            pcd_filename = "{:06d}.pcd".format(frame_id)
            ref_pcd_filename = "ref_{}".format(pcd_filename)
            _, unique_indices = np.unique(frame_indices_in_map, return_index=True)
            ref_pcd = map_pcd.select_by_index(frame_indices_in_map[unique_indices])
            o3d.io.write_point_cloud(os.path.join(output_path, ref_pcd_filename), ref_pcd)
            ref_labels = frame_labels[unique_indices]
            SSEAnnotation.save_to_file(ref_labels, os.path.join(output_path, ref_pcd_filename))
            SSEAnnotation.save_to_file(frame_labels, os.path.join(output_path, pcd_filename))
            o3d.io.write_point_cloud(os.path.join(output_path, pcd_filename), frame_pcd)
            print("Frame {} is debug choice!".format(frame_id))

        print("Frame {} is ready!".format(frame_id))

        if frame_id == 250:
            break
