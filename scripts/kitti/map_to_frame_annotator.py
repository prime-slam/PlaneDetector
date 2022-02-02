import os
import sys

import numpy as np
import open3d as o3d
from pykdtree.kdtree import KDTree
from sklearn.cluster import DBSCAN

from scripts.kitti.map_builder import KittiLoader, load_poses, load_calib_matrix, SSEAnnotation, cloud_to_map


def dbscan_labels(pcd: o3d.geometry.PointCloud, labels: np.array) -> np.array:
    unique_labels, labels_in_unique_indices = np.unique(labels, return_inverse=True)
    result_labels = np.zeros_like(labels)
    for label_index, label in enumerate(unique_labels):
        if label == 0:
            continue

        label_indices = np.where(labels_in_unique_indices == label_index)[0]
        label_points = np.asarray(pcd.points)[label_indices]
        clustering = DBSCAN(eps=0.5).fit(label_points)
        unique_components, counts = np.unique(clustering.labels_, return_counts=True)
        if unique_components.size == 1:
            if unique_components[0] == -1:
                continue
            else:
                most_frequent_positive_id = unique_components[0]
        else:
            most_frequent_component_ids = unique_components[np.argpartition(-counts, kth=1)[:2]]
            for component_id in most_frequent_component_ids:
                if component_id >= 0:
                    most_frequent_positive_id = component_id
                    break
        component_indices = np.where(clustering.labels_ == most_frequent_positive_id)[0]
        component_indices_in_part = label_indices[component_indices]
        result_labels[component_indices_in_part] = label

    print("Removed: {0}/{1}".format(np.count_nonzero(labels - result_labels), np.count_nonzero(labels)))
    return result_labels


def build_map(parts_path) -> (o3d.geometry.PointCloud, np.array):
    data_filenames = [
        os.path.join(parts_path, filename) for filename in os.listdir(parts_path) if filename.endswith(".pcd")
    ]
    max_used_plane_id = 0
    map_pcd = o3d.geometry.PointCloud()
    map_labels_list = []
    labels_filenames = [filename + ".labels" for filename in data_filenames]
    for data_filename, label_filename in zip(data_filenames, labels_filenames):
        data_pcd = o3d.io.read_point_cloud(data_filename)
        labels = SSEAnnotation(label_filename).load_labels()
        is_null_labels = labels != 0
        labels = (labels + max_used_plane_id) * is_null_labels
        max_used_plane_id = max(max_used_plane_id, np.max(labels))
        map_pcd += data_pcd
        map_labels_list.append(labels)

    return map_pcd, np.concatenate(map_labels_list)
    # return o3d.io.read_point_cloud("map.pcd"), np.load("map.pcd.labels.npy")


def annotate_frame_with_map(
        frame_pcd: o3d.geometry.PointCloud,
        map_kd_tree: KDTree,
        map_labels: np.array,
        transform_matrix: np.array,
        calib_matrix: np.array
) -> np.array:
    mapped_frame_pcd = cloud_to_map(frame_pcd, transform_matrix, calib_matrix)

    # map_pcd.paint_uniform_color([0, 0, 0])
    # labels_unique = np.unique(map_labels)
    # colors = np.concatenate([np.asarray([[0,0,0]]), np.random.rand(labels_unique.size, 3)])
    # map_colors = colors[map_labels]
    # map_pcd.colors = o3d.utility.Vector3dVector(map_colors)
    # mapped_frame_pcd.paint_uniform_color([1, 0, 0])
    # o3d.visualization.draw_geometries([map_pcd, mapped_frame_pcd])

    frame_indices_in_map = map_kd_tree.query(np.asarray(mapped_frame_pcd.points), distance_upper_bound=0.5)[1]
    # points with no reference will be marked with len(mapped_frame_pcd) index, so add zero to this index
    map_labels = np.concatenate([map_labels, np.asarray([0])])
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
    map_labels = dbscan_labels(map_pcd, map_labels)
    map_kd_tree = KDTree(np.asarray(map_pcd.points))

    if debug:
        control_frame_ids = np.concatenate([np.random.randint(low=0, high=250, size=3), np.asarray([31, 211])])

    for frame_id in range(loader.get_frame_count()):
        if frame_id != 31:
            continue
        frame_pcd = loader.read_pcd(frame_id)
        transform_matrix = poses[frame_id]
        frame_labels, frame_indices_in_map = annotate_frame_with_map(frame_pcd, map_kd_tree, map_labels, transform_matrix, calib_matrix)
        frame_labels_ref = frame_labels[np.where(frame_indices_in_map != map_labels.size)[0]]
        frame_indices_in_map = frame_indices_in_map[np.where(frame_indices_in_map != map_labels.size)[0]]
        output_filename = "label-{:06d}.npy".format(frame_id)
        np.save(os.path.join(output_path, output_filename), frame_labels)

        if debug and frame_id in control_frame_ids:
            pcd_filename = "{:06d}.pcd".format(frame_id)
            ref_pcd_filename = "ref_{}".format(pcd_filename)
            _, unique_indices = np.unique(frame_indices_in_map, return_index=True)
            ref_pcd = map_pcd.select_by_index(frame_indices_in_map[unique_indices])
            o3d.io.write_point_cloud(os.path.join(output_path, ref_pcd_filename), ref_pcd)
            ref_labels = frame_labels_ref[unique_indices]
            SSEAnnotation.save_to_file(ref_labels, os.path.join(output_path, ref_pcd_filename))
            SSEAnnotation.save_to_file(frame_labels, os.path.join(output_path, pcd_filename))
            o3d.io.write_point_cloud(os.path.join(output_path, pcd_filename), frame_pcd)
            print("Frame {} is debug choice!".format(frame_id))

        print("Frame {} is ready!".format(frame_id))

        if frame_id == 250:
            break
