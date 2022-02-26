import multiprocessing
import os
import sys

import numpy as np
import open3d as o3d
from pykdtree.kdtree import KDTree
from sklearn.cluster import DBSCAN

from scripts.kitti.map_builder import KittiLoader, load_poses, load_calib_matrix, SSEAnnotation, cloud_to_map


def save_debug_pcd(map_pcd: o3d.geometry.PointCloud, label_indices, label, color_indices: list):
    pcd = o3d.geometry.PointCloud()
    pcd_on_map = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(map_pcd.points)[label_indices])
    pcd_on_map.points = o3d.utility.Vector3dVector(np.asarray(map_pcd.points))
    pcd.paint_uniform_color([0, 0, 0])
    pcd_on_map.paint_uniform_color([0, 0, 0])
    colors = np.random.rand(len(color_indices), 3)
    tmp_colors = np.asarray(pcd.colors)
    tmp_map_colors = np.asarray(pcd_on_map.colors)
    for ind, color_ind in enumerate(color_indices):
        color = colors[ind]
        tmp_colors[color_ind] = color
        tmp_map_colors[label_indices[color_ind]] = color
    pcd.colors = o3d.utility.Vector3dVector(tmp_colors)
    pcd_on_map.colors = o3d.utility.Vector3dVector(tmp_map_colors)
    o3d.io.write_point_cloud(os.path.join("debug", "{}.pcd".format(label)), pcd)
    o3d.io.write_point_cloud(os.path.join("debug_map", "{}.pcd".format(label)), pcd_on_map)


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


def dbscan_labels(pcd: o3d.geometry.PointCloud, labels: np.array) -> np.array:
    unique_labels, labels_in_unique_indices = np.unique(labels, return_inverse=True)
    result_labels = np.zeros_like(labels)
    problem_counter = 0
    many_clusters = []
    cluster_sizes = []
    full_minus_one = 0
    for label_index, label in enumerate(unique_labels):
        if label == 0:
            continue

        label_indices = np.where(labels_in_unique_indices == label_index)[0]
        label_points = np.asarray(pcd.points)[label_indices]
        clustering = DBSCAN(eps=0.5).fit(label_points)
        unique_components, counts = np.unique(clustering.labels_, return_counts=True)
        is_problem = True
        if unique_components.size == 1:
            if unique_components[0] == -1:
                full_minus_one += 1
                if debug_miss_clicks:
                    save_debug_pcd(pcd, label_indices, label, [])
                continue
            else:
                is_problem = False
        else:
            if unique_components.size == 2 and -1 in unique_components:
                is_problem = False
            else:
                cmp_cnt = 0
                cluster_size = []
                color_indices = []
                for ind, cmp in enumerate(unique_components):
                    if cmp != -1:
                        cmp_cnt += 1
                        cluster_size.append(counts[ind])
                        color_indices.append(np.where(clustering.labels_ == cmp)[0])
                many_clusters.append(cmp_cnt)
                cluster_sizes.append(cluster_size)
                if debug_miss_clicks:
                    save_debug_pcd(pcd, label_indices, label, color_indices)

        if is_problem:
            problem_counter += 1

        component_indices = np.where(clustering.labels_ != -1)[0]
        component_indices_in_part = label_indices[component_indices]
        result_labels[component_indices_in_part] = label

    print("Removed: {0}/{1}".format(np.count_nonzero(labels - result_labels), np.count_nonzero(labels)))
    print("Problem labels: {0}/{1}".format(problem_counter, unique_labels.size))
    print("Full_minus_one: {}".format(full_minus_one))
    print("Group size with count:")
    vals, counts_vals = np.unique(np.asarray(many_clusters), return_counts=True)
    for tmp1, tmp2 in zip(vals, counts_vals):
        print("{}: {}".format(tmp1, tmp2))
    return result_labels


def build_map(parts_path) -> (o3d.geometry.PointCloud, np.array):
    data_filenames = [
        os.path.join(parts_path, filename) for filename in os.listdir(parts_path) if filename.endswith(".pcd")
    ]
    max_used_plane_id = 0
    prev_max_used_plane_id = 0
    map_pcd = o3d.geometry.PointCloud()
    map_labels_list = []
    labels_filenames = [filename + ".labels" for filename in data_filenames]
    for data_filename, label_filename in zip(data_filenames, labels_filenames):
        data_pcd = o3d.io.read_point_cloud(data_filename)
        labels = SSEAnnotation(label_filename).load_labels()
        is_null_labels = labels != 0
        labels = (labels + max_used_plane_id) * is_null_labels
        prev_max_used_plane_id = max_used_plane_id
        max_used_plane_id = max(max_used_plane_id, np.max(labels))
        if prev_max_used_plane_id + 1 - max_used_plane_id < 0:
            print("'{0}': ({1}, {2})".format(os.path.split(data_filename)[-1], prev_max_used_plane_id + 1, max_used_plane_id))
        map_pcd += data_pcd
        map_labels_list.append(labels)

    return map_pcd, np.concatenate(map_labels_list)
    # return o3d.io.read_point_cloud("map.pcd"), np.load("map.pcd.labels.npy")


def unpacking_apply_along_axis(params):
    """
    Like numpy.apply_along_axis(), but with arguments in a tuple
    instead.

    This function is useful with multiprocessing.Pool().map(): (1)
    map() only handles functions that take a single argument, and (2)
    this function can generally be imported from a module, as required
    by map().
    """
    func1d, axis, arr, args, kwargs = params
    return np.apply_along_axis(func1d, axis, arr, *args, **kwargs)


def parallel_apply_along_axis(func1d, axis, arr, *args, **kwargs):
    """
    Like numpy.apply_along_axis(), but takes advantage of multiple
    cores.
    """

    # Effective axis where apply_along_axis() will be applied by each
    # worker (any non-zero axis number would work, so as to allow the use
    # of `np.array_split()`, which is only done on axis 0):
    effective_axis = 1 if axis == 0 else axis
    if effective_axis != axis:
        arr = arr.swapaxes(axis, effective_axis)

    # Chunks for the mapping (only a few chunks):
    chunks = [(func1d, effective_axis, sub_arr, args, kwargs)
              for sub_arr in np.array_split(arr, multiprocessing.cpu_count())]

    pool = multiprocessing.Pool()
    individual_results = pool.map(unpacking_apply_along_axis, chunks)
    # Freeing the workers:
    pool.close()
    pool.join()

    return np.concatenate(individual_results)


def get_most_popular_label(frame_indices: np.array, *args, **kwargs) -> int:
    map_labels = kwargs['map_labels']
    labels_arr = map_labels[frame_indices]
    # use '-' for labels to set 0 as the last label to help argmax to choose better
    values, counts = np.unique(-labels_arr, return_counts=True)
    if len(values) == 1:
        return -values[0]

    if 0 in values:
        counts[values == 0] -= np.count_nonzero(frame_indices == (map_labels.size - 1))

    return -values[np.argmax(counts)]


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
    # colors = np.concatenate([np.asarray([[0,0,0]]), np.random.rand(np.max(labels_unique), 3)])
    # map_colors = colors[map_labels]
    # map_pcd.colors = o3d.utility.Vector3dVector(map_colors)
    # mapped_frame_pcd.paint_uniform_color([1, 0, 0])
    # o3d.visualization.draw_geometries([map_pcd, mapped_frame_pcd])
    # vis = o3d.visualization.VisualizerWithEditing()
    # vis.create_window()
    # vis.add_geometry(map_pcd)
    # vis.add_geometry(mapped_frame_pcd)
    # vis.run()  # user picks points
    # vis.destroy_window()
    # picked = vis.get_picked_points()
    # print(np.asarray(map_pcd.points)[picked[0]])
    # print(np.asarray(map_pcd.points)[picked[1]])
    # print(map_labels[picked[0]])

    # points with no reference will be marked with len(mapped_frame_pcd) index, so add zero to this index
    map_labels = np.concatenate([map_labels, np.asarray([0])])

    frame_indices_in_map = map_kd_tree.query(
        np.asarray(mapped_frame_pcd.points),
        distance_upper_bound=0.2,
        k=20
    )[1]

    frame_labels = parallel_apply_along_axis(
        get_most_popular_label,
        axis=1,
        arr=frame_indices_in_map,
        map_labels=map_labels
    )

    # return map_labels[frame_indices_in_map], frame_indices_in_map
    return frame_labels


if __name__ == "__main__":
    data_path = sys.argv[1]
    map_parts_path = sys.argv[2]
    path_to_poses = sys.argv[3]
    path_to_calib = sys.argv[4]
    output_path = sys.argv[5]
    debug = True
    debug_miss_clicks = False

    loader = KittiLoader(data_path)
    poses = load_poses(path_to_poses)
    calib_matrix = load_calib_matrix(path_to_calib)

    map_pcd, map_labels = build_map(map_parts_path)
    # map_labels = dbscan_labels(map_pcd, map_labels)

    if debug_miss_clicks:
        sys.exit()

    map_kd_tree = KDTree(np.asarray(map_pcd.points))

    if debug:
        high = loader.get_frame_count()
        control_frame_ids = np.concatenate([np.random.randint(low=0, high=high, size=99), np.asarray([31])])
        # control_frame_ids = np.asarray([95, 118, 335, 413, 779, 880, 1025, 1502, 1530, 1541, 1552, 1554, 1557, 1642, 1649, 1983, 2326, 2349, 2354, 2357, 2366, 2391, 2395, 2403, 2460, 2461, 2479, 2680, 3300, 3312, 3317, 3322, 3374, 3421, 4050, 4154, 4193])
        # control_frame_ids = np.asarray([1])

    for frame_id in range(loader.get_frame_count()):
        # if frame_id not in control_frame_ids:
        #     continue
        frame_pcd = loader.read_pcd(frame_id)
        transform_matrix = poses[frame_id]
        # frame_labels, frame_indices_in_map = annotate_frame_with_map(frame_pcd, map_kd_tree, map_labels, transform_matrix, calib_matrix)
        frame_labels = annotate_frame_with_map(frame_pcd, map_kd_tree, map_labels, transform_matrix, calib_matrix)
        # frame_labels_ref = frame_labels[np.where(frame_indices_in_map != map_labels.size)[0]]
        # frame_indices_in_map = frame_indices_in_map[np.where(frame_indices_in_map != map_labels.size)[0]]
        output_filename = "label-{:06d}.npy".format(frame_id)
        np.save(os.path.join(output_path, output_filename), frame_labels)

        if debug and frame_id in control_frame_ids:
            pcd_filename = "{:06d}.pcd".format(frame_id)
            ref_pcd_filename = "ref_{}".format(pcd_filename)

            # _, unique_indices = np.unique(frame_indices_in_map, return_index=True)
            # ref_pcd = map_pcd.select_by_index(frame_indices_in_map[unique_indices])
            # ref_labels = frame_labels_ref[unique_indices]

            visualize_pcd_labels(frame_pcd, frame_labels, os.path.join(output_path, pcd_filename))
            # visualize_pcd_labels(ref_pcd, ref_labels, os.path.join(output_path, ref_pcd_filename))
            print("Frame {} is debug choice!".format(frame_id))

        print("Frame {} is ready!".format(frame_id))
