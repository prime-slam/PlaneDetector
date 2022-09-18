import argparse
import multiprocessing
import os.path

import numpy as np
import open3d as o3d
import ast

from pykdtree.kdtree import KDTree

PLANAR_IDS = {
    6: 1,
    7: 1,
    8: 2
}


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


def pcd_from_carla_line(line: str) -> (o3d.geometry.PointCloud, np.array):
    points_packed = ast.literal_eval(line.split(",|,")[1])
    points = np.asarray([point_label_id[0] for point_label_id in points_packed])
    labels = [point_label_id[1] for point_label_id in points_packed]
    labels = np.asarray(list(map(lambda x: PLANAR_IDS[x] if x in PLANAR_IDS else 0, labels)))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    return pcd, labels


def build_map(data_path) -> o3d.geometry.PointCloud:
    map_pcd = o3d.geometry.PointCloud()
    with open(data_path) as data_file:
        for index, line in enumerate(data_file):
            if index % 10 != 0:
                continue

            frame_pcd, _ = pcd_from_carla_line(line)
            map_pcd += frame_pcd
            print("Cloud {} loaded!".format(index + 1))

        map_pcd = map_pcd.voxel_down_sample(0.2)

    return map_pcd


def annotate_map(map_pcd: o3d.geometry.PointCloud, annot_path: str, data_path: str) -> np.array:
    mesh_names = [filename[:-4] for filename in os.listdir(annot_path) if filename.endswith(".pcd")]
    map_tree = KDTree(np.asarray(map_pcd.points))
    map_points_count = np.asarray(map_pcd.points).shape[0]
    map_labels = np.zeros(map_points_count, dtype=int)
    max_used_label = -1

    with open(data_path) as data_file:
        for index, line in enumerate(data_file):
            if index % 10 != 0:
                continue

            frame_pcd, frame_labels = pcd_from_carla_line(line)
            frame_points = np.asarray(frame_pcd.points)
            frame_map_indices = map_tree.query(frame_points, distance_upper_bound=0.2)[1]
            frame_labels_not_null_indices = np.where(frame_map_indices < map_points_count)[0]
            frame_map_indices_not_null = frame_map_indices[frame_labels_not_null_indices]
            map_labels[frame_map_indices_not_null] = frame_labels[frame_labels_not_null_indices]

            print("Cloud {} annotations loaded!".format(index + 1))

    max_used_label = np.max(map_labels)

    for index, mesh_name in enumerate(mesh_names):
        mesh_pcd_filename = "{}.pcd".format(mesh_name)
        mesh_labels_filename = "{}.npy".format(mesh_name)
        mesh_pcd = o3d.io.read_point_cloud(os.path.join(annot_path, mesh_pcd_filename))
        mesh_labels = np.load(os.path.join(annot_path, mesh_labels_filename))
        mesh_labels += max_used_label + 1
        prev_max_used_label = max_used_label
        max_used_label = max(max_used_label, np.max(mesh_labels))

        mesh_points = np.asarray(mesh_pcd.points)
        # Swap y and z axis for UE coordinates and go from cm to metres
        mesh_points_z = mesh_points[:, 2].copy()
        mesh_points[:, 2] = mesh_points[:, 1]
        mesh_points[:, 1] = mesh_points_z
        mesh_points /= 100

        mesh_map_indices = map_tree.query(mesh_points, distance_upper_bound=0.2)[1]
        mesh_labels_not_null_indices = np.where(mesh_map_indices < map_points_count)[0]
        mesh_map_indices_not_null = mesh_map_indices[mesh_labels_not_null_indices]
        map_labels[mesh_map_indices_not_null] = mesh_labels[mesh_labels_not_null_indices]

        print("{0} is ready! ({1}/{2})".format(mesh_name, index + 1, len(mesh_names)))
        print("{0} mesh labels: ({1}, {2})".format(mesh_name, prev_max_used_label + 1, max_used_label))

    return map_labels


def annotate_frames(data_path: str, output_path: str, map_pcd: o3d.geometry.PointCloud, map_labels: np.array):
    map_kd_tree = KDTree(np.asarray(map_pcd.points))
    with open(data_path) as data_file:
        for index, line in enumerate(data_file):
            frame_pcd, _ = pcd_from_carla_line(line)
            frame_labels = annotate_frame_with_map(frame_pcd, map_kd_tree, map_labels, map_pcd)
            o3d.io.write_point_cloud(os.path.join(output_path, "{:06d}.pcd".format(index)), frame_pcd)
            np.save(os.path.join(output_path, "{:06d}.npy".format(index)), frame_labels)


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


def annotate_frame_with_map(frame_pcd: o3d.geometry.PointCloud, map_kd_tree: KDTree, map_labels: np.array, map_pcd) -> np.array:

    # Uncomment this if you need to show mapped frame on map
    # -------------------------
    # map_pcd.paint_uniform_color([0, 0, 0])
    # labels_unique = np.unique(map_labels)
    # colors = np.concatenate([np.asarray([[0,0,0]]), np.random.rand(np.max(labels_unique), 3)])
    # map_colors = colors[map_labels]
    # map_pcd.colors = o3d.utility.Vector3dVector(map_colors)
    # frame_pcd.paint_uniform_color([1, 0, 0])
    # o3d.visualization.draw_geometries([map_pcd, frame_pcd])
    # vis = o3d.visualization.VisualizerWithEditing()
    # vis.create_window()
    # vis.add_geometry(map_pcd)
    # vis.add_geometry(frame_pcd)
    # vis.run()  # user picks points
    # vis.destroy_window()
    # picked = vis.get_picked_points()
    # print(np.asarray(map_pcd.points)[picked[0]])
    # print(np.asarray(map_pcd.points)[picked[1]])
    # print(map_labels[picked[0]])
    # -------------------------

    # points with no reference will be marked with len(mapped_frame_pcd) index, so add zero to this index
    map_labels = np.concatenate([map_labels, np.asarray([0])])

    frame_indices_in_map = map_kd_tree.query(
        np.asarray(frame_pcd.points),
        distance_upper_bound=0.2,
        k=20
    )[1]

    frame_labels = parallel_apply_along_axis(
        get_most_popular_label,
        axis=1,
        arr=frame_indices_in_map,
        map_labels=map_labels
    )

    return frame_labels


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
        map_labels = annotate_map(map_pcd, annot_path, data_path)
        np.save(annot_filename, map_labels)

    # visualize_pcd_labels(map_pcd, map_labels)

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
