import os
import sys

import numpy as np
import open3d as o3d
from pykdtree.kdtree import KDTree

from src.annotations.sse.FIC import FIC
from src.annotations.sse.LZW import LZW


def get_matrix_from_kitti_file(line: str) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :4] = np.array(list(map(float, line.rstrip().split(" ")))).reshape(3, 4)

    return matrix


def load_poses(path):
    # Read and parse the poses
    poses = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            transform_matrix = get_matrix_from_kitti_file(line)
            poses.append(transform_matrix)

    return poses


def load_calib_matrix(path):
    with open(path) as file:
        return get_matrix_from_kitti_file(file.readlines()[4][4:])


class KittiLoader:
    def __init__(self, path):
        cloud_filenames = os.listdir(path)
        self.clouds = [os.path.join(path, filename) for filename in cloud_filenames]

    def get_frame_count(self) -> int:
        return len(self.clouds)

    def read_pcd(self, frame_num) -> o3d.geometry.PointCloud:
        cloud_path = self.clouds[frame_num]
        pcd_points = np.fromfile(cloud_path, dtype=np.float32).reshape(-1, 4)
        pcd = o3d.geometry.PointCloud()
        # data contains [x, y, z, reflectance] for each point -- we skip the last one
        pcd.points = o3d.utility.Vector3dVector(pcd_points[:, :3])

        return pcd


class SSEAnnotation:
    def __init__(self, path):
        self.path = path

    def load_labels(self) -> np.array:
        with open(self.path, "rb") as label_file:
            data = bytearray(label_file.read())
            labels_string = LZW.decompress(FIC.decompress(data))
            labels_string = labels_string[1:-1]  # skip []
            return np.asarray(list(map(lambda x: int(x), labels_string.split(","))))

    def get_all_planes_ids(self) -> list:
        loaded_labels = self.load_labels()
        unique_ids = np.unique(loaded_labels)
        planes_ids = []
        for plane_id in unique_ids:
            # 0 label is for unsegmented areas
            if plane_id == 0:
                continue
            plane_indices = np.where(loaded_labels == plane_id)[0]
            planes_ids.append(plane_indices)

        return planes_ids

    @staticmethod
    def save_to_file(labels, common_filename):
        labels_filename = "{}.labels".format(common_filename)
        objects_filename = "{}.objects".format(common_filename)
        labels_string = "[" + ",".join([str(label) for label in labels]) + "]"
        with open(labels_filename, "wb") as label_file:
            compressed_labels_string = FIC.compress(LZW.compress(labels_string))
            label_file.write(b"".join(compressed_labels_string))
        with open(objects_filename, "wb") as object_file:
            compressed_objects_string = FIC.compress(LZW.compress("[]"))
            object_file.write(b"".join(compressed_objects_string))


def cloud_to_map(
        pcd: o3d.geometry.PointCloud,
        transform_matrix: np.array,
        calib_matrix: np.array
) -> o3d.geometry.PointCloud:
    transform_matrix = transform_matrix @ calib_matrix
    return pcd.transform(transform_matrix)


def get_annot_path_pairs(annot_path):
    filenames = os.listdir(annot_path)
    data_filenames = filter(lambda x: x.endswith(".pcd"), filenames)
    labels_filenames = filter(lambda x: x.endswith(".labels"), filenames)
    data_filenames = sorted(data_filenames, key=lambda x: int(x.split("_")[1]))
    labels_filenames = sorted(labels_filenames, key=lambda x: int(x.split("_")[1]))
    data_filenames = list(map(lambda x: os.path.join(annot_path, x), data_filenames))
    labels_filenames = list(map(lambda x: os.path.join(annot_path, x), labels_filenames))

    return zip(data_filenames, labels_filenames)


def map_annotations_to_map(
        map_pcd: o3d.geometry.PointCloud,
        loader: KittiLoader,
        calib_matrix: np.array,
        map_poses: np.array,
        path_to_annot: str,
        path_to_annot_poses: str,
        labels_filename: str
) -> np.array:
    map_points_count = np.asarray(map_pcd.points).shape[0]
    map_tree = KDTree(np.asarray(map_pcd.points))
    last_used_id = 0  # skip 0 for void label
    map_annot_indices = np.zeros(map_points_count, dtype=np.intc)
    frames_poses = load_poses(path_to_annot_poses)
    for (data_path, annot_path) in get_annot_path_pairs(path_to_annot):
        segment_pcd = o3d.io.read_point_cloud(data_path)
        annot = SSEAnnotation(annot_path)
        segment_labels = annot.load_labels()
        is_zero_label = segment_labels != 0
        segment_labels = (segment_labels + last_used_id) * is_zero_label
        first_used_id = last_used_id + 1
        last_used_id = max(np.max(segment_labels), last_used_id)
        start_frame = int(os.path.split(data_path)[-1].split("_")[1])
        end_frame = int(os.path.split(data_path)[-1].split("_")[2])
        segment_tree = KDTree(np.asarray(segment_pcd.points))
        for frame_num in range(start_frame, min(end_frame + 1, loader.get_frame_count() - 1), 10):
            frame_pcd = loader.read_pcd(frame_num)
            frame_segment_transform_matrix = frames_poses[frame_num]
            frame_pcd_mapped_to_segment = cloud_to_map(frame_pcd, frame_segment_transform_matrix, calib_matrix)
            frame_segment_indices = segment_tree.query(np.asarray(frame_pcd_mapped_to_segment.points))[1]
            frame_labels = segment_labels[frame_segment_indices]

            frame_pcd = loader.read_pcd(frame_num)
            frame_map_transform_matrix = map_poses[frame_num]
            frame_pcd_mapped_to_map = cloud_to_map(frame_pcd, frame_map_transform_matrix, calib_matrix)
            frame_map_indices = map_tree.query(np.asarray(frame_pcd_mapped_to_map.points), distance_upper_bound=0.2)[1]
            frame_labels_not_null_indices = np.where(
                np.logical_and(frame_labels > 0, frame_map_indices < map_points_count)
            )[0]
            frame_map_indices_not_null = frame_map_indices[frame_labels_not_null_indices]
            map_annot_indices[frame_map_indices_not_null] = frame_labels[frame_labels_not_null_indices]

        print("Annotations from {0} loaded! Use plane ids: from {1} to {2}".format(data_path, first_used_id, last_used_id))

    if labels_filename.endswith(".npy"):
        np.save(labels_filename, map_annot_indices)
    else:
        SSEAnnotation.save_to_file(map_annot_indices, labels_filename)
    print("Last used plane id: {}".format(last_used_id - 1))

    return map_annot_indices


def get_bboxes_bounds_of_parts(full_bbox: o3d.geometry.AxisAlignedBoundingBox, step: int) -> np.array:
    min_x, min_y, min_z = full_bbox.get_min_bound()
    max_x, max_y, max_z = full_bbox.get_max_bound()
    bboxes_bounds = []
    for x in np.arange(min_x, max_x, step):
        for z in np.arange(min_z, max_z, step):  # we use z as second horizontal axis because of kitty orientation
            bbox_bounds = [(x, min_y, z), (x + step, max_y, z + step)]
            bboxes_bounds.append(bbox_bounds)

    return bboxes_bounds


def get_centers_pos_in_map(loader: KittiLoader, poses, calib_matrix: np.array) -> np.array:
    res = []
    for frame_id in range(loader.get_frame_count()):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray([[0., 0., 0.]]))
        transform_matrix = poses[frame_id] @ calib_matrix
        pcd.transform(transform_matrix)
        res.append(np.asarray(pcd.points)[0, :])

    return np.asarray(res)


if __name__ == "__main__":
    path_to_data = sys.argv[1]
    path_to_poses = sys.argv[2]
    path_to_calib = sys.argv[3]
    path_to_annot = sys.argv[4]
    path_to_annot_poses = sys.argv[5]
    parts_output_path = sys.argv[6]
    debug = False
    calc_included_frames = True

    map_filename = "map.pcd"
    annot_filename = "map.pcd.labels.npy"

    loader = KittiLoader(path_to_data)
    poses = load_poses(path_to_poses)
    calib_matrix = load_calib_matrix(path_to_calib)

    if os.path.isfile(map_filename):
        map_pcd = o3d.io.read_point_cloud(map_filename)
    else:
        map_pcd = o3d.geometry.PointCloud()

        for cloud_id in range(0, loader.get_frame_count(), 10):
            cloud = loader.read_pcd(cloud_id)
            mapped_cloud = cloud_to_map(cloud, poses[cloud_id], calib_matrix)
            map_pcd += mapped_cloud
            print("Done {}".format(cloud_id))

        down_sampled_map_pcd = map_pcd.voxel_down_sample(0.2)
        o3d.io.write_point_cloud("map.pcd", down_sampled_map_pcd)
        print("Built map!")

    if debug:
        low_map = map_pcd.voxel_down_sample(1)
        o3d.io.write_point_cloud("super_low_map.pcd", low_map)
        print("Super low map ready - it has {} points".format(np.asarray(low_map.points).shape[0]))

    if os.path.isfile(annot_filename):
        map_annot_labels = np.load(annot_filename)
    else:
        map_annot_labels = map_annotations_to_map(
            map_pcd,
            loader,
            calib_matrix,
            poses,
            path_to_annot,
            path_to_annot_poses,
            annot_filename
        )
    # map_annot_labels = np.zeros((np.asarray(map_pcd.points).shape[0]), dtype=np.intc)
    print("Annotations loaded to map")

    if debug:
        map_annotations_to_map(
            low_map,
            loader,
            calib_matrix,
            poses,
            path_to_annot,
            path_to_annot_poses,
            "super_low_map.pcd"
        )
        print("Annotations loaded to low map")

    map_bbox = map_pcd.get_axis_aligned_bounding_box()
    bbox_points = np.asarray(map_bbox.get_box_points())
    print(bbox_points)
    bboxes_bounds = get_bboxes_bounds_of_parts(map_bbox, step=150)
    map_points = np.asarray(map_pcd.points)
    centers_points = get_centers_pos_in_map(loader, poses, calib_matrix)
    for bbox_bounds in bboxes_bounds:
        min_x, min_y, min_z = bbox_bounds[0]
        max_x, max_y, max_z = bbox_bounds[1]
        bbox_str = "{0}_{1}_{2}_{3}".format(int(min_x), int(max_x), int(min_z), int(max_z))

        fit_in_part = np.zeros_like(map_points, dtype=bool)
        fit_in_part[:, 0] = np.logical_and(map_points[:, 0] < max_x, map_points[:, 0] >= min_x)
        fit_in_part[:, 1] = np.logical_and(map_points[:, 1] < max_y, map_points[:, 1] >= min_y)
        fit_in_part[:, 2] = np.logical_and(map_points[:, 2] < max_z, map_points[:, 2] >= min_z)
        part_indices = np.where(np.all(fit_in_part, axis=-1))[0]

        if part_indices.size < 1000:
            print("BBox {} is too small!".format(bbox_str))
            continue

        part_pcd = map_pcd.select_by_index(part_indices)
        part_annot = map_annot_labels[part_indices]
        # Fix labels -- we need unique only among this part to prevent label overflow in sse
        # Add zero to the beginning because we always need void label to have zero label
        part_annot = np.concatenate([np.asarray([0]), part_annot])
        unique_labels, unique_labels_indices = np.unique(part_annot, return_inverse=True)
        # remove added zero
        part_annot = unique_labels_indices[1:]

        bbox_str_with_unique = "{0}_u{1}".format(bbox_str, unique_labels.size)

        if calc_included_frames:
            centers_fit_in_part = np.zeros_like(centers_points, dtype=bool)
            centers_fit_in_part[:, 0] = np.logical_and(centers_points[:, 0] < max_x, centers_points[:, 0] >= min_x)
            centers_fit_in_part[:, 1] = np.logical_and(centers_points[:, 1] < max_y, centers_points[:, 1] >= min_y)
            centers_fit_in_part[:, 2] = np.logical_and(centers_points[:, 2] < max_z, centers_points[:, 2] >= min_z)
            fit_centers_indices = np.where(np.all(centers_fit_in_part, axis=-1))[0]
            np.savetxt(
                os.path.join(parts_output_path, "part_{}.frames.txt".format(bbox_str_with_unique)),
                fit_centers_indices,
                fmt='%i'
            )

        part_pcd_filename = "part_{}.pcd".format(bbox_str_with_unique)
        part_pcd_filename = os.path.join(parts_output_path, part_pcd_filename)
        o3d.io.write_point_cloud(part_pcd_filename, part_pcd)
        SSEAnnotation.save_to_file(part_annot, part_pcd_filename)
        print("BBox {} ready!".format(bbox_str_with_unique))

    print("All parts prepared!")
