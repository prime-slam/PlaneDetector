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
    def save_to_file(labels, filename):
        labels_string = "[" + ",".join([str(label) for label in labels]) + "]"
        with open(filename, "wb") as label_file:
            compressed_labels_string = FIC.compress(LZW.compress(labels_string))
            label_file.write(b"".join(compressed_labels_string))


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


def map_annotations_to_map(map_pcd: o3d.geometry.PointCloud, path_to_annot: str, labels_filename: str) -> np.array:
    map_points_count = np.asarray(map_pcd.points).shape[0]
    kd_tree = KDTree(np.asarray(map_pcd.points))
    next_free_id = 1  # skip 0 for void label
    map_annot_indices = np.zeros(map_points_count, dtype=np.intc)
    for (data_path, annot_path) in get_annot_path_pairs(path_to_annot):
        data_pcd = o3d.io.read_point_cloud(data_path)
        annot = SSEAnnotation(annot_path)
        planes_indices = annot.get_all_planes_ids()
        data_indices = kd_tree.query(np.asarray(data_pcd.points))[1]  # take only indices, skip distances
        for plane_indices in planes_indices:
            plane_id = next_free_id
            next_free_id += 1
            plane_map_indices = data_indices[plane_indices]
            map_annot_indices[plane_map_indices] = plane_id

    SSEAnnotation.save_to_file(map_annot_indices, labels_filename)
    print("Last used plane id: {}".format(next_free_id - 1))

    return map_annot_indices


def build_bbox(x_min, x_max, y_min, y_max, z_min, z_max):
    pts = []
    for x in [x_min, x_max]:
        for y in [y_min, y_max]:
            for z in [z_min, z_max]:
                pts.append([x, y, z])

    return np.asarray(pts)


def get_bboxes_of_parts(full_bbox: o3d.geometry.AxisAlignedBoundingBox) -> np.array:
    min_x, min_y, min_z = full_bbox.get_min_bound()
    max_x, max_y, max_z = full_bbox.get_max_bound()
    # min_y = np.min(full_bbox, axis=1)
    # max_x = np.min(full_bbox, axis=1)
    # max_y = np.min(full_bbox, axis=1)
    # min_z = np.min(full_bbox, axis=1)
    # max_z = np.min(full_bbox, axis=1)
    bboxes = []
    for x in np.arange(min_x, max_x, 150):
        for y in np.arange(min_y, max_y, 150):
            bbox_points = build_bbox(x, x + 150, y, y + 150, min_z, max_z)
            bbox_points_o3d = o3d.utility.Vector3dVector(bbox_points)
            bboxes.append(o3d.geometry.AxisAlignedBoundingBox.create_from_points(bbox_points_o3d))

    return bboxes


if __name__ == "__main__":
    path_to_data = sys.argv[1]
    path_to_poses = sys.argv[2]
    path_to_calib = sys.argv[3]
    path_to_annot = sys.argv[4]
    debug = False

    map_filename = "map.pcd"
    annot_filename = "map.pcd.labels"

    if os.path.isfile(map_filename):
        map_pcd = o3d.io.read_point_cloud(map_filename)
    else:
        loader = KittiLoader(path_to_data)
        poses = load_poses(path_to_poses)
        calib_matrix = load_calib_matrix(path_to_calib)
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
        annotation = SSEAnnotation(annot_filename)
        map_annot_indices = annotation.load_labels()
    else:
        map_annot_indices = map_annotations_to_map(map_pcd, path_to_annot, annot_filename)
    print("Annotations loaded to map")

    if debug:
        map_annotations_to_map(low_map, path_to_annot, "super_low_map.pcd.labels")
        print("Annotations loaded to low map")

    map_bbox = map_pcd.get_axis_aligned_bounding_box()
    bbox_points = np.asarray(map_bbox.get_box_points())
    print(bbox_points)
    bboxes = get_bboxes_of_parts(map_bbox)
    # for bbox in bboxes:
    #     part_pcd = map_pcd.crop(bbox)

