import os
import sys

import cv2
import docker
import numpy as np
import open3d as o3d

from scripts.eval.metrics import metrics
from scripts.eval.metrics.metrics import multi_value, mean


class CameraIntrinsics:
    def __init__(self, fx, fy, cx, cy, factor):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.factor = factor


UNSEGMENTED_COLOR = np.asarray([0, 0, 0], dtype=int)

algo_names = [
    ""
]

all_plane_metrics = [
    metrics.iou,
    metrics.dice,
    metrics.precision,
    metrics.recall,
    metrics.fScore
]


def read_pcd_from_depth(depth_frame_path: str, camera_intrinsics: CameraIntrinsics) -> np.array:
    depth_image = cv2.imread(depth_frame_path, cv2.IMREAD_ANYDEPTH)
    image_height, image_width = depth_image.shape[:2]
    pcd_points = np.zeros((image_height * image_width, 3))

    column_indices = np.tile(np.arange(image_width), (image_height, 1)).flatten()
    row_indices = np.transpose(np.tile(np.arange(image_height), (image_width, 1))).flatten()

    pcd_points[:, 2] = depth_image.flatten() / camera_intrinsics.factor
    pcd_points[:, 0] = (column_indices - camera_intrinsics.cx) * pcd_points[:, 2] / camera_intrinsics.fx
    pcd_points[:, 1] = (row_indices - camera_intrinsics.cy) * pcd_points[:, 2] / camera_intrinsics.fy

    return pcd_points


def read_labels(annot_frame_path: str) -> np.array:
    annot_image = cv2.imread(annot_frame_path)
    label_colors = annot_image.reshape((annot_image.shape[0] * annot_image.shape[1], 3))
    labels = np.zeros(label_colors.shape[0], dtype=int)

    unique_colors = np.unique(label_colors, axis=0)
    for index, color in enumerate(unique_colors):
        color_indices = np.where(np.all(label_colors == color, axis=-1))[0]
        if not np.array_equal(color, UNSEGMENTED_COLOR):
            labels[color_indices] = index + 1

    return labels


def predict_labels(pcd_points: np.array, algo_name: str) -> np.array:
    if not os.path.exists("input"):
        os.mkdir("input")
    if not os.path.exists("output"):
        os.mkdir("output")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_points)
    o3d.io.write_point_cloud(os.path.join("input", "data.pcd"), pcd)
    current_dir_abs = os.path.abspath(os.path.curdir)
    path_to_input = os.path.join(current_dir_abs, "input")
    path_to_output = os.path.join(current_dir_abs, "output")

    client = docker.from_env()
    container = client.containers.run(
        "cpf_segmentation:1.0",
        "data.pcd",
        volumes=[
            '{}:/app/build/input'.format(path_to_input),
            '{}:/app/build/output'.format(path_to_output)
        ],
        detach=True
    )
    for line in container.logs(stream=True):
        print(line.strip())

    result_file_path = os.path.join("output", "data_seg.pcd")
    with open(result_file_path) as result_file:
        lines = result_file.readlines()
        labels = np.asarray(list(map(lambda x: int(x.split(" ")[3]), lines[11:])))
    # segmented_pcd = o3d.io.read_point_cloud()
    # label_colors = np.asarray(segmented_pcd.colors)
    # labels = np.zeros(label_colors.shape[0], dtype=int)

    # return np.zeros(pcd_points.shape[0], dtype=int)
    return labels


def get_path_to_frames(depth_path: str, annot_path: str) -> [(str, str)]:
    depth_filenames = os.listdir(depth_path)
    depth_file_paths = [os.path.join(depth_path, filename) for filename in depth_filenames]
    annot_filenames = os.listdir(annot_path)
    annot_file_paths = [os.path.join(annot_path, filename) for filename in annot_filenames]

    return zip(depth_file_paths, annot_file_paths)


def visualize_pcd_labels(pcd_points: np.array, labels: np.array, filename: str = None):
    colors = np.concatenate([UNSEGMENTED_COLOR.astype(dtype=float).reshape(-1, 3), np.random.rand(np.max(labels), 3)])
    pcd_for_vis = o3d.geometry.PointCloud()
    pcd_for_vis.points = o3d.utility.Vector3dVector(pcd_points)
    pcd_for_vis.paint_uniform_color([0, 0, 0])
    pcd_for_vis.colors = o3d.utility.Vector3dVector(colors[labels])
    if filename is None:
        o3d.visualization.draw_geometries([pcd_for_vis])
    else:
        o3d.io.write_point_cloud(filename, pcd_for_vis)


if __name__ == "__main__":
    depth_path = sys.argv[1]
    annot_path = sys.argv[2]
    # output_path = sys.argv[3]

    # for icl_tum format
    camera_intrinsics = CameraIntrinsics(
        fx=481.20,  # X-axis focal length
        fy=-480.00,  # Y-axis focal length
        cx=319.50,  # X-axis principle point
        cy=239.50,  # Y-axis principle point
        factor=5000  # for the 16-bit PNG files
    )

    for depth_frame_path, annot_frame_path in get_path_to_frames(depth_path, annot_path):
        pcd_points = read_pcd_from_depth(depth_frame_path, camera_intrinsics)
        gt_labels = read_labels(annot_frame_path)

        # remove zero depth (for TUM)
        zero_depth_mask = np.sum(pcd_points == 0, axis=-1) == 3
        pcd_points = pcd_points[~zero_depth_mask]
        gt_labels = gt_labels[~zero_depth_mask]

        # visualize_pcd_labels(pcd_points, gt_labels)

        print("Processing {} frame".format(os.path.split(depth_frame_path)[-1]))
        for algo_name in algo_names:
            print("Results for algo: '{}'".format(algo_name))
            pred_labels = predict_labels(pcd_points, algo_name)
            visualize_pcd_labels(pcd_points, pred_labels)

            # print(multi_value(pcd_points, pred_labels, gt_labels))

            for metric in all_plane_metrics:
                print("Mean {0}: {1}".format(metric.__name__, mean(pcd_points, pred_labels, gt_labels, metric)))

        break
