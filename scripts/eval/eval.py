import os
import sys
from shutil import rmtree

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

algos = {
    "ddpff": "ddpff:1.0"
}

all_plane_metrics = [
    metrics.iou,
    metrics.dice,
    metrics.precision,
    metrics.recall,
    metrics.fScore
]

CLOUDS_DIR = "input"
PREDICTIONS_DIR = "output"


def sort_by(x):
    return int(x)


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


def predict_labels(algo_name: str):
    if os.path.exists(PREDICTIONS_DIR):
        rmtree(PREDICTIONS_DIR)
    os.mkdir(PREDICTIONS_DIR)

    current_dir_abs = os.path.abspath(os.path.curdir)
    path_to_input = os.path.join(current_dir_abs, CLOUDS_DIR)
    path_to_output = os.path.join(current_dir_abs, PREDICTIONS_DIR)

    client = docker.from_env()
    docker_image_name = algos[algo_name]
    container = client.containers.run(
        docker_image_name,
        volumes=[
            '{}:/app/build/input'.format(path_to_input),
            '{}:/app/build/output'.format(path_to_output)
        ],
        detach=True
    )
    for line in container.logs(stream=True):
        print(line.strip())


def prepare_clouds(depth_path: str):
    if os.path.exists(CLOUDS_DIR):
        rmtree(CLOUDS_DIR)
    os.mkdir(CLOUDS_DIR)

    for depth_frame_path in get_filepaths_for_dir(depth_path):
        frame_name = os.path.split(depth_frame_path)[-1][:-4]
        pcd_points = read_pcd_from_depth(depth_frame_path, camera_intrinsics)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_points)
        o3d.io.write_point_cloud(os.path.join(CLOUDS_DIR, "{}.pcd".format(frame_name)), pcd)


def get_filepaths_for_dir(dir_path: str):
    filenames = os.listdir(dir_path)
    file_paths = [os.path.join(dir_path, filename) for filename in filenames]
    return file_paths


def get_path_to_frames(annot_path: str) -> [(str, str)]:
    cloud_file_paths = sorted(get_filepaths_for_dir(CLOUDS_DIR), key=lambda x: sort_by(os.path.split(x)[-1][:-4]))
    prediction_folders = sorted(get_filepaths_for_dir(PREDICTIONS_DIR), key=lambda x: sort_by(os.path.split(x)[-1]))
    prediction_grouped_file_paths = [
        list(filter(lambda x: x.endswith(".npy"), get_filepaths_for_dir(folder))) for folder in prediction_folders
    ]
    annot_file_paths = sorted(get_filepaths_for_dir(annot_path), key=lambda x: sort_by(os.path.split(x)[-1][:-4]))

    return zip(cloud_file_paths, annot_file_paths, prediction_grouped_file_paths)


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


def dump_info(info, file=None):
    print(info)
    if file is not None:
        print(info, file=file)


def measure_algo(algo_name: str, annot_path: str, log_file):
    metrics_average = {metric.__name__: 0 for metric in all_plane_metrics}
    dump_info("-------------Results for algo: '{}'--------------".format(algo_name), log_file)
    predict_labels(algo_name)

    for cloud_frame_path, annot_frame_path, prediction_group in get_path_to_frames(annot_path):
        pcd_points = np.asarray(o3d.io.read_point_cloud(cloud_frame_path).points)
        gt_labels = read_labels(annot_frame_path)

        # remove zero depth (for TUM)
        zero_depth_mask = np.sum(pcd_points == 0, axis=-1) == 3
        pcd_points = pcd_points[~zero_depth_mask]
        gt_labels = gt_labels[~zero_depth_mask]

        # Find the best annotation from algorithm for frame
        max_mean_index = 0
        max_mean = 0
        for prediction_index, prediction_frame_path in enumerate(prediction_group):
            pred_labels = np.load(prediction_frame_path)
            # remove zero depth (for TUM)
            pred_labels = pred_labels[~zero_depth_mask]

            metric_res = mean(pcd_points, pred_labels, gt_labels, metrics.iou)
            if metric_res > max_mean:
                max_mean = metric_res
                max_mean_index = prediction_index

        # Load chosen predictions
        chosen_prediction_path = prediction_group[max_mean_index]
        pred_labels = np.load(chosen_prediction_path)
        pred_labels = pred_labels[~zero_depth_mask]
        # visualize_pcd_labels(pcd_points, pred_labels)

        # Print metrics results
        dump_info("********Result for frame: '{}'********".format(os.path.split(cloud_frame_path)[-1][:-4]), log_file)
        dump_info(multi_value(pcd_points, pred_labels, gt_labels), log_file)
        for metric in all_plane_metrics:
            metric_res = mean(pcd_points, pred_labels, gt_labels, metric)
            metrics_average[metric.__name__] += metric_res
            dump_info("Mean {0}: {1}".format(metric.__name__, metric_res), log_file)

    dump_info("--------------------------------------------------------", log_file)
    dump_info("----------------Average of algo: '{}'----------------".format(algo_name), log_file)
    for metric_name, sum_value in metrics_average.items():
        dump_info(
            "Average {0} for dataset is: {1}".format(metric_name, sum_value / len(os.listdir(CLOUDS_DIR))),
            log_file
        )

    dump_info("--------------------------------------------------------", log_file)
    dump_info("----------------End of algo: '{}'--------------------".format(algo_name), log_file)
    dump_info("--------------------------------------------------------", log_file)


if __name__ == "__main__":
    depth_path = sys.argv[1]
    annot_path = sys.argv[2]

    # for icl_tum format
    camera_intrinsics = CameraIntrinsics(
        fx=481.20,  # X-axis focal length
        fy=-480.00,  # Y-axis focal length
        cx=319.50,  # X-axis principle point
        cy=239.50,  # Y-axis principle point
        factor=5000  # for the 16-bit PNG files
    )

    prepare_clouds(depth_path)

    with open("results.txt", 'w') as log_file:
        for algo_name in algos.keys():
            measure_algo(algo_name, annot_path, log_file)
