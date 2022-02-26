import os
from shutil import rmtree

import cv2
import docker
import numpy as np
import open3d as o3d
from pypcd import pypcd

from src.metrics import metrics
from src.metrics.metrics import multi_value, mean
from src.parser import loaders, create_parser

UNSEGMENTED_COLOR = np.asarray([0, 0, 0], dtype=int)

algos = {
    "storm-irit": "akornilova/storm_irit:1.0"
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

annot_sorters = {
    'tum': lambda x: x,
    'icl_tum': lambda x: int(x),
    'icl': lambda x: x
}


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
    # path_to_input = os.path.join(current_dir_abs, CLOUDS_DIR)
    # path_to_output = os.path.join(current_dir_abs, PREDICTIONS_DIR)

    # for filename in os.listdir(path_to_input):
    #     folder_path = os.path.join(path_to_output, filename[:-4])
    #     os.mkdir(folder_path)
    #     pcd = o3d.io.read_point_cloud(os.path.join(path_to_input, filename))
    #     o3d.io.write_point_cloud(os.path.join(folder_path, filename), pcd)
    #     np.save(
    #         os.path.join(folder_path, "{}.npy".format(filename[:-4])),
    #         np.ones(np.asarray(pcd.points).shape[0], dtype=int)
    #     )
    client = docker.from_env()
    docker_image_name = algos[algo_name]
    container = client.containers.run(
        docker_image_name,
        volumes=[
            '{}:/app/build/input'.format(CLOUDS_DIR),
            '{}:/app/build/output'.format(PREDICTIONS_DIR)
        ],
        detach=True
    )
    for line in container.logs(stream=True):
        print(line.strip())


def prepare_clouds(dataset_path: str, loader_name: str, step: int):
    if os.path.exists(CLOUDS_DIR):
        rmtree(CLOUDS_DIR)
    os.mkdir(CLOUDS_DIR)

    loader = loaders[loader_name](dataset_path)
    for depth_frame_num in range(0, loader.get_frame_count(), step):
        pcd_points = loader.read_pcd(depth_frame_num)
        cloud_filepath = os.path.join(CLOUDS_DIR, "{:04d}.pcd".format(depth_frame_num))
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(pcd_points)
        # o3d.io.write_point_cloud(cloud_filepath, pcd)
        pc = pypcd.make_xyz_rgb_point_cloud(pcd_points_rgba)
        pc.width = loader.cam_intrinsics.width
        pc.height = loader.cam_intrinsics.height
        pc.save_pcd(cloud_filepath, compression='binary')


def get_filepaths_for_dir(dir_path: str):
    filenames = os.listdir(dir_path)
    file_paths = [os.path.join(dir_path, filename) for filename in filenames]
    return file_paths


def get_path_to_frames(annot_path: str, loader_name: str) -> [(str, str)]:
    sort_by = annot_sorters[loader_name]
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


def measure_algo(algo_name: str, annot_path: str, loader_name: str, log_file):
    metrics_average = {metric.__name__: 0 for metric in all_plane_metrics}
    dump_info("-------------Results for algo: '{}'--------------".format(algo_name), log_file)
    predict_labels(algo_name)

    for cloud_frame_path, annot_frame_path, prediction_group in get_path_to_frames(annot_path, loader_name):
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
    parser = create_parser()
    args = parser.parse_args()
    CLOUDS_DIR = os.path.join(args.workdir, CLOUDS_DIR)
    PREDICTIONS_DIR = os.path.join(args.workdir, PREDICTIONS_DIR)

    prepare_clouds(args.dataset_path, args.loader, 50)

    with open( os.path.join(args.workdir, "results.txt"), 'w') as log_file:
        for algo_name in algos.keys():
            measure_algo(algo_name, args.annotations_path, args.loader, log_file)
