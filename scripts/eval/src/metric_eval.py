import argparse
import os

import cv2
import numpy as np
import open3d as o3d
import pandas as pd
from evops.metrics import mean, metrics, multi_value

annot_sorters = {
    'tum': lambda x: x,
    'icl_tum': lambda x: int(x),
    'icl': lambda x: x
}

UNSEGMENTED_COLOR = np.asarray([0, 0, 0], dtype=int)

all_plane_metrics = [
    metrics.iou,
    metrics.dice,
    metrics.precision,
    metrics.recall,
    metrics.fScore
]


def get_filepaths_for_dir(dir_path: str):
    filenames = os.listdir(dir_path)
    file_paths = [os.path.join(dir_path, filename) for filename in filenames]
    return file_paths


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


def get_path_to_frames(pred_path: str, annot_path: str, loader_name: str, annot_step: int) -> [(str, str)]:
    sort_by = annot_sorters[loader_name]
    prediction_folders = sorted(get_filepaths_for_dir(pred_path), key=lambda x: sort_by(os.path.split(x)[-1]))
    prediction_grouped_file_paths = [
        list(filter(lambda x: x.endswith(".npy"), get_filepaths_for_dir(folder))) for folder in prediction_folders
    ]
    cloud_grouped_file_paths = [
        list(filter(lambda x: x.endswith(".pcd"), get_filepaths_for_dir(folder))) for folder in prediction_folders
    ]
    annot_file_paths = sorted(get_filepaths_for_dir(annot_path), key=lambda x: sort_by(os.path.split(x)[-1][:-4]))
    annot_file_paths = [annot_file_paths[i] for i in range(0, len(annot_file_paths), annot_step)]

    return zip(cloud_grouped_file_paths, annot_file_paths, prediction_grouped_file_paths)


def measure_algo(pred_path: str, annot_path: str, loader_name: str, annot_step: int):
    pred_amount = len(os.listdir(annot_path))
    metrics_final_res = np.zeros((pred_amount, len(all_plane_metrics) + 6), dtype=float)

    multi_value_keys = ["precision", "recall", "under_segmented", "over_segmented", "missed", "noise"]
    mean_metric_names = ["mean_{}".format(metric.__name__) for metric in all_plane_metrics]
    multi_value_metric_names = ["multi_{}".format(name) for name in multi_value_keys]
    column_names = multi_value_metric_names + mean_metric_names

    for frame_index, data in enumerate(get_path_to_frames(pred_path, annot_path, loader_name, annot_step)):
        cloud_group, annot_frame_path, prediction_group = data
        cloud_frame_path = cloud_group[0]  # we use only xyz which are the same
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

            metric_res = metrics.mean(pcd_points, pred_labels, gt_labels, metrics.iou)
            if metric_res > max_mean:
                max_mean = metric_res
                max_mean_index = prediction_index

        # Load chosen predictions
        chosen_prediction_path = prediction_group[max_mean_index]
        pred_labels = np.load(chosen_prediction_path)
        pred_labels = pred_labels[~zero_depth_mask]

        # Print metrics results
        multi_value_res = metrics.multi_value(pcd_points, pred_labels, gt_labels)
        for index, key in enumerate(multi_value_keys):
            metrics_final_res[frame_index, index] = multi_value_res[key]

        for index, metric in enumerate(all_plane_metrics):
            metric_res = mean(pcd_points, pred_labels, gt_labels, metric)
            metrics_final_res[frame_index, index + 6] = metric_res

        pd.DataFrame(metrics_final_res, columns=column_names, index=None).to_csv("results.csv", index=False)
        print("Metrics calculated for frame: {}".format(os.path.split(cloud_frame_path)[-1][:-4]))


loaders = ['tum', 'icl_tum', 'icl']


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'predictions_path',
        type=str,
        help='Path to dataset'
    )
    parser.add_argument(
        'annotations_path',
        type=str,
        help='Path to annotations folder'
    )
    parser.add_argument(
        '--loader',
        type=str,
        required=True,
        choices=loaders,
        help='Name of loader for dataset'
    )
    parser.add_argument(
        '--step',
        type=int,
        default=50,
        help='Step for annotations'
    )

    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    measure_algo(args.predictions_path, args.annotations_path, args.loader, args.step)
