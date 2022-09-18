import os

import evops
import numpy as np
import pandas as pd
from evops.metrics import multi_value, mean

all_plane_metrics = [
    evops.metrics.iou,
    evops.metrics.dice
]

classic_metrics = [
    evops.metrics.precision,
    evops.metrics.recall,
    evops.metrics.fScore
]


if __name__ == "__main__":
    predicted = np.load("C:\\Users\\dimaj\\Desktop\\instance_preds\\preds-instance.npy")
    scene_names = np.load("C:\\Users\\dimaj\\Desktop\\instance_preds\\scene_nums.npy")[:predicted.shape[0]]
    scene_path = "C:\\kittii\\00\\velodyne"
    annot_path = "C:\\Users\\dimaj\\Documents\\Github\\PlaneDetector\\scripts\\kitti\\annotated_frames"

    multi_value_keys = ["precision", "recall", "under_segmented", "over_segmented", "missed", "noise"]
    mean_metric_names = ["mean_{}".format(metric.__name__) for metric in all_plane_metrics]
    classic_metric_names = ["{}".format(metric.__name__) for metric in classic_metrics]
    multi_value_metric_names = ["multi_{}".format(name) for name in multi_value_keys]
    column_names = multi_value_metric_names + mean_metric_names + classic_metric_names

    metrics_final_res = np.zeros((predicted.shape[0], len(all_plane_metrics) + 6 + len(classic_metrics)), dtype=float)

    for scene_index, scene_name in enumerate(scene_names):
        scene_bin_filename = "{:06d}.bin".format(scene_name)
        scene_annot_filename = "label-{}.npy".format(scene_bin_filename[:-4])
        scene_filepath = os.path.join(scene_path, scene_bin_filename)
        annot_filepath = os.path.join(annot_path, scene_annot_filename)

        scene_points = np.fromfile(scene_filepath, dtype=np.float32).reshape(-1, 4)[:105000, :3]
        gt_labels = np.load(annot_filepath)[:105000]
        pred = predicted[scene_index]

        multi_value_res = multi_value(pred, gt_labels)
        for index, key in enumerate(multi_value_keys):
            metrics_final_res[scene_index, index] = multi_value_res[key]

        for index, metric in enumerate(all_plane_metrics):
            metric_res = mean(pred, gt_labels, metric)
            metrics_final_res[scene_index, index + 6] = metric_res

        for index, metric in enumerate(classic_metrics):
            try:
                metric_res = metric(pred, gt_labels, 'iou')
            except ZeroDivisionError:
                metric_res = 0
            except AssertionError:
                metric_res = 0
            metrics_final_res[scene_index, index + 6 + len(all_plane_metrics)] = metric_res

        print("{0}. Metrics calculated for frame: {1}".format(scene_index, scene_name))

    pd.DataFrame(metrics_final_res, columns=column_names, index=None).to_csv("results.csv", index=False)
