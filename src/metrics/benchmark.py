from src.SegmentedPointCloud import SegmentedPointCloud
from src.metrics.metrics import iou


def benchmark_iou(cloud_predicted: SegmentedPointCloud, cloud_gt: SegmentedPointCloud):
    not_predicted = []
    predicted_ghost = []
    predicted = {}

    for predicted_plane in cloud_predicted.planes:
        max_iou = 0
        max_iou_gt_plane = None
        for gt_plane in cloud_gt.planes:
            iou_value = iou(predicted_plane, gt_plane)
            if iou_value > max_iou:
                max_iou_gt_plane = gt_plane
                max_iou = iou_value

        if max_iou < 0.1:
            predicted_ghost.append(predicted_plane)
        else:
            predicted[max_iou_gt_plane] = (predicted_plane, max_iou)

    for gt_plane in cloud_gt.planes:
        if gt_plane not in predicted.keys():
            not_predicted.append(gt_plane)

    print(not_predicted)
    print(predicted_ghost)
    print(predicted)
