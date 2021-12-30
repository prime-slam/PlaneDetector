from src.SegmentedPointCloud import SegmentedPointCloud
from src.metrics.one_value.IoUBenchmark import iou


def associate_segmented_point_clouds(pcd_a: SegmentedPointCloud, pcd_b: SegmentedPointCloud):
    """
    Associates point clouds and assign track ids from point cloud A to point cloud B
    :param pcd_a: Base cloud for association
    :param pcd_b: Associated cloud
    """

    for plane_b in pcd_b.planes:
        max_iou = 0
        for plane_a in pcd_a.planes:
            iou_value = iou(plane_a, plane_b)
            if iou_value > max_iou:
                max_iou = iou_value
                plane_b.track_id = plane_a.track_id

