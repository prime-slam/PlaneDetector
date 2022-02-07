from src.model.SegmentedPlane import SegmentedPlane
from src.model.SegmentedPointCloud import SegmentedPointCloud
from src.metrics.one_value.IoUBenchmark import iou


def associate_segmented_point_clouds(
    pcd_a: SegmentedPointCloud, pcd_b: SegmentedPointCloud
) -> dict:
    """
    Associates point clouds and assign track ids from point cloud A to point cloud B
    :param pcd_a: Base cloud for association
    :param pcd_b: Associated cloud
    :return dictionary with matches from old track_id of pcd_b to new assigned track_ids
    """

    matches = {}
    for plane_b in pcd_b.planes:
        max_iou = 0
        selected_track_id = SegmentedPlane.NO_TRACK
        for plane_a in pcd_a.planes:
            iou_value = iou(plane_a, plane_b)
            if iou_value > max_iou:
                max_iou = iou_value
                selected_track_id = plane_a.track_id

        matches[plane_b.track_id] = selected_track_id
        plane_b.track_id = selected_track_id

    return matches
