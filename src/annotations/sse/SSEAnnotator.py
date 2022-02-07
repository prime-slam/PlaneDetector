import numpy as np

from src.annotations.BaseAnnotator import BaseAnnotator
from src.annotations.sse.SSEAnnotation import SSEAnnotation
from src.model.SegmentedPlane import SegmentedPlane
from src.model.SegmentedPointCloud import SegmentedPointCloud


class SSEAnnotator(BaseAnnotator):
    def __init__(self, path, start_frame_num: int):
        super().__init__(path, start_frame_num)
        self.annotation = SSEAnnotation(path)

    def annotate(
        self, segmented_pcd: SegmentedPointCloud, frame_num: int
    ) -> SegmentedPointCloud:
        planes = self.annotation.get_all_planes()
        pcd_size = np.asarray(segmented_pcd.pcd.points).shape[0]
        all_segmented_indices = []
        segmented_planes = []
        for plane in planes:
            all_segmented_indices.append(plane.indices)
            segmented_planes.append(
                SegmentedPlane(
                    pcd_indices=plane.indices,
                    zero_depth_pcd_indices=np.asarray([], dtype=int),
                    track_id=SegmentedPlane.NO_TRACK,
                )
            )

        return SegmentedPointCloud(
            pcd=segmented_pcd.pcd,
            planes=segmented_planes,
            unsegmented_cloud_indices=np.setdiff1d(
                np.arange(pcd_size), np.concatenate(all_segmented_indices)
            ),
            zero_depth_cloud_indices=segmented_pcd.zero_depth_cloud_indices,
        )
