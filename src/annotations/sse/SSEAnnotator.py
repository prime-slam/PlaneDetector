import numpy as np

from src.annotations.BaseAnnotator import BaseAnnotator
from src.annotations.sse.SSEAnnotation import SSEAnnotation
from src.model.SegmentedPlane import SegmentedPlane
from src.model.SegmentedPointCloud import SegmentedPointCloud


class SSEAnnotator(BaseAnnotator):
    def __init__(self, path, start_frame_num: int):
        super().__init__(path, start_frame_num)
        self.annotation = SSEAnnotation(path)

    def annotate(self, pcd: SegmentedPointCloud, frame_num: int) -> SegmentedPointCloud:
        planes = self.annotation.get_all_planes()
        all_segmented_indices = []
        segmented_planes = []
        for plane in planes:
            all_segmented_indices.append(plane.indices)
            segmented_planes.append(SegmentedPlane(
                pcd_indices=plane.indices,
                track_id=SegmentedPlane.NO_TRACK
            ))

        return SegmentedPointCloud(
            pcd=pcd.pcd,
            planes=segmented_planes,
            unsegmented_cloud_indices=np.concatenate(all_segmented_indices)
        )
