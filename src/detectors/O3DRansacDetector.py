import numpy as np
import open3d as o3d

from src.detectors.BaseDetector import BaseDetector
from src.model.SegmentedPlane import SegmentedPlane
from src.model.SegmentedPointCloud import SegmentedPointCloud


class O3DRansacDetector(BaseDetector):
    def __init__(self, num_planes: int = 5):
        self.num_planes = num_planes

    @staticmethod
    def detect_plane(pcd: o3d.geometry.PointCloud) -> (np.array, np.array):
        _, inliers = pcd.segment_plane(
            distance_threshold=0.01,  # 0.03 -- for TUM pioneer;  0.01 -- for TUM desk; 0.005 -- for ICL
            ransac_n=3,
            num_iterations=1000
        )
        inliers = np.asarray(inliers)
        outliers = np.setdiff1d(
            np.arange(np.asarray(pcd.points).shape[0]),
            inliers
        )
        return inliers, outliers

    def detect_planes(self, segmented_pcd: SegmentedPointCloud) -> SegmentedPointCloud:
        pcd = segmented_pcd.pcd
        outlier_pcd = pcd
        detected_planes = []
        detected_indices = []
        all_indices = np.arange(np.asarray(pcd.points).shape[0])

        for _ in range(self.num_planes):
            inliers, outliers = O3DRansacDetector.detect_plane(outlier_pcd)  # indices in small pcd with some detached planes
            inlier_indices = all_indices[inliers]  # map local indices to global ones
            all_indices = np.setdiff1d(  # it will work only if setdiff1d doesn't change order in all_indices
                all_indices,  # we update all indices and now they contain only outliers indices
                inlier_indices  # but in global index system
            )
            outlier_pcd = outlier_pcd.select_by_index(outliers)
            detected_planes.append(
                SegmentedPlane(
                    inlier_indices,
                    zero_depth_pcd_indices=np.asarray([], dtype=int),
                    track_id=SegmentedPlane.NO_TRACK
                )
            )
            detected_indices.append(inlier_indices)

        outlier_indices = np.setdiff1d(
            np.arange(np.asarray(pcd.points).shape[0]),
            np.concatenate(detected_indices)
        )

        return SegmentedPointCloud(
            pcd,
            detected_planes,
            unsegmented_cloud_indices=outlier_indices,
            zero_depth_cloud_indices=segmented_pcd.zero_depth_cloud_indices,
            structured_shape=segmented_pcd.structured_shape
        )

