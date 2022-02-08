import numpy as np

from src.model.SegmentedPointCloud import SegmentedPointCloud


class LabelPrinter:
    def __init__(self, labels: np.array):
        self.labels = labels

    def save_to_int_arr(self, filename):
        assert filename[-4:] == ".npy"
        np.save(filename, self.labels)

    @staticmethod
    def build_from_segmented_pcd(segmented_pcd: SegmentedPointCloud):
        pcd_shape = np.asarray(segmented_pcd.pcd.points).shape
        labels = np.zeros(pcd_shape[:-1])

        for index, plane in enumerate(segmented_pcd.planes):
            labels[plane.pcd_indices] = index + 1  # +1 as we want to prevent 0 label usage for planes

        return LabelPrinter(labels)
