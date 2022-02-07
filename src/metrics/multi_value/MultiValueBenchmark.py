from src.metrics.BaseBenchmark import BaseBenchmark
from src.model.SegmentedPointCloud import SegmentedPointCloud
from src.utils.metrics import are_nearly_overlapped


class MultiValueBenchmarkResult:
    def __init__(
        self,
        precision: float,
        recall: float,
        under_segmented: float,
        over_segmented: float,
        missed: float,
        noise: float,
    ):
        self.plane_precision = precision
        self.plane_recall = recall
        self.over_segmented_rate = over_segmented
        self.under_segmented_rate = under_segmented
        self.missed_rate = missed
        self.noise_rate = noise

    def __str__(self):
        return (
            f"Results of 'multi value' metric\n"
            f"Precision: {self.plane_precision}\n"
            f"Recall: {self.plane_recall}\n"
            f"Over segmentation rate: {self.over_segmented_rate}\n"
            f"Under segmentation rate: {self.under_segmented_rate}\n"
            f"Missed rate: {self.missed_rate}\n"
            f"Noise rate: {self.noise_rate}"
        )


class MultiValueBenchmark(BaseBenchmark):
    def __init__(self, overlap_threshold=0.8):
        self.overlap_threshold = overlap_threshold

    def execute(
        self, cloud_predicted: SegmentedPointCloud, cloud_gt: SegmentedPointCloud
    ):
        correctly_segmented_amount = 0
        predicted_amount = len(cloud_predicted.planes)
        gt_amount = len(cloud_gt.planes)
        under_segmented_amount = 0
        noise_amount = 0

        overlapped_predicted_by_gt = {plane: [] for plane in cloud_gt.planes}

        for predicted_plane in cloud_predicted.planes:
            overlapped_gt_planes = []
            for gt_plane in cloud_gt.planes:
                are_well_overlapped = are_nearly_overlapped(
                    predicted_plane, gt_plane, self.overlap_threshold
                )
                if are_well_overlapped:
                    overlapped_gt_planes.append(gt_plane)
                    overlapped_predicted_by_gt[gt_plane].append(predicted_plane)

            if len(overlapped_gt_planes) > 0:
                correctly_segmented_amount += 1
            else:
                noise_amount += 1

            if len(overlapped_gt_planes) > 1:
                under_segmented_amount += 1

        over_segmented_amount = 0
        missed_amount = 0
        for overlapped in overlapped_predicted_by_gt.values():
            if len(overlapped) > 1:
                over_segmented_amount += 1
            elif len(overlapped) == 0:
                missed_amount += 1

        return MultiValueBenchmarkResult(
            precision=correctly_segmented_amount / predicted_amount,
            recall=correctly_segmented_amount / gt_amount,
            under_segmented=under_segmented_amount / predicted_amount,
            over_segmented=over_segmented_amount / gt_amount,
            missed=missed_amount / gt_amount,
            noise=noise_amount / predicted_amount,
        )
