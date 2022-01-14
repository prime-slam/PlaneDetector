from src import OutlierDetector
from src.annotations.BaseAnnotator import BaseAnnotator
from src.annotations.cvat import CVATAnnotator
from src.loaders.BaseLoader import BaseLoader
from src.loaders.depth_image.ImageLoader import ImageLoader
from src.model.SegmentedPointCloud import SegmentedPointCloud
from src.parser import algos, metrics


def load_annotations(
        loader: BaseLoader,
        input_pcd: SegmentedPointCloud,
        depth_frame_num,
        annotator: CVATAnnotator,
        filter_outliers
):
    if isinstance(annotator, ImageLoader):
        annotation_frame_num = loader.depth_to_rgb_index[depth_frame_num]
    else:
        annotation_frame_num = depth_frame_num

    result_pcd = annotator.annotate(
        input_pcd,
        annotation_frame_num
    )
    if filter_outliers:
        result_pcd = OutlierDetector.remove_planes_outliers(result_pcd)

    return result_pcd


def process_frame(
        loader: BaseLoader,
        frame_num: int,
        annotator: BaseAnnotator,
        filter_annotation_outliers,
        algo,
        metric
):
    result_pcd = None
    detected_pcd = None

    input_pcd = loader.read_pcd(frame_num)

    if annotator is not None:
        result_pcd = load_annotations(
            loader,
            input_pcd,
            frame_num,
            annotator,
            filter_annotation_outliers
        )
        # PointCloudPrinter(result_pcd.get_color_pcd_for_visualization()).save_to_ply("result.ply")
        # PointCloudPrinter(result_pcd.get_color_pcd_for_visualization()).save_to_pcd("result.pcd")
    else:
        result_pcd = input_pcd

    if algo is not None:
        detector = algos[algo]
        detected_pcd = detector.detect_planes(input_pcd)

    if annotator is not None and algo is not None and len(metric) > 0:
        for metric_name in metric:
            benchmark = metrics[metric_name]()
            benchmark_result = benchmark.execute(detected_pcd, result_pcd)
            print(benchmark_result)

    return result_pcd, detected_pcd
