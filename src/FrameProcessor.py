from src import OutlierDetector
from src.detectors import AnnotationsDetector
from src.loaders.BaseLoader import BaseLoader
from src.loaders.depth_image.ImageLoader import ImageLoader
from src.annotations.CVATAnnotation import CVATAnnotation
from src.model.SegmentedPointCloud import SegmentedPointCloud
from src.parser import algos, metrics


def load_annotations_depth_image(
        loader: ImageLoader,
        input_pcd: SegmentedPointCloud,
        depth_frame_num,
        annotation,
        filter_outliers
):
    rgb_frame_num = loader.depth_to_rgb_index[depth_frame_num]
    result_pcd = AnnotationsDetector.segment_pcd_by_annotations(
        input_pcd,
        loader.config.get_cam_intrinsic(),
        annotation,
        rgb_frame_num
    )
    if filter_outliers:
        result_pcd = OutlierDetector.remove_planes_outliers(result_pcd)

    return result_pcd


def process_frame(
        loader: BaseLoader,
        frame_num: int,
        annotation: CVATAnnotation,
        filter_annotation_outliers,
        algo,
        metric
):
    result_pcd = None
    detected_pcd = None

    input_pcd = loader.read_pcd(frame_num)

    if annotation is not None and isinstance(loader, ImageLoader):
        result_pcd = load_annotations_depth_image(
            loader,
            input_pcd,
            frame_num,
            annotation,
            filter_annotation_outliers
        )
        # PointCloudPrinter(result_pcd.get_color_pcd_for_visualization()).save_to_ply("result.ply")
        # PointCloudPrinter(result_pcd.get_color_pcd_for_visualization()).save_to_pcd("result.pcd")

    if algo is not None:
        detector = algos[algo]
        detected_pcd = detector.detect_planes(input_pcd.pcd)

    if annotation is not None and algo is not None and len(metric) > 0:
        for metric_name in metric:
            benchmark = metrics[metric_name]()
            benchmark_result = benchmark.execute(detected_pcd, result_pcd)
            print(benchmark_result)

    return result_pcd, detected_pcd
