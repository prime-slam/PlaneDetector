from src import OutlierDetector
from src.detectors import AnnotationsDetector
from src.loaders.depth_image.ImageLoader import ImageLoader
from src.model.CVATAnnotation import CVATAnnotation
from src.parser import algos, metrics
from src.utils.point_cloud import depth_to_pcd


def load_annotations(
        loader: ImageLoader,
        depth_frame_num,
        annotation,
        cam_intrinsic,
        filter_outliers
):
    frame_number = loader.depth_to_rgb_index[depth_frame_num]
    pcd = loader.read_pcd(depth_frame_num)
    result_pcd = AnnotationsDetector.segment_pcd_by_annotations(
        pcd,
        cam_intrinsic,
        annotation,
        frame_number
    )
    if filter_outliers:
        result_pcd = OutlierDetector.remove_planes_outliers(result_pcd)

    return result_pcd


def process_frame(loader, depth_frame_num: int, annotation: CVATAnnotation, filter_annotation_outliers, algo, metric):
    depth_image = loader.read_depth_image(depth_frame_num)
    result_pcd = None
    detected_pcd = None
    image_shape = depth_image.shape
    cam_intrinsic = loader.config.get_cam_intrinsic(image_shape)
    initial_pcd_transform = loader.config.get_initial_pcd_transform()

    if annotation is not None:
        result_pcd = load_annotations(
            loader,
            depth_frame_num,
            annotation,
            cam_intrinsic,
            filter_annotation_outliers
        )
        # PointCloudPrinter(result_pcd.get_color_pcd_for_visualization()).save_to_ply("result.ply")
        # PointCloudPrinter(result_pcd.get_color_pcd_for_visualization()).save_to_pcd("result.pcd")

    if algo is not None:
        pcd = depth_to_pcd(depth_image, cam_intrinsic, initial_pcd_transform)
        detector = algos[algo]
        detected_pcd = detector.detect_planes(pcd)

    if annotation is not None and algo is not None and len(metric) > 0:
        for metric_name in metric:
            benchmark = metrics[metric_name]()
            benchmark_result = benchmark.execute(detected_pcd, result_pcd)
            print(benchmark_result)

    return result_pcd, detected_pcd