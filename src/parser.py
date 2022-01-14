import argparse

from src.annotations.cvat.CVATAnnotator import CVATAnnotator
from src.annotations.sse.SSEAnnotator import SSEAnnotator
from src.detectors import O3DRansacDetector, DDPFFDetector
from src.loaders.depth_image.TumLoader import TumLoader
from src.loaders.depth_image.TumIclLoader import TumIclLoader
from src.loaders.pcd.O3DLoader import O3DLoader
from src.metrics.multi_value.MultiValueBenchmark import MultiValueBenchmark
from src.metrics.one_value.DiceBenchmark import DiceBenchmark
from src.metrics.one_value.IoUBenchmark import IoUBenchmark

loaders = {
    'tum': TumLoader,
    'icl_tum': TumIclLoader,
    'o3d': O3DLoader
}

annotators = {
    'cvat': CVATAnnotator,
    'sse': SSEAnnotator
}

algos = {
    'RANSAC-o3d': O3DRansacDetector,
    'DDPFF': DDPFFDetector
}

metrics = {
    'iou': IoUBenchmark,
    'dice': DiceBenchmark,
    'classic': MultiValueBenchmark
}


def add_dataset_args(parser):
    parser.add_argument(
        'dataset_path',
        type=str,
        help='Path to dataset'
    )
    parser.add_argument(
        '--loader',
        type=str,
        required=True,
        choices=loaders.keys(),
        help='Name of loader for dataset'
    )
    parser.add_argument(
        '--frame_num',
        type=int,
        required=True,
        default=0,
        help='Depth frame number in dataset to start from'
    )

    return parser


def add_annotations_args(parser):
    parser.add_argument(
        '--annotator',
        type=str,
        choices=annotators.keys(),
        help='Name of loader for annotations'
    )
    parser.add_argument(
        '--annotations_path',
        type=str,
        help='Path to annotations.xml file in "CVAT for video" format'
    )
    parser.add_argument(
        '--annotations_start_frame',
        type=int,
        default=0,
        help='Depth frame number in dataset from which annotations starts in the selected file'
    )
    parser.add_argument(
        '--disable_annotation_filter_outliers',
        action='store_false',
        dest='filter_annotation_outliers',
        help='Specify if you want to disable auto remove of outliers from annotated planes with RANSAC'
    )

    return parser


def add_algo_args(parser):
    parser.add_argument(
        '--algo',
        type=str,
        choices=algos.keys(),
        help='Name of the algorithm for benchmarking'
    )
    parser.add_argument(
        '--metric',
        type=str,
        action='append',
        choices=metrics.keys(),
        default='iou',
        help='Name of metrics for algorithm benchmarking'
    )

    return parser


def create_input_parser():
    parser = argparse.ArgumentParser()
    parser = add_dataset_args(parser)
    parser = add_annotations_args(parser)
    parser = add_algo_args(parser)

    return parser
