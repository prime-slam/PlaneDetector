import argparse

from src.parser import add_dataset_args


def create_input_parser():
    parser = argparse.ArgumentParser()
    add_dataset_args(parser)
    add_annotations_args(parser)
    add_output_args(parser)

    return parser


def add_output_args(parser):
    parser.add_argument(
        '--output_path',
        type=str,
        help='Path to output folder'
    )


def add_annotations_args(parser):
    parser.add_argument(
        '--annotations_path',
        type=str,
        help='Path to annotations folders with files in "CVAT for video" format'
    )
    parser.add_argument(
        '--disable_annotation_filter_outliers',
        action='store_false',
        dest='filter_annotation_outliers',
        help='Specify if you want to disable auto remove of outliers from annotated planes with RANSAC'
    )

    return parser
