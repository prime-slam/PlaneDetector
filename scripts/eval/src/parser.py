import argparse

from src.loaders import TumLoader, TumIclLoader, IclLoader

loaders = {
    'tum': TumLoader,
    'icl_tum': TumIclLoader,
    'icl': IclLoader
}


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dataset_path',
        type=str,
        help='Path to dataset'
    )
    parser.add_argument(
        'annotations_path',
        type=str,
        help='Path to annotations folder'
    )
    parser.add_argument(
        'workdir',
        type=str,
        help='Folder to store input and output directories'
    )

    parser.add_argument(
        '--loader',
        type=str,
        required=True,
        choices=loaders.keys(),
        help='Name of loader for dataset'
    )
    
    return parser