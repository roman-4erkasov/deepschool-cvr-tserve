import os
import argparse
from argparse import Namespace
from os import path as osp

from model_archiver.model_packaging import package_model
from model_archiver.model_packaging_utils import ModelExportUtils


DIR_PATH = os.path.dirname(__file__)


def generate_mar(torch_checkpoint_path: str, mar_checkpoint_dir: str, model_name: str, handler_name: str):
    os.makedirs(mar_checkpoint_dir, exist_ok=True)
    handler_path = osp.join(DIR_PATH, handler_name)
    args = Namespace(**{
        'model_file': None,
        'model_name': model_name,
        'version': '1.0',
        'serialized_file': torch_checkpoint_path,
        'handler': handler_path,
        'export_path': mar_checkpoint_dir,
        'force': False,
        'extra_files': osp.join(DIR_PATH, "vocabulary.json"),
        'requirements_file': osp.join(DIR_PATH, 'requirements.txt'),
        'runtime': 'python',
        'archive_format': 'default'
    })
    manifest = ModelExportUtils.generate_manifest_json(args)
    package_model(args, manifest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Conversion from torch models to mar-archives.'
    )
    parser.add_argument('--src', help='path to source pytorch file')
    parser.add_argument('--dst', help='destination directory to save mar-files')
    parser.add_argument('--model_name', help='path to mar-file')
    parser.add_argument('--handler_name', help='path to mar-file')
    args = parser.parse_args()
    generate_mar(args.src, args.dst, args.model_name, args.handler_name)
