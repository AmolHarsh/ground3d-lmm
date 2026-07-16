# Copyright (c) OpenMMLab. All rights reserved.
"""Build the ScanRefer ``infos`` pkls (train / val).

Reads the pre-processed ScanNet data (``scannet_instance_data/``), the
per-scene ScanRefer descriptions and the per-scene ``<scene>_reason.pth``
tensors, then writes ``<info_prefix>_infos_{train,val}.pkl`` in the OpenMMLab
v2 format consumed by the configs.
"""
import argparse
import sys
from os import path as osp

import mmengine

from update_infos_to_v2 import update_pkl_infos


def scanrefer_data_prep(root_path, qa_data_dir, reason_pth_dir, info_prefix,
                        out_dir, workers):
    """Generate ScanRefer info files for the train and val splits.

    Args:
        root_path (str): Root of the pre-processed ScanNet data.
        qa_data_dir (str): Directory holding the ScanRefer by-scene JSONs.
        reason_pth_dir (str): Directory holding ``<split>/<scene>_reason.pth``.
        info_prefix (str): Prefix of the output pkl filenames.
        out_dir (str): Output directory for the generated pkl files.
        workers (int): Number of threads to be used.
    """
    # The converter lives alongside this script in tools/.
    sys.path.insert(0, osp.dirname(__file__))
    from part3d_glamm_data_utils import ScanReferData

    save_path = out_dir

    for split in ('train', 'val'):
        filename = osp.join(save_path, f'{info_prefix}_infos_{split}.pkl')
        dataset = ScanReferData(
            root_path=root_path,
            qa_data_dir=qa_data_dir,
            reason_pth_dir=reason_pth_dir,
            split=split,
            save_path=save_path)
        infos = dataset.get_infos(num_workers=workers, has_label=True)
        mmengine.dump(infos, filename, 'pkl')
        print(f'{info_prefix} info {split} file is saved to {filename}')
        # 'reason3d' selects the whole-scene update routine that keeps
        # per-point masks + qa_data while writing v2-format instances.
        update_pkl_infos('reason3d', out_dir=out_dir, pkl_path=filename)


def parse_args():
    parser = argparse.ArgumentParser(description='ScanRefer data converter')
    parser.add_argument(
        '--root-path',
        type=str,
        default='data/scannet',
        help='root of the pre-processed ScanNet data '
             '(contains scannet_instance_data/)')
    parser.add_argument(
        '--qa-data-dir',
        type=str,
        required=True,
        help='directory holding ScanRefer_filtered_{split}_by_scene.json')
    parser.add_argument(
        '--reason-pth-dir',
        type=str,
        required=True,
        help='directory holding <split>/<scene>_reason.pth tensors')
    parser.add_argument(
        '--out-dir',
        type=str,
        required=True,
        help='output directory for the generated pkl files')
    parser.add_argument(
        '--info-prefix',
        type=str,
        default='scanrefer',
        help='prefix of the output pkl filenames')
    parser.add_argument(
        '--workers', type=int, default=4, help='number of threads to be used')
    return parser.parse_args()


if __name__ == '__main__':
    from mmdet3d.utils import register_all_modules
    register_all_modules()

    args = parse_args()
    scanrefer_data_prep(
        root_path=args.root_path,
        qa_data_dir=args.qa_data_dir,
        reason_pth_dir=args.reason_pth_dir,
        info_prefix=args.info_prefix,
        out_dir=args.out_dir,
        workers=args.workers)
