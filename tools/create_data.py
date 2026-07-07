# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import sys
from os import path as osp

# Add data directory to path for imports
sys.path.insert(0, osp.join(osp.dirname(__file__), '..', 'data'))

from indoor_converter import create_indoor_info_file
from update_infos_to_v2 import update_pkl_infos


def scannet_data_prep(root_path, info_prefix, out_dir, workers):
    """Prepare the info file for scannet dataset.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
    """
    create_indoor_info_file(
        root_path, info_prefix, out_dir, workers=workers)
    info_train_path = osp.join(out_dir, f'{info_prefix}_oneformer3d_infos_train.pkl')
    info_val_path = osp.join(out_dir, f'{info_prefix}_oneformer3d_infos_val.pkl')
    info_test_path = osp.join(out_dir, f'{info_prefix}_oneformer3d_infos_test.pkl')
    update_pkl_infos(info_prefix, out_dir=out_dir, pkl_path=info_train_path)
    update_pkl_infos(info_prefix, out_dir=out_dir, pkl_path=info_val_path)
    update_pkl_infos(info_prefix, out_dir=out_dir, pkl_path=info_test_path)


def part3d_glamm_data_prep(root_path, info_prefix, out_dir, workers, qa_data_dir):
    """Prepare the info file for part3d_glamm dataset.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
    """
    # The converter lives alongside this script in tools/.
    sys.path.insert(0, osp.dirname(__file__))
    from part3d_glamm_data_utils import Part3DGlammData
    
    save_path = root_path if out_dir is None else out_dir
    
    # Generate infos for train, val, test splits

    train_filename = osp.join(save_path, f'{info_prefix}_infos_train.pkl')
    val_filename = osp.join(save_path, f'{info_prefix}_infos_val.pkl')
    # test_filename = osp.join(save_path, f'{info_prefix}_infos_test.pkl')
    
    train_dataset = Part3DGlammData(root_path=root_path, split='train', save_path=save_path, qa_data_dir=qa_data_dir)
    val_dataset = Part3DGlammData(root_path=root_path, split='val', save_path=save_path, qa_data_dir=qa_data_dir)
    # test_dataset = Part3DGlammData(root_path=root_path, split='test', save_path=save_path)
    
    import mmengine



    infos_train = train_dataset.get_infos(num_workers=workers, has_label=True)
    mmengine.dump(infos_train, train_filename, 'pkl')
    print(f'{info_prefix} info train file is saved to {train_filename}')
    update_pkl_infos(info_prefix, out_dir=out_dir, pkl_path=train_filename)
    
    infos_val = val_dataset.get_infos(num_workers=workers, has_label=True)
    mmengine.dump(infos_val, val_filename, 'pkl')
    print(f'{info_prefix} info val file is saved to {val_filename}')
    update_pkl_infos(info_prefix, out_dir=out_dir, pkl_path=val_filename)

    
    # infos_test = test_dataset.get_infos(num_workers=workers, has_label=False)
    # mmengine.dump(infos_test, test_filename, 'pkl')
    # print(f'{info_prefix} info test file is saved to {test_filename}')
    
    # Update to v2 format
    
    


parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('--dataset', default='part3d_glamm', help='name of the dataset')
parser.add_argument(
    '--root-path',
    type=str,
    default='data/scannet',
    help='specify the root path of dataset')
parser.add_argument(
    '--out-dir',
    type=str,
    default='data/scannet/',
    required=False,
    help='name of info pkl')

parser.add_argument(
    '--qa_data_dir',
    type=str,
    default='data/scannet/refined_qa_data',
    required=False,
    help='name of info pkl')

parser.add_argument('--extra-tag', type=str, default='part3d_glamm')
parser.add_argument(
    '--workers', type=int, default=4, help='number of threads to be used')
args = parser.parse_args()

if __name__ == '__main__':
    from mmdet3d.utils import register_all_modules
    register_all_modules()

    if args.dataset in ('scannet', 'scannet200'):
        scannet_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            out_dir=args.out_dir,
            workers=args.workers)
    elif args.dataset == 'part3d_glamm':
        part3d_glamm_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            out_dir=args.out_dir,
            workers=args.workers,
            qa_data_dir=args.qa_data_dir)
    else:
        raise NotImplementedError(f'Don\'t support {args.dataset} dataset.')
