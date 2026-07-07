#!/usr/bin/env python3
"""Batch-extract the RGB frames the Joint model needs, driven by a Ground3D infos ``.pkl``.

The Joint model reads ``model.decoder.image_dir/<scene>/<frame>.jpg`` for every ``(scene, frame)``
in the dataset. This script reads ``part3d_glamm_infos_{train,val}.pkl``, collects every unique
``(scene, frame)``, and extracts each RGB frame into ``<out>/<scene>/<frame>.jpg`` using
``extract_frame.py`` (ScanNet ``color/`` or ScanNet++ ``rgb.mkv``). See ``docs/DATA_IMAGES.md`` for
how to obtain and lay out the raw datasets.

    python tools/build_frame_images.py \
        --pkl      data/scannet/part3d_glamm_infos_val.pkl \
        --raw_root $GROUND3D_DATA \
        --out      data/scannet/frame_images
"""
import os
import sys
import argparse
import pickle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract_frame as ef  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pkl', required=True, help='part3d_glamm_infos_{train,val}.pkl')
    ap.add_argument('--raw_root', required=True,
                    help='root holding scannet/scans and/or scannetpp/data (see docs/DATA_IMAGES.md)')
    ap.add_argument('--out', required=True, help='frame_images output dir (= model.decoder.image_dir)')
    a = ap.parse_args()

    d = pickle.load(open(a.pkl, 'rb'))
    data_list = d['data_list'] if isinstance(d, dict) and 'data_list' in d else d

    seen, ok, miss = set(), 0, 0
    for s in data_list:
        name = os.path.basename(s['lidar_points']['lidar_path']).rsplit('.', 1)[0]  # <scene>_frame_<n>
        if '_frame_' not in name:
            continue
        scene, frame = name.split('_frame_')
        if (scene, frame) in seen:
            continue
        seen.add((scene, frame))
        out = os.path.join(a.out, scene, f'{frame}.jpg')
        if os.path.exists(out):
            ok += 1
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        try:
            dataset = ef.detect_dataset(scene)
            (ef.extract_scannet if dataset == 'scannet' else ef.extract_scannetpp)(
                a.raw_root, scene, int(frame), out)
            ok += 1
        except SystemExit as e:        # extract_frame.py exits on a missing raw file
            print('MISS', scene, frame, '->', e)
            miss += 1

    print(f'\nDone: {ok} frame(s) under {a.out}; {miss} missing '
          f'({len(seen)} unique (scene,frame) pairs).')


if __name__ == '__main__':
    main()
