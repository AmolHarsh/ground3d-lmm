#!/usr/bin/env python3
"""Export Ground3D-LMM predicted ``<SEG>`` masks as colored ``.ply`` point clouds.

Two-step, fully offline (no GPU, no Open3D):

1. Run inference with mask dumping enabled. Either set the env var::

       MASK_DUMP_DIR=viz_dumps  bash eval_ground3d_scannet.sh

   or pass ``model.decoder.save_viz_dir=viz_dumps`` via ``--cfg-options``. This writes, per scene,
   ``<scene>_frame_<n>_qa_pred_masks.npy`` (one predicted mask per QA / conversation turn).

2. Turn the dumps into colored ``.ply`` files::

       python tools/export_grounding_ply.py \
           --masks_dir  viz_dumps \
           --points_dir data/scannet/points \
           --out        viz_ply

Coloring (see ``uniseg3d/grounding_viz.py``):
* ``--mode combine`` (default): one ``.ply`` per scene; the i-th grounded region gets the i-th
  palette color (1st = red, 2nd = green, 3rd = blue, ...). This is the "common .ply per
  conversation with two masks in different colors" case.
* ``--mode per_qa``: one ``.ply`` per QA, the single grounded region in red.

``--points_dir`` holds ``<scene>_frame_<n>.bin`` (XYZ+RGB, ``load_dim=6``) — i.e. the ``points/``
folder produced by ``tools/create_data.py``.
"""
import os
import glob
import argparse
import sys

import importlib.util

import numpy as np

# Load the viz writer directly from its file so this tool stays offline (no torch / MinkowskiEngine
# pulled in via the uniseg3d package __init__).
_gv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'uniseg3d', 'grounding_viz.py')
_spec = importlib.util.spec_from_file_location('grounding_viz', _gv_path)
grounding_viz = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(grounding_viz)
write_grounding_ply = grounding_viz.write_grounding_ply


def _load_points(points_dir, scene):
    bin_path = os.path.join(points_dir, scene + '.bin')
    if not os.path.exists(bin_path):
        return None, None
    pc = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 6)
    return pc[:, :3], pc[:, 3:6].clip(0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--masks_dir', required=True,
                    help='dir of <scene>_qa_pred_masks.npy dumps (MASK_DUMP_DIR / save_viz_dir)')
    ap.add_argument('--points_dir', required=True,
                    help='dir of <scene>.bin point clouds (XYZ+RGB) from create_data.py')
    ap.add_argument('--out', required=True, help='output dir for .ply files')
    ap.add_argument('--mode', choices=['combine', 'per_qa'], default='combine',
                    help="combine: one .ply/scene, regions colored red,green,...; "
                         "per_qa: one .ply per QA (red).")
    ap.add_argument('--max_frac', type=float, default=0.85,
                    help='skip a mask covering > this fraction of points (degenerate grounding)')
    ap.add_argument('--group', type=int, default=0,
                    help='in combine mode, emit one .ply per GROUP consecutive masks '
                         '(e.g. 2 -> red/green pairs per conversation). 0 = all in one .ply.')
    a = ap.parse_args()

    dumps = sorted(glob.glob(os.path.join(a.masks_dir, '*_qa_pred_masks.npy')))
    if not dumps:
        sys.exit(f'No *_qa_pred_masks.npy found in {a.masks_dir}. '
                 f'Run eval with MASK_DUMP_DIR set first.')

    n_ply = 0
    for pf in dumps:
        scene = os.path.basename(pf)[:-len('_qa_pred_masks.npy')]
        xyz, rgb = _load_points(a.points_dir, scene)
        if xyz is None:
            print(f'skip {scene}: {scene}.bin not in {a.points_dir}')
            continue
        qa_masks = list(np.load(pf, allow_pickle=True))
        if len(qa_masks) == 0:
            continue

        if a.mode == 'per_qa':
            for qi, m in enumerate(qa_masks):
                out = os.path.join(a.out, scene, f'qa{qi:02d}.ply')
                lab = write_grounding_ply(out, xyz, rgb, [m], max_frac=a.max_frac)
                if lab.max() > 0:
                    n_ply += 1
        else:  # combine
            groups = ([qa_masks] if a.group <= 0
                      else [qa_masks[i:i + a.group] for i in range(0, len(qa_masks), a.group)])
            for gi, g in enumerate(groups):
                name = f'{scene}.ply' if a.group <= 0 else f'{scene}_grp{gi:02d}.ply'
                out = os.path.join(a.out, name)
                lab = write_grounding_ply(out, xyz, rgb, g, max_frac=a.max_frac)
                print(f'{scene}: {int(lab.max())} colored region(s) -> {out}')
                n_ply += 1

    print(f'\nDone. Wrote {n_ply} .ply file(s) to {a.out}')


if __name__ == '__main__':
    main()
