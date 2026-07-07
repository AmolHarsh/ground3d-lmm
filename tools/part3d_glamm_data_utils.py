# Copyright (c) OpenMMLab. All rights reserved.
"""Part3D-Glamm data converter.

Reads the Ground3D dataset as published on Hugging Face
(``amolharsh/Ground3D_Dataset``) and produces the per-frame ``infos`` consumed
by ``tools/update_infos_to_v2.py`` (which writes the final
``part3d_glamm_infos_{split}.pkl`` used by the configs).

Published HF layout (per source, e.g. ``data/scannet``)::

    processed_data/<scene>/frame_<n>_vert.npy        # XYZ+RGB points  (load_dim=6)
    processed_data/<scene>/frame_<n>_sp_label.npy    # superpoint ids
    processed_data/<scene>/frame_<n>_part_label.npy  # per-point part labels
    processed_data/<scene>/frame_<n>_object_label.npy# per-point object labels
    refined_qa_data/part_qa/<scene>/final_json_outputs/<n>_part_mapping.json
    refined_qa_data/object_qa/<scene>/final_json_outputs/<n>_object_mapping.json
    refined_qa_data/multiturn_qa/<scene>/frame_<n>_multi_conv.json
    part_ground3d_{train,val}.txt                    # scene split lists

A frame is identified by ``<scene>_frame_<n>``. The converter flattens each
frame's point cloud / masks into ``points/``, ``super_points/``, ``part_mask/``,
``object_mask/`` as ``<frame_id>.bin`` and records the QA for that frame.
"""

import os
import copy
import json
from os import path as osp

import mmengine
import numpy as np
from tqdm import tqdm

ALL_TASKS = [
    'functional_part_grounding', 'functional_object_grounding',
    'scale_comparison_size', 'distance_estimation',
    'relative_position_forward_reasoning', 'relative_depth_forward',
    'existence_verification', 'scale_estimation', 'grounded_dimension_reasoning',
]


def _flatten_qa_data(origin_qa_json):
    """Flatten ``{id: {task: [{question, answer}, ...]}}`` into
    ``{'qa_data': {task: [items]}}``; ``<SEG>`` is appended to questions that
    lack it. Returns ``None`` if the file is absent."""
    if not os.path.exists(origin_qa_json):
        return None
    origin = json.load(open(origin_qa_json, 'r'))
    qa_data = {'qa_data': {key: [] for key in ALL_TASKS}}
    for _id in origin:
        if not isinstance(origin[_id], dict):  # skip scene_level_metrics etc.
            continue
        for task, qa_list in origin[_id].items():
            if task not in ALL_TASKS or not isinstance(qa_list, list):
                continue
            for qa_item in qa_list:
                try:
                    item = copy.deepcopy(qa_item)
                    if '<SEG>' not in item['question']:
                        item['question'] = item['question'] + '<SEG>'
                    qa_data['qa_data'][task].append(item)
                except Exception:
                    continue
    qa_data['qa_data'] = {k: v for k, v in qa_data['qa_data'].items() if v}
    return qa_data


def _flatten_multi_turn_qa(origin_qa_json):
    """Flatten multi-turn conversations into a list of turn-lists, or
    ``None`` if absent."""
    if not os.path.exists(origin_qa_json):
        return None
    origin = json.load(open(origin_qa_json, 'r'))
    out = []
    for _id in origin:
        if not isinstance(origin[_id], dict):
            continue
        for qa_list in origin[_id].values():
            if not isinstance(qa_list, list):
                continue
            for turn in qa_list:
                try:
                    if '<SEG>' in turn['answer'] and '<SEG>' not in turn['question']:
                        turn['question'] = turn['question'] + '<SEG>'
                except Exception:
                    continue
            out.append(qa_list)
    return out if out else None


class Part3DGlammData(object):
    """Generate part3d_glamm infos from the published (HF) dataset layout.

    Args:
        root_path (str): Dataset source root, e.g. ``data/scannet``.
        split (str): One of ``train`` / ``val`` / ``test``.
        save_path (str, optional): Where to write ``points/`` etc. and the pkl.
            Defaults to ``root_path``.
        qa_data_dir (str, optional): Root of the QA JSONs, e.g.
            ``data/scannet/refined_qa_data``.
    """

    def __init__(self, root_path, split='train', save_path=None, qa_data_dir=None):
        assert split in ('train', 'val', 'test')
        self.root_dir = root_path
        self.save_path = root_path if save_path is None else save_path
        self.split = split
        self.qa_data_dir = qa_data_dir
        self.test_mode = (split == 'test')

        # HF ships split lists as <root>/part_ground3d_{split}.txt
        split_file = osp.join(root_path, f'part_ground3d_{split}.txt')
        if os.path.exists(split_file):
            scene_ids = set(mmengine.list_from_file(split_file))
            self.sample_id_list = self._scan_frame_ids(scene_ids)
        else:
            print(f'Warning: split file {split_file} not found; using all '
                  f'scenes under processed_data/.')
            self.sample_id_list = self._scan_frame_ids(None)
        print(f'{split}: {len(self.sample_id_list)} frames')

    # ---- HF nested-layout helpers ------------------------------------------
    def _frame_npy(self, frame_id, suffix):
        """``<scene>_frame_<n>`` -> ``processed_data/<scene>/frame_<n>_<suffix>.npy``."""
        scene_id, frame_num = frame_id.split('_frame_')
        return osp.join(self.root_dir, 'processed_data', scene_id,
                        f'frame_{frame_num}_{suffix}.npy')

    def _scan_frame_ids(self, scene_id_set):
        """Walk ``processed_data/<scene>/*_vert.npy`` -> sorted ``<scene>_frame_<n>``."""
        data_dir = osp.join(self.root_dir, 'processed_data')
        if not os.path.exists(data_dir):
            return []
        frame_ids = []
        for scene_id in os.listdir(data_dir):
            if scene_id_set is not None and scene_id not in scene_id_set:
                continue
            scene_dir = osp.join(data_dir, scene_id)
            if not os.path.isdir(scene_dir):
                continue
            for fn in os.listdir(scene_dir):
                if fn.startswith('frame_') and fn.endswith('_vert.npy'):
                    frame_num = fn[len('frame_'):-len('_vert.npy')]
                    frame_ids.append(f'{scene_id}_frame_{frame_num}')
        return sorted(frame_ids)

    def __len__(self):
        return len(self.sample_id_list)

    # ---- QA ----------------------------------------------------------------
    def _process_qa(self, frame_id):
        scene_id, frame_num = frame_id.split('_frame_')
        part = _flatten_qa_data(osp.join(
            self.qa_data_dir, 'part_qa', scene_id, 'final_json_outputs',
            f'{frame_num}_part_mapping.json'))
        if part is not None and len(part['qa_data']) == 0:
            part = None
        obj = _flatten_qa_data(osp.join(
            self.qa_data_dir, 'object_qa', scene_id, 'final_json_outputs',
            f'{frame_num}_object_mapping.json'))
        if obj is not None and len(obj['qa_data']) == 0:
            obj = None
        multi = _flatten_multi_turn_qa(osp.join(
            self.qa_data_dir, 'multiturn_qa', scene_id,
            f'frame_{frame_num}_multi_conv.json'))
        if part is None and obj is None and multi is None:
            return None
        return {'part_qa_data': part, 'object_qa_data': obj,
                'multi_turn_qa_data': multi}

    # ---- per-frame ---------------------------------------------------------
    def _process_frame(self, frame_id):
        qa_data = self._process_qa(frame_id)
        if qa_data is None:
            return None
        info = {'point_cloud': {'num_features': 6, 'lidar_idx': frame_id}}
        info['qa_data'] = qa_data

        pts_file = self._frame_npy(frame_id, 'vert')
        if not os.path.exists(pts_file):
            print(f'Warning: points file not found: {pts_file}')
            return None
        mmengine.mkdir_or_exist(osp.join(self.save_path, 'points'))
        np.load(pts_file).tofile(osp.join(self.save_path, 'points', f'{frame_id}.bin'))
        info['pts_path'] = osp.join('points', f'{frame_id}.bin')

        sp_file = self._frame_npy(frame_id, 'sp_label')
        if os.path.exists(sp_file):
            mmengine.mkdir_or_exist(osp.join(self.save_path, 'super_points'))
            np.load(sp_file).tofile(osp.join(self.save_path, 'super_points', f'{frame_id}.bin'))
            info['super_pts_path'] = osp.join('super_points', f'{frame_id}.bin')

        if not self.test_mode:
            obj_file = self._frame_npy(frame_id, 'object_label')
            if os.path.exists(obj_file):
                mmengine.mkdir_or_exist(osp.join(self.save_path, 'object_mask'))
                np.load(obj_file).astype(np.int64).tofile(
                    osp.join(self.save_path, 'object_mask', f'{frame_id}.bin'))
                info['pts_object_mask_path'] = osp.join('object_mask', f'{frame_id}.bin')
            part_file = self._frame_npy(frame_id, 'part_label')
            if os.path.exists(part_file):
                mmengine.mkdir_or_exist(osp.join(self.save_path, 'part_mask'))
                np.load(part_file).astype(np.int64).tofile(
                    osp.join(self.save_path, 'part_mask', f'{frame_id}.bin'))
                info['pts_part_mask_path'] = osp.join('part_mask', f'{frame_id}.bin')

        # Frame-based partial scenes carry no separate boxes/alignment;
        # record an empty annos so the v2 updater fills axis_align_matrix.
        info['annos'] = {'gt_num': 0, 'axis_align_matrix': np.eye(4, dtype=np.float32)}
        return info

    def get_infos(self, num_workers=4, has_label=True, sample_id_list=None):
        ids = sample_id_list if sample_id_list is not None else self.sample_id_list
        infos = []
        for frame_id in tqdm(ids, desc=f'{self.split} processing'):
            info = self._process_frame(frame_id)
            if info is not None:
                infos.append(info)
        return infos
