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

        # HF ships split lists as <root>/part_ground3d_{split}.txt. The split is the
        # authority on which scenes belong to a run: processed_data/ can legitimately
        # hold point clouds for scenes that carry no Q/A, so falling back to "whatever
        # is on disk" would silently build a different split than the released one.
        split_file = osp.join(root_path, f'part_ground3d_{split}.txt')
        if not os.path.exists(split_file):
            raise FileNotFoundError(
                f'Split file not found: {split_file}\n'
                f'It ships with the dataset — re-download the {split} split from '
                f'https://huggingface.co/datasets/amolharsh/Ground3D_Dataset '
                f'(or point --root-path at the directory that contains it).')
        scene_ids = set(mmengine.list_from_file(split_file))
        self.sample_id_list = self._scan_frame_ids(scene_ids)
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


# ----------------------------------------------------------------------------
# Whole-scene (ScanNet) grounding/reasoning datasets.
#
# Unlike ``Part3DGlammData`` (single-view partial frames), the two classes
# below operate on entire ScanNet scenes. They reuse the pre-processed
# ScanNet point clouds / superpoints / instance labels / aligned boxes (the
# ``scannet_instance_data/`` produced by the standard ScanNet converter) and
# attach per-scene question/answer annotations.
# ----------------------------------------------------------------------------

# ScanNet-18 detection classes (shared by ScanRefer / Reason3D).
SCANNET_GROUNDING_CLASSES = [
    'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
    'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator',
    'showercurtrain', 'toilet', 'sink', 'bathtub', 'garbagebin', 'unknown',
]


class _ScanNetSceneGroundingData(object):
    """Shared logic for whole-scene ScanNet grounding datasets.

    Builds per-scene infos from pre-processed ScanNet data plus per-scene QA
    annotations. The points / superpoints / instance masks / aligned boxes are
    read from ``<root_path>/scannet_instance_data/``; the per-point semantic
    label (ScanNet-200) and the instance label used downstream come from the
    matching ``<scene>_reason.pth`` tensor.

    Args:
        root_path (str): Root of the pre-processed ScanNet data (contains
            ``scannet_instance_data/``).
        qa_data_dir (str): Directory holding the per-scene QA JSON.
        reason_pth_dir (str): Directory holding ``<split>/<scene>_reason.pth``.
        split (str): One of ``train`` / ``val`` / ``test``.
        save_path (str, optional): Where to write ``points/`` etc. and the pkl.
            Defaults to ``root_path``.
    """

    # ``<split>`` is substituted to locate the QA JSON; subclasses set this.
    qa_json_template = None

    def __init__(self, root_path, qa_data_dir, reason_pth_dir,
                 split='train', save_path=None):
        assert split in ('train', 'val', 'test')
        assert self.qa_json_template is not None, \
            'subclasses must set qa_json_template'
        self.root_dir = root_path
        self.save_path = root_path if save_path is None else save_path
        self.qa_data_dir = qa_data_dir
        self.reason_pth_dir = reason_pth_dir
        self.split = split
        self.test_mode = (split == 'test')

        self.classes = list(SCANNET_GROUNDING_CLASSES)
        self.cat2label = {cat: i for i, cat in enumerate(self.classes)}
        self.label2cat = {i: cat for cat, i in self.cat2label.items()}

        qa_file = osp.join(self.qa_data_dir,
                           self.qa_json_template.format(split=split))
        self.qa_files = json.load(open(qa_file, 'r'))
        sample_id_list = list(self.qa_files.keys())
        self.sample_id_list = [s for s in sample_id_list
                               if s.startswith('scene')]

    def __len__(self):
        return len(self.sample_id_list)

    # ---- raw-data accessors ------------------------------------------------
    def get_aligned_bbox_label(self, idx):
        box_file = osp.join(self.root_dir, 'scannet_instance_data',
                            f'{idx}_aligned_bbox.npy')
        mmengine.check_file_exist(box_file)
        return np.load(box_file)

    def get_axis_align_matrix(self, idx):
        matrix_file = osp.join(self.root_dir, 'scannet_instance_data',
                               f'{idx}_axis_align_matrix.npy')
        mmengine.check_file_exist(matrix_file)
        return np.load(matrix_file)

    def _read_reason_pth(self, idx):
        """Load ``(coords, colors, superpoint, semantic_gt200,
        instance_labels)`` from ``<reason_pth_dir>/<split>/<scene>_reason.pth``."""
        import torch
        pth_path = osp.join(self.reason_pth_dir, self.split,
                            f'{idx}_reason.pth')
        mmengine.check_file_exist(pth_path)
        return torch.load(pth_path, map_location='cpu')

    # ---- subclass hooks ----------------------------------------------------
    def _instance_mask(self, idx, reason_data):
        """Return the per-point instance label to persist for this scene."""
        raise NotImplementedError

    def get_infos(self, num_workers=4, has_label=True, sample_id_list=None):
        """Build per-scene infos consumed by ``update_pkl_infos``."""

        def process_single_scene(sample_idx):
            import torch
            info = dict()
            info['point_cloud'] = {'num_features': 6, 'lidar_idx': sample_idx}

            pts_filename = osp.join(self.root_dir, 'scannet_instance_data',
                                    f'{sample_idx}_vert.npy')
            points = np.load(pts_filename)
            mmengine.mkdir_or_exist(osp.join(self.save_path, 'points'))
            points.tofile(osp.join(self.save_path, 'points',
                                   f'{sample_idx}.bin'))
            info['pts_path'] = osp.join('points', f'{sample_idx}.bin')

            sp_filename = osp.join(self.root_dir, 'scannet_instance_data',
                                   f'{sample_idx}_sp_label.npy')
            super_points = np.load(sp_filename)
            mmengine.mkdir_or_exist(osp.join(self.save_path, 'super_points'))
            super_points.tofile(osp.join(self.save_path, 'super_points',
                                         f'{sample_idx}.bin'))
            info['super_pts_path'] = osp.join('super_points',
                                              f'{sample_idx}.bin')

            if not self.test_mode:
                reason_data = self._read_reason_pth(sample_idx)

                pts_instance_mask = self._instance_mask(
                    sample_idx, reason_data).astype(np.int64)
                mmengine.mkdir_or_exist(
                    osp.join(self.save_path, 'instance_mask'))
                pts_instance_mask.tofile(
                    osp.join(self.save_path, 'instance_mask',
                             f'{sample_idx}.bin'))
                info['pts_instance_mask_path'] = osp.join(
                    'instance_mask', f'{sample_idx}.bin')

                # _reason.pth fields:
                #   (coords, colors, superpoint, semantic_gt200, instance_labels)
                semantic_gt200 = reason_data[3]
                if isinstance(semantic_gt200, torch.Tensor):
                    semantic_gt200 = semantic_gt200.numpy()
                pts_semantic_mask = semantic_gt200.astype(np.int64)
                mmengine.mkdir_or_exist(
                    osp.join(self.save_path, 'semantic_mask'))
                pts_semantic_mask.tofile(
                    osp.join(self.save_path, 'semantic_mask',
                             f'{sample_idx}.bin'))
                info['pts_semantic_mask_path'] = osp.join(
                    'semantic_mask', f'{sample_idx}.bin')

                info['qa_data'] = self.qa_files[sample_idx]

            if has_label:
                annotations = {}
                aligned_box_label = self.get_aligned_bbox_label(sample_idx)
                annotations['gt_num'] = aligned_box_label.shape[0]
                if annotations['gt_num'] != 0:
                    aligned_box = aligned_box_label[:, :-1]  # k, 6
                    classes = aligned_box_label[:, -1]  # k
                    annotations['name'] = np.array([
                        self.label2cat.get(
                            int(classes[i]) % len(self.classes), 'unknown')
                        for i in range(annotations['gt_num'])
                    ])
                    annotations['location'] = aligned_box[:, :3]
                    annotations['dimensions'] = aligned_box[:, 3:6]
                    annotations['gt_boxes_upright_depth'] = aligned_box
                    annotations['index'] = np.arange(
                        annotations['gt_num'], dtype=np.int32)
                    annotations['class'] = np.array([
                        self.cat2label.get(
                            self.label2cat.get(
                                int(classes[i]) % len(self.classes),
                                'unknown'), 0)
                        for i in range(annotations['gt_num'])
                    ])
                annotations['axis_align_matrix'] = \
                    self.get_axis_align_matrix(sample_idx)  # 4x4
                info['annos'] = annotations
            return info

        ids = sample_id_list if sample_id_list is not None \
            else self.sample_id_list
        infos = []
        for sample_idx in tqdm(ids, desc=f'{self.split} processing'):
            infos.append(process_single_scene(sample_idx))
        return infos


class ScanReferData(_ScanNetSceneGroundingData):
    """Whole-scene ScanRefer grounding dataset.

    Reads ``ScanRefer_filtered_{split}_by_scene.json`` for per-scene
    descriptions and uses the ``instance_labels`` field of ``<scene>_reason.pth``
    as the per-point instance mask.
    """

    qa_json_template = 'ScanRefer_filtered_{split}_by_scene.json'

    def _instance_mask(self, idx, reason_data):
        import torch
        instance_labels = reason_data[4]
        if isinstance(instance_labels, torch.Tensor):
            instance_labels = instance_labels.numpy()
        return np.asarray(instance_labels)


class Reason3dData(_ScanNetSceneGroundingData):
    """Whole-scene Reason3D grounding/reasoning dataset.

    Reads ``reason3d_scannet_{split}_by_scene.json`` for per-scene reasoning
    questions and uses the pre-processed ScanNet instance label
    (``<scene>_ins_label.npy``) as the per-point instance mask.
    """

    qa_json_template = 'reason3d_scannet_{split}_by_scene.json'

    def _instance_mask(self, idx, reason_data):
        ins_file = osp.join(self.root_dir, 'scannet_instance_data',
                            f'{idx}_ins_label.npy')
        mmengine.check_file_exist(ins_file)
        return np.load(ins_file)
