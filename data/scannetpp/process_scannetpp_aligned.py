"""
Preprocessing script for ScanNet++ segmentation with ScanNet-style outputs.

This script keeps ScanNet++ label mapping logic, but writes files in the
same naming/layout style used by batch_load_scannetpp_data.py:
  <scene>_vert.npy
  <scene>_sp_label.npy
  <scene>_sem_label.npy
  <scene>_ins_label.npy
"""

import argparse
import gc
import json
import multiprocessing as mp
from collections import OrderedDict
from pathlib import Path
from time import time

import numpy as np
import open3d as o3d
import pandas as pd
import segmentator
import torch


def filter_map_classes(mapping, mapping_type):
    if mapping_type == "semantic":
        map_key = "semantic_map_to"
    elif mapping_type == "instance":
        map_key = "instance_map_to"
    else:
        raise NotImplementedError

    map_dict = OrderedDict()
    for i in range(mapping.shape[0]):
        row = mapping.iloc[i]
        class_name = row["class"]
        map_target = row[map_key]
        try:
            if len(map_target) > 0:
                if map_target != "None":
                    map_dict[class_name] = map_target
        except TypeError:
            if class_name not in map_dict:
                map_dict[class_name] = class_name
    return map_dict


def parse_scene(
    name,
    split,
    dataset_root,
    output_folder,
    label_mapping,
    class2idx,
    ignore_index=-1,
):
    start = time()
    print(f"[START] {name} ({split})", flush=True)
    try:
        dataset_root = Path(dataset_root)
        output_folder = Path(output_folder)
        scene_path = dataset_root / "data" / name / "scans"
        mesh_path = scene_path / "mesh_aligned_0.05.ply"
        segs_path = scene_path / "segments.json"
        anno_path = scene_path / "segments_anno.json"

        output_folder.mkdir(parents=True, exist_ok=True)
        output_prefix = output_folder / name
        if (output_folder / f"{name}_vert.npy").exists():
            print(f"[SKIP] {name} ({split}) exists", flush=True)
            return (name, split, "skipped", "exists", time() - start)

        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        coord = np.array(mesh.vertices).astype(np.float32)
        color = (np.array(mesh.vertex_colors) * 255).astype(np.float32)
        vert = np.concatenate([coord, color], axis=1)

        vertices = torch.from_numpy(coord)
        faces = torch.from_numpy(np.array(mesh.triangles).astype(np.int64))
        superpoints = segmentator.segment_mesh(vertices, faces).numpy()

        np.save(f"{output_prefix}_vert.npy", vert)
        np.save(f"{output_prefix}_sp_label.npy", superpoints)

        if split == "test":
            print(f"[DONE] {name} ({split}) test_only {time() - start:.1f}s", flush=True)
            return (name, split, "done", "test_only", time() - start)

        with open(segs_path) as f:
            segments = json.load(f)
        with open(anno_path) as f:
            anno = json.load(f)

        seg_indices = np.array(segments["segIndices"], dtype=np.uint32)
        num_vertices = len(seg_indices)
        assert num_vertices == len(coord)

        semantic_gt = np.ones((num_vertices, 3), dtype=np.int32) * ignore_index
        # Keep ScanNet-style convention: 0 means unannotated instance.
        instance_gt = np.zeros((num_vertices, 3), dtype=np.int32)
        instance_size = np.ones((num_vertices, 3), dtype=np.float32) * np.inf
        labels_used = np.zeros(num_vertices, dtype=np.int16)

        for instance in anno["segGroups"]:
            label = instance["label"]
            mapped_label = label_mapping.get(label, None)
            label_index = class2idx.get(mapped_label, ignore_index)

            if label_index == ignore_index:
                continue

            mask = np.isin(seg_indices, instance["segments"]) & (labels_used < 3)
            size = int(mask.sum())
            if size == 0:
                continue

            label_position = labels_used[mask]
            semantic_gt[mask, label_position] = label_index
            instance_gt[mask, label_position] = int(instance["objectId"])
            instance_size[mask, label_position] = size
            labels_used[mask] += 1

        # When vertices have multiple labels, keep "major" label at column 0.
        multi_label_mask = labels_used > 1
        if multi_label_mask.any():
            major_pos = np.argmin(instance_size[multi_label_mask], axis=1)

            major_sem = semantic_gt[multi_label_mask, major_pos]
            semantic_gt[multi_label_mask, major_pos] = semantic_gt[:, 0][multi_label_mask]
            semantic_gt[:, 0][multi_label_mask] = major_sem

            major_ins = instance_gt[multi_label_mask, major_pos]
            instance_gt[multi_label_mask, major_pos] = instance_gt[:, 0][multi_label_mask]
            instance_gt[:, 0][multi_label_mask] = major_ins

        sem_label = semantic_gt[:, 0].astype(np.int32)
        ins_label = instance_gt[:, 0].astype(np.int32)

        np.save(f"{output_prefix}_sem_label.npy", sem_label)
        np.save(f"{output_prefix}_ins_label.npy", ins_label)
        print(f"[DONE] {name} ({split}) ok {time() - start:.1f}s", flush=True)
        return (name, split, "done", "ok", time() - start)
    except Exception as exc:
        print(
            f"[FAILED] {name} ({split}) after {time() - start:.1f}s: {type(exc).__name__}: {exc}",
            flush=True,
        )
        return (name, split, "failed", f"{type(exc).__name__}: {exc}", time() - start)
    finally:
        gc.collect()


def _parse_scene_star(args):
    return parse_scene(*args)


def run_single_process(
    data_list,
    split_list,
    dataset_root,
    output_folder,
    label_mapping,
    class2idx,
    ignore_index,
):
    from tqdm import tqdm

    failures = []
    total = len(data_list)
    for name, split in tqdm(zip(data_list, split_list), total=total):
        scene, scene_split, status, msg, elapsed = parse_scene(
            name=name,
            split=split,
            dataset_root=dataset_root,
            output_folder=output_folder,
            label_mapping=label_mapping,
            class2idx=class2idx,
            ignore_index=ignore_index,
        )
        if status == "failed":
            failures.append((scene, scene_split, msg))
            print(f"[FAILED] {scene} ({scene_split}) after {elapsed:.1f}s: {msg}")

    print(f"Single-process finished. failed={len(failures)}")
    if failures:
        print("First failed scenes:")
        for scene, scene_split, msg in failures[:10]:
            print(f"  - {scene} ({scene_split}): {msg}")


def run_multi_process(
    data_list,
    split_list,
    dataset_root,
    output_folder,
    label_mapping,
    class2idx,
    ignore_index,
    num_workers,
    max_tasks_per_child,
):
    tasks = [
        (
            name,
            split,
            dataset_root,
            output_folder,
            label_mapping,
            class2idx,
            ignore_index,
        )
        for name, split in zip(data_list, split_list)
    ]

    failures = []
    with mp.Pool(processes=num_workers, maxtasksperchild=max_tasks_per_child) as pool:
        for scene, scene_split, status, msg, elapsed in pool.imap_unordered(
            _parse_scene_star, tasks, chunksize=1
        ):
            if status == "failed":
                failures.append((scene, scene_split, msg))
                print(f"[FAILED] {scene} ({scene_split}) after {elapsed:.1f}s: {msg}")

    print(f"Multiprocessing finished. failed={len(failures)}")
    if failures:
        print("First failed scenes:")
        for scene, scene_split, msg in failures[:10]:
            print(f"  - {scene} ({scene_split}): {msg}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        default="/vast/users/rao.anwer/zongyan/datasets/scannet_pp",
        help="Path to the ScanNet++ dataset containing data/metadata/splits.",
    )
    parser.add_argument(
        "--output_folder",
        default="/vast/users/rao.anwer/zongyan/datasets/scannet_pp_processed_aligned",
        help="Output folder with ScanNet-style flat npy files.",
    )
    parser.add_argument(
        "--ignore_index",
        default=-1,
        type=int,
        help="Ignore index for semantic labels.",
    )
    parser.add_argument(
        "--num_workers",
        default=32,
        type=int,
        help="Number of workers for preprocessing.",
    )
    parser.add_argument(
        "--single_process",
        action="store_true",
        help="Run scenes sequentially in a single process for easier debugging.",
    )
    parser.add_argument(
        "--max_tasks_per_child",
        default=20,
        type=int,
        help="Recycle each worker after N scenes to avoid native-memory growth.",
    )
    config = parser.parse_args()

    config.dataset_root = Path(config.dataset_root)

    print("Loading split files...")
    train_list = np.loadtxt(
        config.dataset_root / "splits" / "nvs_sem_train.txt",
        dtype=str,
    )
    val_list = np.loadtxt(
        config.dataset_root / "splits" / "nvs_sem_val.txt",
        dtype=str,
    )
    test_list = np.loadtxt(
        config.dataset_root / "splits" / "sem_test.txt",
        dtype=str,
    )
    print("Num samples in training split:", len(train_list))
    print("Num samples in validation split:", len(val_list))
    print("Num samples in testing split:", len(test_list))

    data_list = np.concatenate([train_list, val_list, test_list])
    split_list = np.concatenate(
        [
            np.full_like(train_list, "train"),
            np.full_like(val_list, "val"),
            np.full_like(test_list, "test"),
        ]
    )

    print("Loading label mapping...")
    segment_class_names = np.loadtxt(
        config.dataset_root / "metadata" / "semantic_benchmark" / "top100.txt",
        dtype=str,
        delimiter=".",
    )
    label_mapping = pd.read_csv(
        config.dataset_root / "metadata" / "semantic_benchmark" / "map_benchmark.csv"
    )
    label_mapping = filter_map_classes(label_mapping, mapping_type="semantic")
    class2idx = {
        class_name: idx for (idx, class_name) in enumerate(segment_class_names)
    }

    print("Processing scenes...")
    if config.single_process or config.num_workers <= 1:
        print("Execution mode: single process")
        run_single_process(
            data_list=data_list,
            split_list=split_list,
            dataset_root=config.dataset_root,
            output_folder=config.output_folder,
            label_mapping=label_mapping,
            class2idx=class2idx,
            ignore_index=config.ignore_index,
        )
    else:
        print(f"Execution mode: multiprocessing ({config.num_workers} workers)")
        run_multi_process(
            data_list=data_list,
            split_list=split_list,
            dataset_root=config.dataset_root,
            output_folder=config.output_folder,
            label_mapping=label_mapping,
            class2idx=class2idx,
            ignore_index=config.ignore_index,
            num_workers=config.num_workers,
            max_tasks_per_child=config.max_tasks_per_child,
        )
    print("Done.")


if __name__ == "__main__":
    main()
