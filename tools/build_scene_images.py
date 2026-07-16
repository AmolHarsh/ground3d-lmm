#!/usr/bin/env python3
"""Build the per-scene 20-frame RGB sets the Joint model reads as multi-view context.

WHAT THIS PRODUCES
------------------
For each scene it writes 20 evenly-spaced RGB frames to::

    <out>/<scene>/<frame_index>.jpg

which is exactly the layout the decoder expects from ``model.decoder.image_dir``.
At inference the decoder lists ``<image_dir>/<scene>/`` (sorted by numeric file
name, skipping ``._`` AppleDouble files) and feeds *all* frames found there to the
vision-language backbone as the 2D view of the scene. Pairing those 20 views with
the point cloud is what turns the 3D-only grounding setup (ScanRefer / Reason3D
style: a text query resolved to a 3D instance mask) into the joint 3D+2D setting:
the language model sees the room from 20 viewpoints in addition to the geometry.

THE SAMPLING SCHEME (deterministic, no randomness)
--------------------------------------------------
A ScanNet scene decodes to a contiguous run of color frames ``0 .. N-1`` under
``scannet/scans/<scene>/color/<i>.jpg`` (produced by the official SensReader; see
``tools/extract_frame.py`` and ``docs/DATA_IMAGES.md``). We pick 20 frames as::

    indices = numpy.linspace(0, N - 1, 20).astype(int)

``.astype(int)`` truncates toward zero (it does NOT round half-to-even), so e.g.
``linspace(0, 5577, 20)`` -> ``0, 293, 587, 880, ...`` and not ``0, 294, 587, ...``.
This is a pure function of the single integer ``N`` (the frame count), so the output
is bit-for-bit reproducible. ``N`` is the number of decoded ``*.jpg`` files in the
scene's ``color/`` directory (equivalently ``max_index + 1``, since the frames are a
contiguous ``0..N-1`` range); the largest sampled index is always ``N-1``, so reading
``N`` as the count or as ``max_index + 1`` yields identical results.

Scenes with fewer than 20 decoded frames are handled gracefully: every available
frame is taken (no duplication, no padding).

USAGE
-----
Provide the list of scenes either from a dataset infos ``.pkl`` (each sample's point
cloud path encodes ``<scene>_frame_<n>``) or from a plain text/glob ``--scenes``
source, plus the root that holds the decoded ScanNet color frames::

    python tools/build_scene_images.py \
        --pkl      data/scannet/part3d_glamm_infos_val.pkl \
        --raw_root $DATA_ROOT \
        --out      data/scannet/scene_images

    python tools/build_scene_images.py \
        --scenes   "scene0000_00 scene0001_00" \
        --raw_root $DATA_ROOT \
        --out      data/scannet/scene_images

``--raw_root`` is the data root containing ``scannet/scans/<scene>/color/<i>.jpg``
(decode the raw ``<scene>.sens`` once with the official SensReader first; the recipe
lives in ``tools/extract_frame.py`` / ``docs/DATA_IMAGES.md``).
"""
import argparse
import glob
import os
import pickle
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract_frame as ef  # noqa: E402  (reuse the SensReader color-frame recipe)

NUM_FRAMES = 20
SCENE_RE = re.compile(r"scene\d{4}_\d{2}")


def list_color_indices(raw_root, scene):
    """Return the sorted decoded color-frame indices for a scene (skip ``._`` files).

    Frames live at ``<raw_root>/scannet/scans/<scene>/color/<i>.jpg`` after the
    official SensReader unpack. Returns an ascending list of integer indices.
    """
    color_dir = os.path.join(raw_root, "scannet", "scans", scene, "color")
    if not os.path.isdir(color_dir):
        return None
    idx = []
    for f in os.listdir(color_dir):
        if f.startswith("._") or not f.lower().endswith(".jpg"):
            continue
        stem = f[:-4]
        if stem.isdigit():
            idx.append(int(stem))
    return sorted(idx)


def sample_indices(frame_indices):
    """Pick the evenly-spaced subset of frame indices for a scene.

    ``frame_indices`` is the ascending list of available color-frame indices
    (a contiguous ``0..N-1`` range for a normally decoded scene). With N>=20 this
    returns ``linspace(0, N-1, 20)`` truncated to int and mapped back onto the
    available indices; with fewer than 20 frames it returns all of them.
    """
    n = len(frame_indices)
    if n == 0:
        return []
    if n <= NUM_FRAMES:
        return list(frame_indices)
    positions = np.linspace(0, n - 1, NUM_FRAMES).astype(int)
    return [frame_indices[p] for p in positions]


def scenes_from_pkl(pkl_path):
    """Collect the unique scene ids referenced by a dataset infos ``.pkl``."""
    with open(pkl_path, "rb") as fh:
        d = pickle.load(fh)
    data_list = d["data_list"] if isinstance(d, dict) and "data_list" in d else d
    scenes = []
    seen = set()
    for s in data_list:
        name = os.path.basename(s["lidar_points"]["lidar_path"]).rsplit(".", 1)[0]
        scene = name.split("_frame_")[0] if "_frame_" in name else name
        if scene not in seen:
            seen.add(scene)
            scenes.append(scene)
    return scenes


def scenes_from_arg(spec):
    """Resolve ``--scenes`` (a file, a glob, or a whitespace/comma list) to scene ids."""
    if os.path.isfile(spec):
        with open(spec) as fh:
            tokens = fh.read().split()
    else:
        matches = glob.glob(spec)
        if matches:
            tokens = [os.path.basename(m.rstrip("/")) for m in matches]
        else:
            tokens = re.split(r"[\s,]+", spec.strip())
    out, seen = [], set()
    for t in tokens:
        m = SCENE_RE.search(t)
        scene = m.group(0) if m else t
        if scene and scene not in seen:
            seen.add(scene)
            out.append(scene)
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pkl", help="dataset infos .pkl; scenes read from sample paths")
    src.add_argument("--scenes", help="scene-list file, glob, or whitespace/comma list")
    ap.add_argument("--raw_root", required=True,
                    help="root holding scannet/scans/<scene>/color/<i>.jpg (decode .sens first)")
    ap.add_argument("--out", required=True,
                    help="output dir (= model.decoder.image_dir): <out>/<scene>/<idx>.jpg")
    a = ap.parse_args()

    scenes = scenes_from_pkl(a.pkl) if a.pkl else scenes_from_arg(a.scenes)
    if not scenes:
        sys.exit("No scenes resolved from the given --pkl/--scenes.")

    total_written, total_missing = 0, 0
    for scene in scenes:
        frame_indices = list_color_indices(a.raw_root, scene)
        if frame_indices is None:
            print(f"[skip] {scene}: no color/ directory under raw_root "
                  f"(run the SensReader to decode <scene>.sens first)")
            total_missing += 1
            continue
        if not frame_indices:
            print(f"[skip] {scene}: color/ directory is empty")
            total_missing += 1
            continue

        chosen = sample_indices(frame_indices)
        scene_out = os.path.join(a.out, scene)
        os.makedirs(scene_out, exist_ok=True)
        written = 0
        for idx in chosen:
            dst = os.path.join(scene_out, f"{idx}.jpg")
            if os.path.exists(dst):
                written += 1
                continue
            ef.extract_scannet(a.raw_root, scene, idx, dst)  # copies color/<idx>.jpg
            written += 1

        note = "" if len(chosen) == NUM_FRAMES else f"  (only {len(frame_indices)} frames available)"
        print(f"[ok]   {scene}: N={len(frame_indices)} -> {written} frame(s): "
              f"{chosen[:3]}...{chosen[-3:]}{note}")
        total_written += written

    print(f"\nDone: {total_written} frame(s) under {a.out}; "
          f"{len(scenes) - total_missing}/{len(scenes)} scene(s) built, {total_missing} skipped.")


if __name__ == "__main__":
    main()
