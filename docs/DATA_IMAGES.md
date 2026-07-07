# RGB frames for the Joint model (ScanNet & ScanNet++)

The **Joint** checkpoint (`Ground3D-LMM-4B-Joint`, the main paper result) feeds the model **one RGB
frame** per question, alongside the point cloud. The eval config reads it from:

```python
model.decoder.image_dir = 'data/scannet/frame_images'   # -> <image_dir>/<scene>/<frame>.jpg
```

i.e. for a sample whose point cloud is `<scene>_frame_<N>.bin`, the model loads
`frame_images/<scene>/<N>.jpg`. **Only the RGB image is used at inference** — pose, intrinsics and
depth are *not* model inputs (they are dataset metadata).

> The **3D-only** checkpoint (`Ground3D-LMM-4B-3D`) needs **no images** — skip this whole page and
> set `model.decoder.image_dir=None`.

These frames are **ScanNet / ScanNet++ imagery and cannot be redistributed** (dataset licenses), so
they are **not** part of the Ground3D HF dataset. Obtain them from the official sources below, then
run the bundled extraction scripts.

---

## 1. Get the raw datasets (official sources)

| Dataset | Where | What you need |
|---|---|---|
| **ScanNet v2** | https://github.com/ScanNet/ScanNet (sign the ToU; they email `download-scannet.py`) | each scene's `<scene>.sens` (packed RGB-D stream) |
| **ScanNet++** | https://kaldir.vc.in.tum.de/scannetpp/ + https://github.com/scannetpp/scannetpp | each scene's `iphone/rgb.mkv` |

Arrange them under a single root (`$GROUND3D_DATA`):

```
$GROUND3D_DATA/
├── scannet/scans/<scene>/<scene>.sens            # ScanNet: raw packed stream
└── scannetpp/data/<scene_id>/iphone/rgb.mkv      # ScanNet++: raw RGB video
```

## 2. One-time unpack

**ScanNet** — decode `.sens` to per-frame files with the official SensReader
(https://github.com/ScanNet/ScanNet/tree/master/SensReader/python):

```bash
python reader.py --filename $GROUND3D_DATA/scannet/scans/<scene>/<scene>.sens \
                 --output_path $GROUND3D_DATA/scannet/scans/<scene> --export_color_images
# -> $GROUND3D_DATA/scannet/scans/<scene>/color/<frame>.jpg
```

**ScanNet++** — nothing to unpack; frames are decoded straight from `rgb.mkv` by index (the QA
`frame_num` is the 0-based `rgb.mkv` frame index).

## 3. Produce `frame_images/<scene>/<frame>.jpg`

Two ways, both shipped in `tools/`:

**A. Batch (recommended) — extract exactly the frames a split needs, from its `.pkl`:**

```bash
python tools/build_frame_images.py \
    --pkl      data/scannet/part3d_glamm_infos_val.pkl \
    --raw_root $GROUND3D_DATA \
    --out      data/scannet/frame_images
```

This walks every `(scene, frame)` in the pkl and writes `frame_images/<scene>/<frame>.jpg` — the
exact layout `model.decoder.image_dir` expects. ScanNet frames are copied from `color/`; ScanNet++
frames are decoded from `rgb.mkv` (OpenCV `CAP_PROP_POS_FRAMES`, matching the dataset generator;
falls back to `ffmpeg`).

**B. Single frame (per question), e.g. for a demo:**

```bash
python tools/extract_frame.py --dataset scannet  --root $GROUND3D_DATA --scene scene0019_00 --frame 195 --out f.jpg
python tools/extract_frame.py --qa_file <task>/<scene>_frame_<frame>.json --root $GROUND3D_DATA --out f.jpg
```

## 4. Point the config at it

```bash
... --cfg-options model.decoder.image_dir=data/scannet/frame_images   # ScanNet
... --cfg-options model.decoder.image_dir=data/scannetpp/frame_images  # ScanNet++
```

---

## Notes / gotchas

- **ScanNet++ uses a different pose/intrinsic convention** from ScanNet (per-frame intrinsics in
  `iphone/pose_intrinsic_imu.json`; COLMAP/OpenCV poses). Irrelevant for inference (image only), but
  matters if you render your own camera views.
- A small **sample dataset** mirroring the real on-disk layout (1 ScanNet + 1 ScanNet++ scene, a few
  real frames) is handy for testing the extraction scripts before downloading the full datasets.
- Frame ids are the **raw sensor / video indices** (e.g. ScanNet `scene0000_00` has frames `0…5577`).
