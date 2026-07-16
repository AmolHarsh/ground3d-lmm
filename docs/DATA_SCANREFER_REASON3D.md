# Reproducing the ScanRefer (Table 6) and Reason3D (Table 5) results

Ground3D-LMM is evaluated on two external 3D referring/reasoning-segmentation benchmarks after
per-dataset fine-tuning, in both **3D** (point cloud only) and **3D+2D** (point cloud + the scene's
RGB views) settings. This page covers obtaining the data, building the inputs, and running the four
evaluations.

| Row | Config | Checkpoint (HF) | mIoU (paper) |
|---|---|---|---|
| ScanRefer 3D | `configs/ground3dlmm_eval_scanrefer.py` | `Ground3D-LMM-ScanRefer-4B-3D` | 38.72 |
| ScanRefer 3D+2D | `configs/ground3dlmm_eval_scanrefer_image.py` | `Ground3D-LMM-ScanRefer-4B-Joint` | 41.30 |
| Reason3D 3D | `configs/ground3dlmm_eval_reason3d.py` | `Ground3D-LMM-Reason3D-4B-3D` | 36.35 |
| Reason3D 3D+2D | `configs/ground3dlmm_eval_reason3d_image.py` | `Ground3D-LMM-Reason3D-4B-Joint` | 41.29 |

> Both benchmarks are built on **ScanNet**, so they inherit the [ScanNet Terms of Use](http://www.scan-net.org/).
> The annotations and RGB frames below are **not redistributable** — obtain them from the official
> sources and build the inputs locally with the scripts in this repo.

---

## 0. Prerequisites

- The repo installed per [`docs/INSTALL.md`](INSTALL.md) (mmdet3d, MinkowskiEngine, Qwen3-VL-4B, etc.).
- ScanNet v2 preprocessed into `data/scannet/scannet_instance_data/` (the standard oneformer3d/ScanNet
  point+mask prep — `data/scannet/batch_load_scannet_data.py`, already in this repo).
- `data/scannet/scannet_cls_embedding.pth` (shipped) and the ScanNet split lists under
  `data/scannet/meta_data/`.

---

## 1. Annotations (external)

Download the referring/reasoning annotations from their official releases and group them **per scene**
(one JSON keyed by `scene_id`, each value a list of that scene's QA items):

| Dataset | Source | Per-scene file expected |
|---|---|---|
| ScanRefer | https://github.com/daveredrum/ScanRefer (sign the ScanNet ToU) | `ScanRefer_filtered_{train,val}_by_scene.json` |
| Reason3D | https://github.com/KuanchihHuang/Reason3D | `reason3d_scannet_{train,val}_by_scene.json` |

Reason3D additionally ships per-scene GT tensors `<split>/<scene>_reason.pth` (from its
`prepare_data_reason.py`); ScanRefer's GT comes from the same ScanNet instance masks. Place them so the
converter can find them (see `--qa-data-dir` / `--reason-pth-dir` below).

---

## 2. Build the dataset `.pkl` files

`tools/create_data_scanrefer.py` and `tools/create_data_reason3d.py` turn the ScanNet base + the
per-scene annotations into the `*_infos_{train,val}.pkl` the configs read. Both take the same flags:

```bash
# ScanRefer  ->  data/scannet/oneformer3d_scanrefer/{scanrefer_infos_train.pkl, ..._val.pkl}
python tools/create_data_scanrefer.py \
    --root-path      data/scannet \
    --qa-data-dir    <dir with ScanRefer_filtered_{train,val}_by_scene.json> \
    --reason-pth-dir <dir with <split>/<scene>_reason.pth> \
    --out-dir        data/scannet/oneformer3d_scanrefer

# Reason3D  ->  data/scannet/oneformer3d_reason3d/{reason3d_infos_train.pkl, ..._val.pkl}
python tools/create_data_reason3d.py \
    --root-path      data/scannet \
    --qa-data-dir    <dir with reason3d_scannet_{train,val}_by_scene.json> \
    --reason-pth-dir <dir with <split>/<scene>_reason.pth> \
    --out-dir        data/scannet/oneformer3d_reason3d
```

This writes `points/`, `instance_mask/`, `semantic_mask/`, `super_points/` and the `*_infos_*.pkl`
under each `--out-dir` — exactly the `data_root` + `data_prefix` the configs expect. Also place the
per-scene annotation JSONs under that same dataset root (the configs reference
`<data_root>/<dataset>_..._by_scene.json`).

---

## 3. RGB frames for the 3D+2D rows only

The 3D+2D configs feed the **20 evenly-spaced RGB views** of each scene to the model. Decode the raw
ScanNet `.sens` once (official SensReader — see [`docs/DATA_IMAGES.md`](DATA_IMAGES.md)), then sample
the 20 frames deterministically:

```bash
python tools/build_scene_images.py \
    --pkl      data/scannet/oneformer3d_scanrefer/scanrefer_infos_val.pkl \
    --raw_root $SCANNET_RAW \
    --out      data/scannet/scannet_20_img
# repeat with the reason3d val (and train) pkl(s) to cover all scenes you evaluate
```

`build_scene_images.py` picks `numpy.linspace(0, N-1, 20).astype(int)` frames per scene (`N` = decoded
color-frame count) — a pure function of the frame count, so every user gets the **same** 20 frames.
The decoder reads all `*.jpg` it finds under `image_dir/<scene>/`. **3D-only rows skip this section.**

---

## Expected `data/scannet/` layout (after steps 1–3)

```
data/scannet/
├── scannet_instance_data/            # ScanNet v2 base prep (points + masks)   [prereq]
├── meta_data/                        # split lists                             [shipped]
├── scannet_cls_embedding.pth         #                                         [shipped]
├── oneformer3d_scanrefer/            # <- tools/create_data_scanrefer.py
│   ├── scanrefer_infos_{train,val}.pkl
│   ├── points/  instance_mask/  semantic_mask/  super_points/
│   └── ScanRefer_filtered_{train,val}_by_scene.json
├── oneformer3d_reason3d/             # <- tools/create_data_reason3d.py
│   ├── reason3d_infos_{train,val}.pkl
│   ├── points/  instance_mask/  semantic_mask/  super_points/
│   └── reason3d_scannet_{train,val}_by_scene.json
└── scannet_20_img/                   # <- tools/build_scene_images.py  (3D+2D rows only)
    └── <scene_id>/*.jpg
```

- **`data_root` per config** — ScanRefer → `data/scannet/oneformer3d_scanrefer`, Reason3D →
  `data/scannet/oneformer3d_reason3d`; the 3D+2D configs additionally read
  `image_dir=data/scannet/scannet_20_img`.
- **3D vs 3D+2D share inputs, never clobber outputs.** A dataset's 3D and 3D+2D rows use the *same*
  `.pkl`/annotations and differ only by the config (RGB frames + LoRA rank). Each of the four evals
  writes to its own `save_pred_qa_dir` — `pred_qa_scanrefer`, `pred_qa_scanrefer_image`,
  `pred_qa_reason3d`, `pred_qa_reason3d_image` — so you can run all four back-to-back safely.

---

## 4. Evaluate

Download the checkpoint for the row from Hugging Face, then run:

```bash
# example: ScanRefer 3D+2D
bash tools/dist_test.sh \
    configs/ground3dlmm_eval_scanrefer_image.py \
    <path/to/Ground3D-LMM-ScanRefer-4B-Joint/pytorch_model.pth> \
    4
```

Swap the config + checkpoint per the table at the top. The metric prints
`QA Seg mIoU` (= mIoU), `QA Seg Precision_quarter` (= Acc@0.25), and `QA Seg Precision_half` (= Acc@0.50).

---

## Notes / gotchas

- **Run-to-run variance (~±1–1.5 pt).** Decoding is greedy, but the point pooling / sparse convs use
  non-deterministic GPU atomics, so mIoU wobbles ~1 pt between runs even with the seed fixed. Treat a
  result within ~1.5 pt of the table as a match; don't over-index on a single epoch.
- **Multi-GPU eval timeout.** The 3D+2D rows feed 20 images/scene and are ~2× slower, so the final
  cross-rank metric gather can exceed the default 10-min NCCL timeout and abort. The configs set
  `env_cfg.dist_cfg.timeout=7200` to prevent this; keep it (or run single-GPU).
- **Which epoch.** The released checkpoints are the best-val epoch per row; given the variance above,
  expect the reported numbers within noise rather than to the decimal.
