# Installation

Ground3D-LMM is built on the OpenMMLab stack (MMEngine + mmdet3d). The install order matters because some packages (mmcv, spconv, torch-scatter) ship CUDA-specific wheels that must match your PyTorch + CUDA versions exactly.

## Tested environment

| Component | Version |
|---|---|
| OS | Ubuntu 20.04 / 22.04 |
| Python | 3.10 |
| CUDA | 11.8 |
| GPU | NVIDIA A100 80GB, RTX 3090 24GB |
| PyTorch | 2.4.1 |

Other CUDA versions (12.x) and PyTorch versions may work but are not validated.

## Option A — Conda (recommended)

```bash
conda env create -f environment.yml
conda activate ground3d
pip install -e .
```

If `conda env create` fails on one of the OpenMMLab packages, fall back to Option B below.

## Option B — Manual install

```bash
# 1. Create environment
conda create -n ground3d python=3.10 -y
conda activate ground3d

# 2. PyTorch with CUDA 11.8
pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 torchaudio==2.4.1+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# 3. OpenMMLab stack (order matters!)
pip install -U openmim
mim install "mmengine==0.10.7"

# mmcv 2.1.0 ships no prebuilt wheel for torch 2.4.1, so pip builds it from
# source. Pin an older setuptools (which still provides `pkg_resources`) and
# disable build isolation, otherwise the build fails with
# "ModuleNotFoundError: No module named 'pkg_resources'".
export CUDA_HOME=/usr/local/cuda-11.8
pip install "setuptools<70" wheel ninja
MMCV_WITH_OPS=1 pip install "mmcv==2.1.0" --no-build-isolation

mim install "mmdet==3.3.0"
mim install "mmdet3d==1.4.0"

# 4. Sparse-conv and scatter (CUDA-specific wheels)
pip install spconv-cu118==2.3.8
pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.4.0+cu118.html

# 4b. MinkowskiEngine 0.5.4 — REQUIRED (imported at the top of uniseg3d/uniseg3d.py).
#     Built from source; needs the CUDA toolkit and OpenBLAS headers. This is the
#     most fragile dependency; do not skip it (without it `import uniseg3d` fails).
sudo apt-get install -y libopenblas-dev    # prefer SYSTEM OpenBLAS — conda-forge openblas-devel can
                                           # trigger a "libgfortran.so.3 / GFORTRAN_1.0 not found" runtime error.
export CUDA_HOME=/usr/local/cuda-11.8 MAX_JOBS=8
pip install "setuptools<70"                # newer setuptools/pip dropped the build flags ME relies on
# NOTE: the v0.5.4 *tag* fails to compile against torch 2.4 (CUDA template errors).
# Build from the master branch instead (it still self-reports version 0.5.4);
# the tested environment used commit 02fc608.
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine && python setup.py install --blas=openblas --force_cuda && cd ..

# 5. LLM stack
pip install transformers==4.57.6 peft==0.18.1 huggingface_hub accelerate deepspeed

# 6. Misc
pip install "numpy<2.0" scipy scikit-learn plyfile tensorboard

# 6b. Eval Steps 2-3 (tools/evaluate_text_metrics.py: APE/delta + LLM judge) need only
#     transformers + torch (installed above) — no extra packages required.

# 7. This repo (editable install)
pip install -e .
```

## Verify the install

```bash
python -c "
import torch, mmengine, mmcv, mmdet, mmdet3d, spconv, torch_scatter, transformers, peft
import MinkowskiEngine
print('torch    :', torch.__version__, 'CUDA:', torch.version.cuda)
print('MinkowskiEngine:', MinkowskiEngine.__version__)
print('mmengine :', mmengine.__version__)
print('mmcv     :', mmcv.__version__)
print('mmdet    :', mmdet.__version__)
print('mmdet3d  :', mmdet3d.__version__)
print('spconv   :', spconv.__version__)
print('scatter  :', torch_scatter.__version__)
print('transformers:', transformers.__version__)
print('peft     :', peft.__version__)
print('GPUs     :', torch.cuda.device_count())
"
```

Expected output:
```
torch    : 2.4.1+cu118 CUDA: 11.8
mmengine : 0.10.7
mmcv     : 2.1.0
mmdet    : 3.3.0
mmdet3d  : 1.4.0
spconv   : 2.3.8
scatter  : 2.1.2+pt24cu118
transformers: 4.57.6
peft     : 0.18.1
GPUs     : 1   (or however many)
```

## Quick smoke test (imports + model build, no data needed)

```bash
PYTHONPATH=. python -c "
from mmengine.config import Config
from mmdet3d.utils import register_all_modules
from mmdet3d.registry import MODELS
import uniseg3d  # registers all model/transform/dataset classes
register_all_modules()  # set default scope to mmdet3d (classes register there)
cfg = Config.fromfile('configs/ground3dlmm_eval_ground3d_scannet.py')
# Override Qwen path to use a smaller model or local copy if needed:
# cfg.model['decoder']['qwen_model_path'] = '/path/to/Qwen3-VL-4B-Instruct'
model = MODELS.build(cfg.model)
n_params = sum(p.numel() for p in model.parameters()) / 1e9
print(f'OK: model built. {n_params:.2f}B params total.')
"
```

If this prints `OK: model built. 4.47B params total.` you're ready to train/eval.

## Common issues

### `ModuleNotFoundError: No module named 'mmcv._ext'`

mmcv was compiled for a different torch / CUDA combo. Reinstall:
```bash
pip uninstall -y mmcv mmcv-full
mim install --force-reinstall "mmcv==2.1.0"
```

### `ImportError: ... libcuda.so.1: cannot open shared object file`

CUDA runtime not on `LD_LIBRARY_PATH`. Add:
```bash
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
```

### `RuntimeError: CUDA error: no kernel image is available for execution on the device`

Your GPU compute capability isn't supported by the prebuilt spconv wheel.
- For sm_90 (H100): use `spconv-cu120` instead and switch to PyTorch CUDA 12.x.
- For older GPUs (sm_60 and below): compile spconv from source.

### `torch-scatter` install fails

Make sure the `-f https://data.pyg.org/whl/torch-2.4.0+cu118.html` flag is present — without it pip tries to compile from source and usually fails.

### `transformers` complains about Qwen3-VL not found

Qwen3-VL support was added in transformers 4.45+. Ensure your installed version is ≥ 4.45.

## GPU memory requirements

| Use case | Min GPU memory |
|---|---|
| Inference (single scene) | ~16 GB |
| LoRA fine-tuning (batch=1) | ~40 GB (A100 40GB minimum) |
| LoRA fine-tuning (batch=4) | ~80 GB (A100 80GB recommended) |

For training on smaller GPUs (24 GB), reduce `max_point_tokens` in the config from 2000 to 1500 and/or shorten `model.decoder.max_seq_length`.
