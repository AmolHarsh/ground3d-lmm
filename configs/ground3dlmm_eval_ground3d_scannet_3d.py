"""ScanNet evaluation config for the point-only (3D-only) checkpoint.

Inherits the shared ScanNet joint eval config and overrides only the
three fields that differ between the two released variants:

  * ``lora_r`` / ``lora_alpha`` — the point-only checkpoint
    (``amolharsh/Ground3D-LMM-4B-3D``) was trained with a wider LoRA
    (r=32, alpha=64) than the joint checkpoint (r=16, alpha=32); loading
    the point-only weights into an r=16 model produces mismatched
    adapters that get silently ignored by ``strict=False``, so the model
    silently falls back to base Qwen. Matching the config to the
    checkpoint keeps the adapters live.

  * ``image_dir`` — the 3D-only variant does not consume RGB frames.

For the joint (3D + 2D) variant, use ``ground3dlmm_eval_ground3d_scannet.py``.
"""

_base_ = ['./ground3dlmm_eval_ground3d_scannet.py']

model = dict(
    decoder=dict(
        lora_r=32,
        lora_alpha=64,
        image_dir=None,
    ),
)
