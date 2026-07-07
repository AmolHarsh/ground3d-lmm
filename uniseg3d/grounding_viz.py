"""Grounding visualization for Ground3D-LMM.

Colors the model's predicted ``<SEG>`` mask(s) onto the scene point cloud and writes a single
``.ply`` per conversation / QA:

* 1 ``<SEG>``  -> the grounded region is painted **red**.
* 2 ``<SEG>``  -> first region **red**, second **green** (e.g. the two objects of a multi-turn
  cross-object dialogue), both baked into **one** ``.ply``.
* N ``<SEG>``  -> a fixed palette (red, green, blue, orange, ...).

Background points keep their (dimmed) true RGB. Uses ``plyfile`` (already a project dependency) —
no Open3D required. This module is import-light so it can be called from the eval loop or offline
via ``tools/export_grounding_ply.py``.
"""
import os
import numpy as np
from plyfile import PlyData, PlyElement

# SEG_1 -> red, SEG_2 -> green, SEG_3 -> blue, ... (RGB uint8)
DEFAULT_PALETTE = [
    (220, 30, 30),     # red
    (45, 175, 70),     # green
    (40, 90, 230),     # blue
    (230, 160, 30),    # orange
    (160, 60, 200),    # purple
]


def _to_bool_1d(mask, n_points):
    """Normalize a predicted mask to a 1-D boolean of length ``n_points``.

    Accepts ``[n_seg, N]`` (unioned over rows), ``[N]``, or object/ragged arrays.
    Returns ``None`` if it cannot be aligned to ``n_points``.
    """
    m = np.asarray(mask)
    if m.dtype == object:
        m = np.asarray(m.item()) if m.ndim == 0 else np.stack([np.asarray(r) for r in m])
    if m.ndim == 2:
        m = m.any(0)
    m = np.asarray(m).astype(bool).reshape(-1)
    return m if m.shape[0] == n_points else None


def masks_to_labels(masks, n_points, max_frac=0.85):
    """List of predicted masks -> per-point int label (``0`` = background, ``i`` = i-th SEG region).

    Each mask is unioned over its rows. Degenerate masks (coverage ``0`` or ``> max_frac`` of the
    scene — i.e. the model grounded "everything") are skipped, mirroring the figure pipeline.
    Earlier masks win on overlap.
    """
    labels = np.zeros(n_points, dtype=np.int32)
    next_id = 0
    for mask in masks:
        m = _to_bool_1d(mask, n_points)
        if m is None:
            continue
        frac = float(m.mean())
        if frac == 0.0 or frac > max_frac:
            continue
        next_id += 1
        labels[m & (labels == 0)] = next_id
    return labels


def write_grounding_ply(out_path, xyz, base_rgb, masks,
                        palette=DEFAULT_PALETTE, max_frac=0.85, dim_bg=0.55):
    """Write one ``.ply`` coloring each predicted SEG region; return the per-point label array.

    Args:
        out_path (str): destination ``.ply``.
        xyz (np.ndarray): ``(N, 3)`` point coordinates.
        base_rgb (np.ndarray): ``(N, 3)`` uint8 true colors (background keeps these, dimmed).
        masks (list): one entry per SEG region (each ``[n_seg, N]`` or ``[N]``).
        palette (list[tuple]): RGB per region; region ``i`` uses ``palette[(i-1) % len]``.
        max_frac (float): drop a mask covering more than this fraction (degenerate grounding).
        dim_bg (float): multiply background RGB by this so colored regions pop.

    Returns:
        np.ndarray: ``(N,)`` int labels (0 = background, i = i-th region).
    """
    xyz = np.asarray(xyz, np.float32)
    n = xyz.shape[0]
    labels = masks_to_labels(masks, n, max_frac=max_frac)
    rgb = (np.asarray(base_rgb, np.float32) * dim_bg).clip(0, 255).astype(np.uint8)
    for i in range(1, int(labels.max()) + 1):
        rgb[labels == i] = palette[(i - 1) % len(palette)]

    verts = np.zeros(n, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                               ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    verts['x'], verts['y'], verts['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    verts['red'], verts['green'], verts['blue'] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    PlyData([PlyElement.describe(verts, 'vertex')], text=False).write(out_path)
    return labels
