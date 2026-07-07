#!/usr/bin/env bash
# Backward-compatible wrapper -> the unified launcher, ScanNet.
# Pass any eval_ground3d.sh flags through, e.g.:
#   bash eval_ground3d_scannet.sh --levels part --sub_tasks distance_estimation --checkpoint work_dirs/ground3dlmm/pytorch_model.pth
# See: bash eval_ground3d.sh --help
exec bash "$(dirname "$0")/eval_ground3d.sh" --dataset scannet "$@"
