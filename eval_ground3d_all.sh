#!/usr/bin/env bash
# Backward-compatible wrapper -> the unified launcher, both datasets.
# See: bash eval_ground3d.sh --help
exec bash "$(dirname "$0")/eval_ground3d.sh" --dataset all "$@"
