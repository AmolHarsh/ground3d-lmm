#!/usr/bin/env bash
# Backward-compatible wrapper -> the unified launcher, ScanNet++.
# See: bash eval_ground3d.sh --help
exec bash "$(dirname "$0")/eval_ground3d.sh" --dataset scannetpp "$@"
