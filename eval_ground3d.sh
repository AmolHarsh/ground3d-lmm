#!/usr/bin/env bash
# Unified Ground3D evaluation launcher — segmentation mIoU (per sub-task) + per-scene prediction JSONs.
# Choose what to run with flags (no editing/commenting required):
#
#   bash eval_ground3d.sh [options]
#     --dataset     scannet | scannetpp | all              (default: scannet)
#     --levels      part,object,multi_turn | all           (default: part,object)
#     --sub_tasks   all | <comma-separated sub-tasks>      (default: all)
#     --checkpoint  <path>                                 (default: work_dirs/ground3dlmm/pytorch_model.pth)
#     --variant     joint | 3d | auto                      (default: auto — detected from checkpoint LoRA rank)
#     --gpus        <N>                                     (default: auto-detected)
#     --out-root    <dir>                                   (default: val_outputs_<variant>)
#     --mask-dump   <dir>                                   (optional: also dump per-QA masks -> viz / GM-delta)
#
# Variants:
#     joint  = 3D + 2D checkpoint  (amolharsh/Ground3D-LMM-4B-Joint, LoRA r=16, uses RGB frames)
#     3d     = point-only checkpoint (amolharsh/Ground3D-LMM-4B-3D,     LoRA r=32, no RGB frames)
#     auto   = probe the checkpoint's LoRA rank and pick the matching config
#
# Examples:
#     bash eval_ground3d.sh --dataset all --levels all
#     bash eval_ground3d.sh --dataset scannet --levels part --sub_tasks distance_estimation,scale_estimation
#     bash eval_ground3d.sh --dataset scannet --levels part --sub_tasks grounded_dimension_reasoning \
#                           --mask-dump viz_dumps     # then GM-delta (see tools/evaluate_text_metrics.py --stages gmdelta)
#     bash eval_ground3d.sh --variant 3d --checkpoint /path/to/Ground3D-LMM-4B-3D/pytorch_model.pth
set -euo pipefail

DATASET=scannet
LEVELS=part,object
SUB_TASKS=all
CHECKPOINT=work_dirs/ground3dlmm/pytorch_model.pth
VARIANT=auto
GPUS=""
OUT_ROOT=""            # default set after variant is resolved (variant-specific, so 3d/joint outputs don't collide)
MASK_DUMP=""

while [ $# -gt 0 ]; do
  case "$1" in
    --dataset)    DATASET=$2;    shift 2;;
    --levels)     LEVELS=$2;     shift 2;;
    --sub_tasks)  SUB_TASKS=$2;  shift 2;;
    --checkpoint) CHECKPOINT=$2; shift 2;;
    --variant)    VARIANT=$2;    shift 2;;
    --gpus)       GPUS=$2;       shift 2;;
    --out-root)   OUT_ROOT=$2;   shift 2;;
    --mask-dump)  MASK_DUMP=$2;  shift 2;;
    -h|--help)    grep '^#' "$0" | sed 's/^#\s\?//'; exit 0;;
    *) echo "Unknown option: $1 (see --help)" >&2; exit 1;;
  esac
done

case "$VARIANT" in joint|3d|auto) ;;
  *) echo "bad --variant: $VARIANT (expected joint | 3d | auto)" >&2; exit 1;;
esac

# --- variant detection & suffix -------------------------------------------------------------
# The joint checkpoint (Ground3D-LMM-4B-Joint) was trained at LoRA r=16; the point-only
# checkpoint (Ground3D-LMM-4B-3D) at r=32. Loading a checkpoint into a config with the wrong
# rank leaves the adapters silently unloaded (strict=False), which reverts inference to base
# Qwen. So we probe the checkpoint's LoRA rank and pick the matching config.
probe_lora_rank () {
  local ckpt=$1
  python - "$ckpt" <<'PY'
import sys, torch
p = sys.argv[1]
try:
    sd = torch.load(p, map_location='cpu', weights_only=False)
except TypeError:  # older torch without weights_only
    sd = torch.load(p, map_location='cpu')
if isinstance(sd, dict) and 'state_dict' in sd:
    sd = sd['state_dict']
ranks = set()
for k, v in sd.items():
    if 'lora_A' in k and hasattr(v, 'shape') and v.ndim >= 1:
        ranks.add(int(v.shape[0]))
if not ranks:
    print("none"); sys.exit(0)
if len(ranks) > 1:
    print(f"mixed:{','.join(str(r) for r in sorted(ranks))}"); sys.exit(0)
print(next(iter(ranks)))
PY
}

if [ "$VARIANT" = auto ]; then
  if [ ! -f "$CHECKPOINT" ]; then
    echo "checkpoint not found at $CHECKPOINT — cannot auto-detect variant." >&2
    echo "Pass --variant joint|3d explicitly, or point --checkpoint at a real file." >&2
    exit 1
  fi
  echo "[variant] probing checkpoint LoRA rank: $CHECKPOINT"
  RANK="$(probe_lora_rank "$CHECKPOINT")"
  case "$RANK" in
    16) VARIANT=joint; echo "[variant] rank=16 → joint (r=16, uses RGB frames)";;
    32) VARIANT=3d;    echo "[variant] rank=32 → 3d    (r=32, no RGB frames)";;
    none)
      echo "[variant] no LoRA weights found in checkpoint; defaulting to joint." >&2
      VARIANT=joint;;
    mixed:*)
      echo "[variant] $RANK — checkpoint mixes multiple LoRA ranks; pick --variant explicitly." >&2
      exit 1;;
    *)
      echo "[variant] unrecognized LoRA rank ($RANK); pick --variant explicitly." >&2
      exit 1;;
  esac
fi
[ "$VARIANT" = 3d ] && CONFIG_SUFFIX="_3d" || CONFIG_SUFFIX=""
# Default output root is variant-specific so a 3d run never overwrites a joint run's predictions
# (and vice versa). Override with --out-root to choose your own.
[ -z "$OUT_ROOT" ] && OUT_ROOT="val_outputs_$VARIANT"

ALL_SUB_TASKS="functional_part_grounding functional_object_grounding scale_comparison_size \
distance_estimation relative_position_forward_reasoning relative_depth_forward \
existence_verification scale_estimation grounded_dimension_reasoning"

case "$DATASET" in
  all)                 DATASETS="scannet scannetpp";;
  scannet|scannetpp)   DATASETS="$DATASET";;
  *) echo "bad --dataset: $DATASET" >&2; exit 1;;
esac
[ "$LEVELS" = all ] && LEVELS="part,object,multi_turn"
declare -A LEVEL_MAP=( [part]=part_qa_data [object]=object_qa_data [multi_turn]=multi_turn_qa_data )
if [ "$SUB_TASKS" = all ]; then SUBS="$ALL_SUB_TASKS"; else SUBS=$(echo "$SUB_TASKS" | tr ',' ' '); fi
if [ -z "$GPUS" ]; then
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}'); else GPUS=1; fi
fi
DIR="$(cd "$(dirname "$0")" && pwd)"

run_test () {                       # $1=config  $2=save_dir  $3..=extra --cfg-options
  local config=$1 save_dir=$2; shift 2
  if [ -n "${DRYRUN:-}" ]; then
    echo "  [dry-run] config=$(basename "$config")  save_dir=$save_dir  gpus=$GPUS  mask_dump=${MASK_DUMP:-none}"
    return 0
  fi
  local port; while true; do port=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 )); \
    nc -z 127.0.0.1 "$port" </dev/null &>/dev/null || break; done
  # mirror the prediction tree under the mask-dump root so GM-delta can line them up 1:1
  local dump_env=""; [ -n "$MASK_DUMP" ] && dump_env="MASK_DUMP_DIR=$MASK_DUMP/${save_dir#"$OUT_ROOT/"}"
  env PYTHONPATH="$DIR:${PYTHONPATH:-}" $dump_env \
    python -m torch.distributed.launch --nproc_per_node="$GPUS" --master_port="$port" \
      "$DIR/tools/test.py" "$config" "$CHECKPOINT" --launcher pytorch \
      --work-dir "$(dirname "$save_dir")/work" \
      --cfg-options "model.decoder.save_pred_qa_dir=$save_dir" "$@"
}

for ds in $DATASETS; do
  CONFIG="$DIR/configs/ground3dlmm_eval_ground3d_${ds}${CONFIG_SUFFIX}.py"
  BASE="$OUT_ROOT/$ds/pred_qa_data_val"
  IFS=',' read -ra LV <<< "$LEVELS"
  for lv in "${LV[@]}"; do
    task_level=${LEVEL_MAP[$lv]:-}
    [ -z "$task_level" ] && { echo "bad --levels entry: $lv" >&2; exit 1; }
    if [ "$task_level" = multi_turn_qa_data ]; then
      echo "======== $ds | $task_level ========"
      run_test "$CONFIG" "$BASE/$task_level" \
        "test_dataloader.dataset.pipeline.2.transforms.2.task_level=$task_level"
    else
      for sub in $SUBS; do
        echo "======== $ds | $task_level | $sub ========"
        run_test "$CONFIG" "$BASE/$task_level/qa_data/$sub" \
          "test_dataloader.dataset.pipeline.2.transforms.2.task_level=$task_level" \
          "test_dataloader.dataset.pipeline.2.transforms.2.task_type=qa_data" \
          "test_dataloader.dataset.pipeline.2.transforms.2.sub_task=$sub"
      done
    fi
  done
done
echo "DONE. predictions under $OUT_ROOT/<dataset>/pred_qa_data_val/"
