#!/usr/bin/env bash
# Generate every Minari dataset in the benchmark from the available PPO checkpoints.
#
# This is the union of the per-group scripts (getup/pose, locomotion, direction).
# For each (env, command_type) spec we feed ALL of that env's training runs as
# model_checkpoint (collect_data.py merges them into one ordered list of
# step-checkpoints), then collect the four difficulties.
#
# Tasks produced (spec -> resolved task-id):
#   Go2Footstand            (no command)   -> go2-footstand
#   Go2Getup                (no command)   -> go2-getup
#   Go2Handstand            (no command)   -> go2-handstand
#   Go2JoystickFlatTerrain  (fowardfixed)  -> go2-flat-forward        (Tier 1)
#   Go2JoystickFlatTerrain  (no command)   -> go2-joystick-direction  (Tier 2)
#   Go2GetupWalk            (no command)   -> go2-getup-walk
#   Go2PushRecovery         (no command)   -> go2-push-recovery
#   Go2RoughCurriculum      (no command)   -> go2-rough-terrain
#   G1JoystickFlatTerrain   (no command)   -> g1-joystick-direction
#   H1JoystickGaitTracking  (fowardfixed)  -> h1-gait-tracking
#
# Paths are derived from this script's location and can be overridden via env vars.
# Layout assumed (siblings of this repo):
#   <ROOT>/Offline-RL-Benchmark   (this repo, REPO)
#   <ROOT>/CORL                   (venv + expert checkpoint logs)
#   <ROOT>/mujoco_playground      (env registry)
#
# Usage:
#   ./generate_all_datasets.sh                 # all envs, all difficulties
#   ./generate_all_datasets.sh expert medium   # only these difficulties
#
# Overridable env vars:
#   REPO, ROOT, CORL, MUJOCO_PLAYGROUND, PY, LOGS,
#   NUM_SAMPLES, DATASET_VERSION, MAX_EVAL_WORKERS
set -euo pipefail

# --- Generic, overridable paths -------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$SCRIPT_DIR}"
ROOT="${ROOT:-$(dirname "$REPO")}"
CORL="${CORL:-$ROOT/CORL}"
MUJOCO_PLAYGROUND="${MUJOCO_PLAYGROUND:-$ROOT/mujoco_playground}"
PY="${PY:-$CORL/.venv/bin/python}"
LOGS="${LOGS:-$CORL/expert/logs}"
export PYTHONPATH="$REPO:$MUJOCO_PLAYGROUND${PYTHONPATH:+:$PYTHONPATH}"

NUM_SAMPLES=${NUM_SAMPLES:-1000000}
DATASET_VERSION=${DATASET_VERSION:-0}
MAX_EVAL_WORKERS=${MAX_EVAL_WORKERS:-1}

# --- Difficulties: from CLI args, or default to all four. -----------------------
if [ "$#" -gt 0 ]; then
  DIFFICULTIES=("$@")
else
  DIFFICULTIES=(expert medium medium-replay medium-expert)
fi

# --- Dataset specs: "env|command_type|num_envs". --------------------------------
# command_type "-"  => no --command_type flag.
# num_envs     "-"  => omit --num_envs (use collect_data.py default).
# Go2JoystickFlatTerrain appears twice on purpose: the fixed-command Tier 1 task
# and the variable-command Tier 2 task are distinct datasets.
SPECS=(
  "Go2Footstand|-|-"
  "Go2Getup|-|-"
  "Go2Handstand|-|-"
  "Go2JoystickFlatTerrain|fowardfixed|-"
  "Go2JoystickFlatTerrain|-|-"
  "Go2GetupWalk|-|16"
  "Go2PushRecovery|-|16"
  "Go2RoughCurriculum|-|8"
  "G1JoystickFlatTerrain|-|16"
  "H1JoystickGaitTracking|fowardfixed|16"
)

# Collect all run dirs of one env, one per line.
run_dirs() {
  local env="$1"
  for d in "$LOGS/$env"-*/; do
    [ -d "${d}checkpoints" ] && printf '%s\n' "${d}checkpoints"
  done
}

cd "$REPO"
for spec in "${SPECS[@]}"; do
  IFS='|' read -r env cmd_type num_envs <<<"$spec"

  mapfile -t dirs < <(run_dirs "$env")
  if [ "${#dirs[@]}" -eq 0 ]; then
    echo "[skip] $env: no checkpoints found under $LOGS/$env-*/checkpoints"
    continue
  fi
  # pyrallis parses list fields as a single bracketed, comma-separated value.
  ckpt_list="[$(IFS=,; echo "${dirs[*]}")]"

  for diff in "${DIFFICULTIES[@]}"; do
    echo "==================================================================="
    echo "[run] env=$env difficulty=$diff runs=${#dirs[@]} cmd=${cmd_type} num_envs=${num_envs}"
    echo "==================================================================="
    extra=()
    [ "$cmd_type" != "-" ] && extra+=(--command_type "$cmd_type")
    [ "$num_envs" != "-" ] && extra+=(--num_envs "$num_envs")
    "$PY" collect_data.py \
      --env_name "$env" \
      --difficulty "$diff" \
      --num_samples "$NUM_SAMPLES" \
      --dataset_version "$DATASET_VERSION" \
      --max_eval_workers "$MAX_EVAL_WORKERS" \
      --model_checkpoint "$ckpt_list" \
      "${extra[@]}"
  done
done

echo "All datasets generated."
