#!/usr/bin/env bash
set -euo pipefail

# Run from the project root:
#   bash scripts/batch_train_rl.sh
#
# Optional overrides:
#   PYTHON=/path/to/python EPISODES=500 MAX_STEPS=1200 REPEATS=5 BASE_SEED=0 bash scripts/batch_train_rl.sh
#   SPEED_PROFILE=A1 bash scripts/batch_train_rl.sh
#   SPEED_PROFILE=CUSTOM PROXIMAL_PUSH_MM=0.90 DISTAL_PUSH_MM=0.35 SWITCH_PROGRESS=0.82 bash scripts/batch_train_rl.sh

PYTHON_BIN="${PYTHON:-python}"
EPISODES="${EPISODES:-500}"
MAX_STEPS="${MAX_STEPS:-1200}"
REPEATS="${REPEATS:-2}"
BASE_SEED="${BASE_SEED:-0}"
SUCCESS_PROGRESS="${SUCCESS_PROGRESS:-0.92}"
EVAL_FREQ="${EVAL_FREQ:-5000}"
EVAL_EPISODES="${EVAL_EPISODES:-1}"
SPEED_PROFILE="${SPEED_PROFILE:-B0}"
PROXIMAL_PUSH_MM="${PROXIMAL_PUSH_MM:-}"
DISTAL_PUSH_MM="${DISTAL_PUSH_MM:-}"
SWITCH_PROGRESS="${SWITCH_PROGRESS:-}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${OUTPUT_ROOT:-projects/cache/rl_batch/${RUN_ID}}"
ALGORITHMS="${ALGORITHMS:-PPO SAC TD3 DDPG DQN}"
ENTRIES="${ENTRIES:-0 1}"
EXITS="${EXITS:-0 1 2}"

mkdir -p "${OUTPUT_ROOT}/logs"

echo "[batch] python=${PYTHON_BIN}"
echo "[batch] algorithms=${ALGORITHMS}"
echo "[batch] episodes=${EPISODES} max_steps=${MAX_STEPS} base_seed=${BASE_SEED}"
echo "[batch] entries=${ENTRIES} exits=${EXITS} repeats=${REPEATS}"
echo "[batch] speed_profile=${SPEED_PROFILE} proximal=${PROXIMAL_PUSH_MM:-preset} distal=${DISTAL_PUSH_MM:-preset} switch=${SWITCH_PROGRESS:-preset}"
echo "[batch] output=${OUTPUT_ROOT}"

speed_args=(--speed-profile "${SPEED_PROFILE}")
if [[ -n "${PROXIMAL_PUSH_MM}" ]]; then
  speed_args+=(--proximal-push-mm "${PROXIMAL_PUSH_MM}")
fi
if [[ -n "${DISTAL_PUSH_MM}" ]]; then
  speed_args+=(--distal-push-mm "${DISTAL_PUSH_MM}")
fi
if [[ -n "${SWITCH_PROGRESS}" ]]; then
  speed_args+=(--switch-progress "${SWITCH_PROGRESS}")
fi

for algo in ${ALGORITHMS}; do
  for entry in ${ENTRIES}; do
    for exit_idx in ${EXITS}; do
      for ((repeat=0; repeat<REPEATS; repeat++)); do
        seed=$((BASE_SEED + repeat))
        route="entry$((entry + 1))_exit$((exit_idx + 1))"
        log_dir="${OUTPUT_ROOT}/logs/${algo,,}/${route}"
        mkdir -p "${log_dir}"
        log_file="${log_dir}/repeat_$(printf '%02d' "${repeat}").log"
        echo "[batch] start algo=${algo} route=${route} repeat=${repeat} seed=${seed}"
        "${PYTHON_BIN}" scripts/train_rl_cli.py \
          --algo "${algo}" \
          --entry "${entry}" \
          --exit "${exit_idx}" \
          --episodes "${EPISODES}" \
          --max-steps "${MAX_STEPS}" \
          --success-progress "${SUCCESS_PROGRESS}" \
          --eval-freq "${EVAL_FREQ}" \
          --eval-episodes "${EVAL_EPISODES}" \
          --output-root "${OUTPUT_ROOT}" \
          --repeat-index "${repeat}" \
          --seed "${seed}" \
          "${speed_args[@]}" \
          2>&1 | tee "${log_file}"
      done
    done
  done
done

echo "[batch] done"
