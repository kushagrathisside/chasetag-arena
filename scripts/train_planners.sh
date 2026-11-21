#!/usr/bin/env bash
# Train the four planners: one for each format (20 s, 60 s, 180 s) and one for all three.
#
# Each planner is cloned from the bot-episode dataset first (chasetag_arena.bc), then trained
# with PPO from that warm start (chasetag_arena.train --init). The runs go one after another,
# because each one already uses every CPU core. A run that was interrupted resumes from its
# checkpoint when this script is started again.
#
#   scripts/train_planners.sh DATA_DIR RUNS_DIR
set -euo pipefail

DATA=${1:?the dataset folder, holding the limitNNNs__*.zip files}
RUNS=${2:?a folder for the runs}
ITERS=${ITERS:-400}
PY=(uv run --no-sync python -m)
mkdir -p "$RUNS"

need_zips() {
  local limit=$1 count
  count=$(find "$DATA" -maxdepth 1 -name "limit$(printf %03d "$limit")s__*.zip" | wc -l)
  if [ "$count" -lt 24 ]; then
    echo "only $count of 24 zips at ${limit}s in $DATA; finish the dataset first" >&2
    exit 1
  fi
}

# name, round lengths, games per zip to clone, rounds between evaluations
run_one() {
  local name=$1 limits=$2 per_zip=$3 eval_every=$4 dir="$RUNS/$1"
  mkdir -p "$dir"
  if [ ! -f "$dir/bc.npz" ]; then
    echo "== $name: cloning from $DATA at $limits"
    "${PY[@]}" chasetag_arena.bc --data "$DATA" --limits "$limits" \
      --games-per-zip "$per_zip" --out "$dir/bc.npz" 2>&1 | tee "$dir/bc.log"
  fi
  local resume=()
  if [ -f "$dir/checkpoint.pkl" ]; then resume=(--resume); fi
  echo "== $name: PPO from the warm start, $ITERS rounds"
  "${PY[@]}" chasetag_arena.train --run-dir "$dir" --name "$name" --limits "$limits" \
    --iters "$ITERS" --eval-every "$eval_every" --init "$dir/bc.npz" \
    --critic-warmup 10 --init-log-std -1.0 "${resume[@]}" 2>&1 | tee -a "$dir/train.log"
}

for limit in 20 60 180; do need_zips "$limit"; done
run_one planner_k20 20 1000 25
run_one planner_k60 60 500 25
run_one planner_k180 180 250 50
run_one planner_k_all 20,60,180 300 50
echo "== done: $RUNS/planner_k*/planner.npz"
