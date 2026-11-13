#!/usr/bin/env bash
# Все измерения задачи 1 для одного датасета data/<name>/ (у которого уже есть base и корневые
# отложенные запросы): пул jitter, статистики, balanced k-means, ρ(q), квантизация.
#   experiments/task1_synthetic/run_dataset.sh <name> [nq_quant] [seeds]
# Переменные: SKIP_QUANT=1 — без квантизации; QUANT_METHODS="rabitq-b1 ..." — подмножество методов.
set -euo pipefail
name=$1
nq_quant=${2:-1000}
seeds=${3:-0}
S=$(dirname "$0")
PY="conda run --no-capture-output -n vector-thesis python"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6}

$PY $S/make_views.py heldout "$name"
if [ ! -f "data/$name/pools/jitter/gt.ibin" ]; then
  $PY scripts/make_query_pool.py build "$name" --pool jitter --n 10000 --normalize --threads "$OMP_NUM_THREADS" --check 5
fi
$PY $S/stats.py "$name" --qsets heldout jitter
parts=()
for s in $seeds; do
  [ -f "data/$name/partitions/bkm16_s$s.npz" ] || $PY $S/partition.py "$name" --N 16 --seed "$s"
  parts+=("bkm16_s$s")
done
$PY $S/rho_curves.py "$name" --qsets heldout jitter --partition "${parts[@]}"
if [ -z "${SKIP_QUANT:-}" ]; then
  # shellcheck disable=SC2086
  $PY $S/quant.py "$name" --qsets heldout jitter --nq "$nq_quant" ${QUANT_METHODS:+--methods $QUANT_METHODS}
fi
