set -euo pipefail
cd /home/heapof/uni/thesis/itmo-vector-search
export OMP_NUM_THREADS=6
PY="conda run --no-capture-output -n vector-thesis python"
S=experiments/task1_synthetic
export QUANT_METHODS="rabitq-b1 rabitq-b2 rabitq-b3 rabitq-b4 rabitq-b5 rabitq-b6 rabitq-b7 rabitq-b8 sq4 pq-m96"
# name:fingerprint:tail:extra
for v in "diag-blur1.0:laion_ss100k_nc1000_c32:diag:--blur 1.0" \
         "nc10k-diag:laion_ss1M_nc10k_c32:diag:" \
         "diag-blur0.5:laion_ss100k_nc1000_c32:diag:--blur 0.5" \
         "nc10k:laion_ss1M_nc10k_c32:fp:" \
         "diag-m0.6:laion_ss100k_nc1000_c32:diag:--tail_mult 0.6" \
         "nc100:laion_ss10k_nc100_c32:fp:"; do
  IFS=: read -r n fp t extra <<< "$v"
  echo "=== s1m-$n"
  # shellcheck disable=SC2086
  [ -f data/s1m-$n/gt.ibin ] || $PY $S/cuvs_synth.py generate --fingerprint data/fp/$fp.npz --total_rows 1000000 --holdout 10000 --tail $t $extra --name s1m-$n --workers 6
  $S/run_dataset.sh s1m-$n 1000 "0"
  echo "DONE s1m-$n"
done
echo ITER4_DONE
set -euo pipefail
cd /home/heapof/uni/thesis/itmo-vector-search
export OMP_NUM_THREADS=6
PY="conda run --no-capture-output -n vector-thesis python"
S=experiments/task1_synthetic
until grep -q ITER4_DONE logs/iter4.log; do sleep 30; done
export QUANT_METHODS="rabitq-b1 rabitq-b2 rabitq-b3 rabitq-b4 rabitq-b5 rabitq-b6 rabitq-b7 rabitq-b8 sq4 pq-m96"
for v in "fp-blur1.0:fp:--blur 1.0" "diag-m0.5-blur1.0:diag:--tail_mult 0.5 --blur 1.0" "diag-m0.5-blur0.7:diag:--tail_mult 0.5 --blur 0.7"; do
  IFS=: read -r n t extra <<< "$v"
  echo "=== s1m-$n"
  # shellcheck disable=SC2086
  [ -f data/s1m-$n/gt.ibin ] || $PY $S/cuvs_synth.py generate --fingerprint data/fp/laion_ss100k_nc1000_c32.npz --total_rows 1000000 --holdout 10000 --tail $t $extra --name s1m-$n --workers 6
  $S/run_dataset.sh s1m-$n 1000 "0"
  echo "DONE s1m-$n"
done
echo ITER5_DONE
set -euo pipefail
cd /home/heapof/uni/thesis/itmo-vector-search
export OMP_NUM_THREADS=6
PY="conda run --no-capture-output -n vector-thesis python"
S=experiments/task1_synthetic
until grep -q ITER5_DONE logs/iter5.log; do sleep 30; done
export QUANT_METHODS="rabitq-b1 rabitq-b2 rabitq-b3 rabitq-b4 rabitq-b5 rabitq-b6 rabitq-b7 rabitq-b8 sq4 pq-m96"
for v in "nc10k-diag-blur0.5:diag:--blur 0.5" "nc10k-diag-m0.7-blur1.0:diag:--tail_mult 0.7 --blur 1.0"; do
  IFS=: read -r n t extra <<< "$v"
  echo "=== s1m-$n"
  # shellcheck disable=SC2086
  [ -f data/s1m-$n/gt.ibin ] || $PY $S/cuvs_synth.py generate --fingerprint data/fp/laion_ss1M_nc10k_c32.npz --total_rows 1000000 --holdout 10000 --tail $t $extra --name s1m-$n --workers 6
  $S/run_dataset.sh s1m-$n 1000 "0"
  echo "DONE s1m-$n"
done
echo ITER6_DONE
set -euo pipefail
cd /home/heapof/uni/thesis/itmo-vector-search
export OMP_NUM_THREADS=6
PY="conda run --no-capture-output -n vector-thesis python"
S=experiments/task1_synthetic
until grep -q ITER6_DONE logs/iter6.log; do sleep 30; done
export QUANT_METHODS="rabitq-b1 rabitq-b2 rabitq-b3 rabitq-b4 rabitq-b5 rabitq-b6 rabitq-b7 rabitq-b8 sq4 pq-m96"
echo "=== seeds s1m-fp-blur1.0"
for s in 1 2; do [ -f data/s1m-fp-blur1.0/partitions/bkm16_s$s.npz ] || $PY $S/partition.py s1m-fp-blur1.0 --N 16 --seed $s; done
$PY $S/rho_curves.py s1m-fp-blur1.0 --qsets heldout jitter --partition bkm16_s0 bkm16_s1 bkm16_s2
echo "DONE seeds"
for v in "fp-blur1.0-g1:laion_ss100k_nc1000_c32:fp:--blur 1.0 --seed 1" \
         "nc10k-fp-blur0.5:laion_ss1M_nc10k_c32:fp:--blur 0.5" \
         "nc10k-fp-blur1.0:laion_ss1M_nc10k_c32:fp:--blur 1.0"; do
  IFS=: read -r n fp t extra <<< "$v"
  echo "=== s1m-$n"
  # shellcheck disable=SC2086
  [ -f data/s1m-$n/gt.ibin ] || $PY $S/cuvs_synth.py generate --fingerprint data/fp/$fp.npz --total_rows 1000000 --holdout 10000 --tail $t $extra --name s1m-$n --workers 6
  $S/run_dataset.sh s1m-$n 1000 "0"
  echo "DONE s1m-$n"
done
echo ITER7_DONE
