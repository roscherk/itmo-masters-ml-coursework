set -euo pipefail
cd /home/heapof/uni/thesis/itmo-vector-search
export OMP_NUM_THREADS=6
export QUANT_METHODS="rabitq-b1 rabitq-b2 rabitq-b3 rabitq-b4 rabitq-b5 rabitq-b6 rabitq-b7 rabitq-b8 sq4 pq-m96"
S=experiments/task1_synthetic
echo "=== laion10m"
$S/run_dataset.sh laion10m 500 "0"
echo "DONE laion10m"
echo ITER10M_DONE
set -euo pipefail
cd /home/heapof/uni/thesis/itmo-vector-search
export OMP_NUM_THREADS=6
PY="conda run --no-capture-output -n vector-thesis python"
S=experiments/task1_synthetic
until grep -q ITER10M_DONE logs/iter10m.log; do sleep 30; done
export QUANT_METHODS="rabitq-b1 rabitq-b2 rabitq-b3 rabitq-b4 rabitq-b5 rabitq-b6 rabitq-b7 rabitq-b8 sq4 pq-m96"
for v in "fp-blur1.0:fp:--blur 1.0" "fp:fp:"; do
  IFS=: read -r n t extra <<< "$v"
  echo "=== s10m-$n"
  # shellcheck disable=SC2086
  [ -f data/s10m-$n/gt.ibin ] || $PY $S/cuvs_synth.py generate --fingerprint data/fp/laion_ss100k_nc1000_c32.npz --total_rows 10000000 --holdout 10000 --tail $t $extra --name s10m-$n --workers 6
  $S/run_dataset.sh s10m-$n 500 "0"
  echo "DONE s10m-$n"
done
echo ITER8_DONE
set -euo pipefail
cd /home/heapof/uni/thesis/itmo-vector-search
export OMP_NUM_THREADS=6
PY="conda run --no-capture-output -n vector-thesis python"
S=experiments/task1_synthetic
export QUANT_METHODS="rabitq-b1 rabitq-b2 rabitq-b3 rabitq-b4 rabitq-b5 rabitq-b6 rabitq-b7 rabitq-b8 sq4 pq-m96"
echo "=== seeds 10M"
for d in laion10m s10m-fp-blur1.0; do
  for s in 1 2; do [ -f data/$d/partitions/bkm16_s$s.npz ] || $PY $S/partition.py $d --N 16 --seed $s; done
  $PY $S/rho_curves.py $d --qsets heldout jitter --partition bkm16_s0 bkm16_s1 bkm16_s2
done
echo "DONE seeds 10M"
rm -f data/s10m-fp/base.fbin
n=s10m-nc10k-fp-blur1.0
echo "=== $n"
[ -f data/$n/gt.ibin ] || $PY $S/cuvs_synth.py generate --fingerprint data/fp/laion_ss1M_nc10k_c32.npz --total_rows 10000000 --holdout 10000 --tail fp --blur 1.0 --name $n --workers 6
$S/run_dataset.sh $n 500 "0"
echo "DONE $n"
echo ITER9_DONE
set -euo pipefail
cd /home/heapof/uni/thesis/itmo-vector-search
export OMP_NUM_THREADS=6
PY="conda run --no-capture-output -n vector-thesis python"
S=experiments/task1_synthetic
export QUANT_METHODS="rabitq-b1 rabitq-b2 rabitq-b3 rabitq-b4 rabitq-b5 rabitq-b6 rabitq-b7 rabitq-b8 sq4 pq-m96"
n=s1m-nc100-fp-blur1.0
echo "=== $n"
[ -f data/$n/gt.ibin ] || $PY $S/cuvs_synth.py generate --fingerprint data/fp/laion_ss10k_nc100_c32.npz --total_rows 1000000 --holdout 10000 --tail fp --blur 1.0 --name $n --workers 6
$S/run_dataset.sh $n 1000 "0"
echo "DONE $n"
echo ITER9B_DONE
set -uo pipefail
cd /home/heapof/uni/thesis/itmo-vector-search
export OMP_NUM_THREADS=6
PY="conda run --no-capture-output -n vector-thesis python"
S=experiments/task1_synthetic
q() { $PY $S/quant.py $1 --qsets heldout --nq 5000 --methods rabitq-b6 rabitq-b7 > logs/q5k_$1.log 2>&1; echo "DONE $1"; }
q laion10m & q s10m-fp-blur1.0 & wait
q s10m-nc10k-fp-blur1.0
echo ITER11_DONE
set -euo pipefail
cd /home/heapof/uni/thesis/itmo-vector-search
export OMP_NUM_THREADS=6
PY="conda run --no-capture-output -n vector-thesis python"
S=experiments/task1_synthetic
d=s10m-nc10k-fp-blur1.0
for s in 1 2; do [ -f data/$d/partitions/bkm16_s$s.npz ] || $PY $S/partition.py $d --N 16 --seed $s; done
$PY $S/rho_curves.py $d --qsets heldout jitter --partition bkm16_s0 bkm16_s1 bkm16_s2
echo ITER12_DONE
