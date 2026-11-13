set -euo pipefail
cd /home/heapof/uni/thesis/itmo-vector-search
export OMP_NUM_THREADS=6
PY="conda run --no-capture-output -n vector-thesis python"
S=experiments/task1_synthetic
FP=data/fp/laion_ss100k_nc1000_c32.npz
[ -f data/laion1m/gt.ibin ] || $PY $S/make_views.py subset laion10m laion1m --n 1000000
$S/run_dataset.sh laion1m 1000 "0 1 2"
for t in fp diag; do
  n=s1m-$t
  [ -f data/$n/gt.ibin ] || $PY $S/cuvs_synth.py generate --fingerprint $FP --total_rows 1000000 --holdout 10000 --tail $t --name $n --workers 6
  $S/run_dataset.sh $n 1000 "0"
done
echo ITER1_DONE
