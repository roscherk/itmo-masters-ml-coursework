set -euo pipefail
cd /home/heapof/uni/thesis/itmo-vector-search
export OMP_NUM_THREADS=6
PY="conda run --no-capture-output -n vector-thesis python"
S=experiments/task1_synthetic
# Чекпоинт 04: 10 сидов balanced k-means, сиды генератора, SE recall. Склейка logs/iter14.sh и logs/iter15.sh

# сиды balanced k-means 0..$2-1 для датасета $1 и ρ(q) по всем
seeds() {
  local d=$1 parts=()
  for s in $(seq 0 $(($2 - 1))); do
    [ -f data/$d/partitions/bkm16_s$s.npz ] || $PY $S/partition.py $d --N 16 --seed $s
    parts+=("bkm16_s$s")
  done
  $PY $S/rho_curves.py $d --qsets heldout jitter --partition "${parts[@]}"
}

echo "=== 1M: 10 сидов k-means"
seeds laion1m 10
seeds s1m-fp-blur1.0 10

echo "=== 1M: сиды генератора 1–4, по 3 сида k-means"
for g in 1 2 3 4; do
  n=s1m-fp-blur1.0-g$g
  [ -f data/$n/base.fbin ] || $PY $S/cuvs_synth.py generate --fingerprint data/fp/laion_ss100k_nc1000_c32.npz \
      --total_rows 1000000 --holdout 10000 --tail fp --blur 1.0 --seed $g --name $n --workers 6
  if [ $g = 1 ]; then   # у g1 уже есть разбиение s0 по старой базе: база должна совпасть побитно
    (cd data/$n && echo "083b91b55cc4b34168852f9dd0b548c3  gt.ibin
4b8859ce07734c898dee4975b5014b83  query.fbin" | md5sum -c -) || { echo "g1 НЕ совпал"; exit 1; }
  fi
  $PY $S/make_views.py heldout $n
  [ -f data/$n/pools/jitter/gt.ibin ] || $PY scripts/make_query_pool.py build $n --pool jitter --n 10000 \
      --normalize --threads $OMP_NUM_THREADS --check 5
  seeds $n 3
  rm -f data/$n/base.fbin          # 3 ГБ; generate с тем же --seed воспроизводит её
done
$PY $S/rho_ci.py --real laion1m --synth s1m-fp-blur1.0 --gens s1m-fp-blur1.0-g1 s1m-fp-blur1.0-g2 \
    s1m-fp-blur1.0-g3 s1m-fp-blur1.0-g4

echo "=== 1M: SE recall по запросам, RaBitQ 4–8 бит"
for d in laion1m s1m-fp-blur1.0; do
  $PY $S/quant.py $d --qsets heldout jitter --nq 1000 --methods rabitq-b4 rabitq-b5 rabitq-b6 rabitq-b7 rabitq-b8
done
$PY -c "
import csv, sys
key = lambda r: (r['dataset'], r['qset'], r['method'])
old = {key(r): r for r in csv.DictReader(open('logs/quant_before_iter14.csv'))}
bad = [k for r in csv.DictReader(open('results/task1/quant.csv')) if r.get('recall100_se') and r['nq'] == old[(k := key(r))]['nq']
       and abs(float(r['recall100']) - float(old[k]['recall100'])) > 1e-9]
print('quant: средние совпали' if not bad else f'quant: средние ИЗМЕНИЛИСЬ {bad}')
"
echo DONE_1M

echo "=== 10M: 10 сидов k-means"
seeds laion10m 10
seeds s10m-fp-blur1.0 10
$PY $S/rho_ci.py --real laion10m --synth s10m-fp-blur1.0
echo DONE_10M_RHO

echo "=== 10M: SE recall при 7 битах, 5000 запросов"
for d in laion10m s10m-fp-blur1.0; do
  $PY $S/quant.py $d --qsets heldout jitter --nq 5000 --methods rabitq-b7
done
echo ITER14_DONE

echo "=== 10M: сиды генератора 1–2, по 3 сида k-means, только heldout"
for g in 1 2; do
  n=s10m-fp-blur1.0-g$g
  [ -f data/$n/base.fbin ] || $PY $S/cuvs_synth.py generate --fingerprint data/fp/laion_ss100k_nc1000_c32.npz \
      --total_rows 10000000 --holdout 10000 --tail fp --blur 1.0 --seed $g --name $n --workers 6
  $PY $S/make_views.py heldout $n
  for s in 0 1 2; do [ -f data/$n/partitions/bkm16_s$s.npz ] || $PY $S/partition.py $n --N 16 --seed $s; done
  $PY $S/rho_curves.py $n --qsets heldout --partition bkm16_s0 bkm16_s1 bkm16_s2
  (cd data/$n && md5sum gt.ibin query.fbin > ../../logs/$n.md5)
  rm -f data/$n/base.fbin          # 31 ГБ; generate с тем же --seed воспроизводит её, md5 в logs/
  echo "DONE $n"
done
$PY $S/rho_ci.py --real laion10m --synth s10m-fp-blur1.0 --gens s10m-fp-blur1.0-g1 s10m-fp-blur1.0-g2 \
    --qsets heldout --out rho_ci_laion10m_gens.md
echo ITER15_DONE
