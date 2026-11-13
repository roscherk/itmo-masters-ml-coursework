# E-means prototype benchmark

The `partition.py --method e-means` mode implements mini-batch EMA centroid
updates with the batch-size and step-size scaling laws from Faiss
`contrib/e_means/base.py`. It is a CPU prototype for this coursework repo,
not a cuVS GPU implementation or a verbatim copy of Faiss's implementation:
cluster revival and the moving-pass closing means are not included.

Run the 1M A/B benchmark with:

```bash
conda run --no-capture-output -n vector-thesis python \
  experiments/task1_synthetic/benchmark_emeans.py tg1m-cuvs \
  --train-size 100000 --clusters 16 --epochs 5 --threads 6
```

The runner uses the same deterministic training sample and seed for both
methods. It records fit and full-base assignment times, final mean squared
distance, and cluster sizes in `results/task1/emeans_1m.json`. Faiss Lloyd uses
20 iterations by default; E-means uses five passes by default. The benchmark
host has no CUDA/PyTorch, so these measurements are CPU-only.

## Result

Measured on `tg1m-cuvs` (1,000,000 x 768), seed 81027, 100,000 fit rows,
6 Faiss/OpenMP threads:

| Method | Fit | Full assignment | Mean squared distance |
|---|---:|---:|---:|
| Faiss Lloyd, 20 iterations | 0.83 s | 0.28 s | 0.528002 |
| E-means prototype, 5 passes | 2.48 s | 0.34 s | 0.526909 |

This is one seed and the iteration/pass budgets are not compute-matched. E-means
was slower on this CPU run; its slightly lower final objective is descriptive,
not evidence of a quality advantage at equal runtime. Lloyd cluster sizes ranged
from 28,683 to 104,610; E-means ranged from 29,450 to 111,218. The JSON file
contains the per-cluster sizes and timings. These results do not measure cuVS
GPU performance.
