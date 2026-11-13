# Three-generator comparison (1M pilot)

## Scope and protocol

This is an isolated pilot. It uses a single random 100k-row fit sample from `laion1m`, saved as `fit_rows.npy` with seed `20261115`. Output is 1,000,000 float32 vectors in 768 dimensions, L2-normalized to unit norm. All generators use 1,000 clusters/components and seed `20261115`; CPU threads: 6. The common comparison queries are 1,000 LAION-1M heldout queries selected with seed `81027`, with exact brute-force `k=200` ground truth recomputed against each synthetic base.

The perturbation method is the coursework implementation: Faiss k-means, cluster sampling weighted by empirical cluster frequency, coordinate-wise Gaussian noise scaled by within-cluster standard deviation, then L2 normalization. **This is not the DisturbGen algorithm from VecBench.** The paper's DisturbGen perturbs vector records while keeping associated scalar attributes fixed; the paper does not specify an embedding-only generator for this experiment. The comparison therefore treats the existing coursework method as a perturbation baseline, not as a faithful reproduction of VecBench DisturbGen.

Diagonal GMM uses sklearn `GaussianMixture(covariance_type="diag", n_components=1000, max_iter=100, n_init=1, random_state=20261115, reg_covar=1e-6)` and L2-normalizes generated vectors. cuVS uses the CPU port in `experiments/task1_synthetic/cuvs_synth.py`, fitted on the same sample with 32 PCA components and 25 Faiss k-means iterations, default cuVS tail, no blur, and unit normalization.

For each method, a separate 1,000-query heldout set sampled from its fitted generator is also stored at the dataset root for per-generator diagnostics. Do not confuse it with the common `pools/laion-heldout` set used for cross-method routing and quantization.

## Pilot results

### Routing

Balanced k-means uses 16 shards, `r=100`, training sample 100k, seed `81027`, four balance rounds. One seed only; these values are exploratory and do not establish the planned 10-seed confidence intervals.

| Method | Centroid rho, k=1 / 2 / 4 / 8 | Oracle rho, k=1 / 2 / 4 / 8 |
|---|---|---|
| Perturbation baseline | 0.7041 / 0.8575 / 0.9384 / 0.9908 | 0.8642 / 0.9644 / 0.9956 / 0.9999 |
| Diagonal GMM | 0.7093 / 0.8557 / 0.9440 / 0.9894 | 0.8704 / 0.9650 / 0.9953 / 0.9999 |
| cuVS | 0.7001 / 0.8592 / 0.9450 / 0.9914 | 0.7786 / 0.9206 / 0.9858 / 0.9996 |

![Routing recall curves](rho_curves.png)

### RaBitQ

Recall is measured from codes, before reranking. `SE` is across the 1,000 common queries. `L99` is the smallest tested rerank depth reaching mean recall 0.99; it is `>` when no tested depth up to 2,000 reaches the threshold.

| Method | 1 bit | 3 bits | 4 bits | 5 bits | 6 bits | 7 bits | 8 bits | min bits >= .95 / .99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Perturbation baseline | 0.3819 ± .0032 | 0.7337 ± .0026 | 0.8325 ± .0019 | 0.9105 ± .0011 | 0.9547 ± .0006 | 0.9762 ± .0005 | 0.9871 ± .0004 | 6 / >8 |
| Diagonal GMM | 0.3970 ± .0035 | 0.7369 ± .0029 | 0.8329 ± .0021 | 0.9116 ± .0011 | 0.9556 ± .0007 | 0.9771 ± .0005 | 0.9877 ± .0004 | 6 / >8 |
| cuVS | 0.6710 ± .0037 | 0.8734 ± .0023 | 0.9207 ± .0016 | 0.9597 ± .0007 | 0.9798 ± .0004 | 0.9892 ± .0003 | 0.9941 ± .0002 | 5 / 8 |

![RaBitQ recall by bits](rabitq_recall.png)

The two perturbation-style methods are very close per bit (maximum difference 0.0152 at 1 bit); cuVS is substantially easier at 1-3 bits. At 7 bits cuVS is near, but below, 0.99 on this 1,000-query sample; at 8 bits all three reach different levels. Threshold conclusions for a 0.99 target need larger query samples and confidence bounds.

### Diagnostic metrics

Global metrics are computed from the generator base; query-level values use the **common LAION-heldout** set. Hub skewness follows the current `stats.py` sampling procedure and should be interpreted cautiously for cluster-ordered cuVS output.

| Metric | Perturbation baseline | Diagonal GMM | cuVS |
|---|---:|---:|---:|
| mean vector norm | 0.625 | 0.624 | 0.627 |
| mean pair cosine | 0.389 | 0.389 | 0.393 |
| covariance PR | 203.2 | 205.5 | 131.2 |
| covariance d50 / d90 / d95 | 169 / 626 / 694 | 168 / 625 / 694 | 72 / 401 / 485 |
| TwoNN ID | 117.9 | 117.9 | 25.6 |
| hub skew N10 | 5.05 | 15.38 | 2.67 |
| LID100 median | 99.4 | 96.7 | 36.5 |
| RC100 median | 1.28 | 1.29 | 1.40 |
| local d90 median | 80 | 80 | 36 |
| exact duplicate-neighbor share | 0 | 0 | 0 |
| purity 256 / 1024 median | .96 / .83 | .97 / .905 | .66 / .50 |

## Artifacts

- `protocol.json`: shared seeds and protocol notes.
- `fit_rows.npy`: exact common fit sample row ids.
- `rho.csv`, `routing_summary.csv`: routing curves and summary.
- `quant.csv`, `quant_summary.csv`: all tested methods and per-query SE.
- `stats/`: full global and query-level metric outputs.
- `stats_summary.csv`: compact comparison table.
- `data/tg1m-*-heldout-v2/`: bases, own heldout query sets, common LAION-heldout GT, partitions and per-query quant recalls.

## Limitations / next run

- The exact VecBench DisturbGen algorithm is not applicable to vector-only LAION data as specified in that paper; this is explicitly a coursework perturbation baseline.
- Only 1M was run. The 10M pass is blocked for now by available disk capacity: 90 GiB remained after the 1M artifacts; one additional 10M float32 base alone needs about 30.7 GB, before GT, quantization, and work files.
- Routing has one partition seed, not the planned ten. No seed-based confidence interval is reported.
- Quant uses 1,000 queries. This is adequate for a pilot but not for decisions close to 0.99; repeat with at least 5,000 common queries.
- H1 compares the perturbation baseline with LAION routing and needs the real LAION reference routing curves on this exact query set and multiple partition seeds before a threshold verdict.
- The generated query GT is exact brute force (`k=200`); quality was checked on sampled queries against Faiss flat search.

Reproduce data generation with:

```bash
conda run --no-capture-output -n itmo-ml python experiments/task1_synthetic/three_generators.py \
  --n 1000000 --fit-size 100000 --queries 1000 --seed 20261115 --clusters 1000 --threads 6
```

The shared-query, metrics and plot scripts are `prepare_three_generator_queries.py`, the existing `stats.py`, `partition.py`, `rho_curves.py`, `quant.py`, and `report_three_generators.py`. Use `T1_RESULTS_ROOT=results/task1/three_generators` for all metric scripts to keep their CSV outputs isolated.
