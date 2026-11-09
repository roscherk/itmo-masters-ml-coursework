# Comparing Synthetic Data Generation Algorithms for Vector Search Benchmarking

**Course:** Machine Learning, ITMO, Autumn 2026  
**Topic:** Empirical comparison of three algorithms for generating synthetic high-dimensional datasets intended for benchmarking approximate nearest-neighbour (ANN) / vector search systems.

---

## Problem Statement

Evaluating ANN indexes (HNSW, IVF, graph-based methods) requires large high-dimensional datasets.
Real datasets such as LAION (CLIP ViT-L/14, 768-d) are expensive to collect and may carry privacy constraints.
Synthetic datasets offer a scalable, reproducible alternative — but only if they faithfully reproduce the statistical
properties that determine index behaviour: neighbourhood structure, intrinsic dimensionality, and quantisation
difficulty.

**Research question:** Which synthetic generation algorithm best reproduces the properties of a real embedding
dataset (LAION-10M, 768-d) that matter for benchmarking ANN indexes?

**Success criterion:** For each algorithm, we measure how much it shifts three target metrics relative to LAION:

| Metric | Why it matters |
|--------|---------------|
| Routing recall ρ(q) at k | Measures balanced-partitioning quality; systematic over- or under-estimation means benchmark scores don't transfer to real data |
| RaBitQ recall@100 at 4–8 bits | Minimum bits for 0.99 recall; predicts compression requirements in production |
| 1-NN C2ST distinguishability | Overall distributional similarity; a classifier score near 0.5 means indistinguishable |

A method "passes" if its max shift in ρ̄ is ≤ 0.02 and its per-bit recall shift is ≤ 0.01.

---

## Algorithms Compared

### Baseline — DisturbGen-style perturbation
Fit a flat k-means on a sample of real data; generate new points by sampling cluster centroids
(proportional to cluster size) and adding Gaussian noise scaled to each cluster's intra-cluster standard deviation.
No PCA subspace, no blur — a simple, interpretable baseline.

### Alternative — GMM with diagonal covariance (sklearn)
Fit a Gaussian Mixture Model with diagonal covariance on the same sample using scikit-learn's
`GaussianMixture(covariance_type="diag")`. Sample from the fitted mixture.
Richer than the baseline (per-component diagonal variance), but no explicit dimensionality reduction.

### Reference — cuVS fingerprint generator (+ blur)
The generator used in the ITMO vector-search thesis project.
Fits a PCA-compressed fingerprint (k-means on first 32 principal components) and generates points
in that subspace with optional blur (`--blur 1.0`) to soften cluster boundaries.
This is the "reference" method whose properties are already measured against LAION.

---

## Hypotheses

1. **H1 (baseline over-separates):** The simple perturbation baseline will over-estimate ρ(q) by more than 0.02 because
   its Gaussian clusters are too clean, mirroring the unblurred cuVS behaviour.
2. **H2 (GMM captures variance better):** Diagonal-covariance GMM will reproduce per-bit recall@100 more accurately
   than the baseline because per-dimension variance is a first-order predictor of quantisation difficulty.
3. **H3 (cuVS + blur is the best at ρ but worst at hubs):** cuVS + blur achieves the smallest ρ shift but introduces
   hub artefacts (high hub skewness N₁₀) that the other methods avoid.
4. **H4 (all three fail at 1–3 bits):** At 1–3 bits, all methods underestimate the difficulty of LAION-10M because
   none captures the continuous intrinsic dimensionality (~24) of real CLIP embeddings.

---

## Data

| Dataset | n × d | Source | License |
|---------|--------|--------|---------|
| LAION-10M (CLIP ViT-L/14) | ~10M × 768 | [LAION-5B](https://laion.ai/blog/laion-5b/) subset, Parquet | CC-BY 4.0 |

The real dataset is used only to fit generator parameters (a 100k-row sample) and as the ground truth for comparison.
Synthetic datasets are generated fresh for each experiment (1M and 10M vectors).

No raw LAION vectors are stored in this repository.

---

## Repository Structure

```
coursework/
├── data/                   # not committed — datasets (real sample + generated)
├── notebooks/
│   ├── 01_data_exploration.ipynb    # LAION sample statistics, baseline stats comparison
│   ├── 02_fit_generators.ipynb      # Fit all three generators on the 100k sample
│   ├── 03_generate_and_stats.ipynb  # Generate 1M + 10M datasets, compute statistics
│   ├── 04_routing_recall.ipynb      # ρ(q) curves: balanced k-means, oracle vs centroid router
│   └── 05_quantisation.ipynb        # RaBitQ recall@100, minimum bits, L99
├── src/
│   ├── perturbation.py     # DisturbGen-style perturbation generator
│   ├── gmm_generator.py    # Diagonal-covariance GMM generator
│   ├── metrics.py          # ρ(q), recall@k, C2ST, hub skewness, LID
│   └── utils.py            # fbin I/O, balanced k-means wrapper
├── results/
│   ├── perturbation/       # CSVs and plots for baseline
│   ├── gmm/                # CSVs and plots for GMM
│   └── cuvs/               # copied / symlinked from thesis results
├── slides/                 # Slidev presentation
│   ├── slides.md
│   └── package.json
├── requirements.txt
└── README.md
```

---

## How to Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Fit generators on the 100k LAION sample (requires data/laion_sample_100k.fbin)
jupyter nbconvert --to notebook --execute notebooks/02_fit_generators.ipynb

# 2. Generate 1M synthetic datasets
jupyter nbconvert --to notebook --execute notebooks/03_generate_and_stats.ipynb

# 3. Measure routing recall and quantisation
jupyter nbconvert --to notebook --execute notebooks/04_routing_recall.ipynb
jupyter nbconvert --to notebook --execute notebooks/05_quantisation.ipynb
```

Presentation:

```bash
cd slides && npm install && npm run dev
```

---

## Results Summary

> To be filled in after experiments.

---

## References

- Gottesbüren et al., *Scaling Graph-Based ANNS Algorithms to Billion-Size Datasets*, VLDB 2025 ([arXiv:2403.01797](https://arxiv.org/abs/2403.01797))
- Gao & Long, *RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search*, SIGMOD 2024 ([arXiv:2405.12497](https://arxiv.org/abs/2405.12497))
- NVIDIA cuVS Benchmarking Guide — [Synthetic Dataset Generation](https://docs.nvidia.com/cuvs/user-guide/benchmarking-guide/cu-vs-bench-tool/synthetic-dataset-generation)
- Lopez-Paz & Oquab, *Revisiting Classifier Two-Sample Tests*, ICLR 2017
- Facco et al., *Estimating the intrinsic dimension of datasets by a minimal neighborhood information*, Scientific Reports 2017
- Radovanović et al., *Hubs in Space: Popular Nearest Neighbors in High-Dimensional Data*, JMLR 2010
