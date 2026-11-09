---
theme: default
title: "Synthetic Data Generation for Vector Search Benchmarking"
info: |
  Comparing DisturbGen-style perturbation, diagonal-GMM, and cuVS fingerprint generation
  against LAION-10M (CLIP ViT-L/14, 768-d).
highlighter: shiki
drawings:
  persist: false
transition: slide-left
mdc: true
---

# Synthetic Data Generation for Vector Search Benchmarking

Comparing three algorithms against LAION-10M

<div class="pt-12 text-gray-400">
  Machine Learning · ITMO · Autumn 2026
</div>

---
layout: two-cols
---

# Why Synthetic Data?

Real embedding datasets are **expensive and constrained**

- LAION-10M: ~31 GB, requires download pipeline
- Privacy constraints on proprietary embeddings
- Scaling experiments need controllable dataset size

::right::

## What Makes a Good Synthetic Dataset?

A synthetic dataset is useful for benchmarking if it reproduces properties that **determine index behaviour**:

1. **Neighbourhood structure** — routing recall ρ(q)
2. **Quantisation difficulty** — bits needed for recall ≥ 0.99
3. **Overall distribution** — C2ST distinguishability

---

# Problem Statement

> Which synthetic generation algorithm best reproduces LAION-10M properties relevant to benchmarking ANN indexes?

**Target dataset:** LAION-10M, CLIP ViT-L/14 embeddings, d = 768

**Success criterion (defined before experiments):**

| Criterion | Threshold |
|-----------|-----------|
| Max shift in routing recall ρ̄ | ≤ 0.02 |
| Per-bit recall shift \|Δ recall@100\| | ≤ 0.01 |
| Minimum bits for recall 0.99 | exact match |

---
layout: three-cols
---

# Three Algorithms

::left::

### Baseline
**DisturbGen-style Perturbation**

Fit flat k-means on 100k sample.
Generate: sample centroid ∝ cluster size,
add Gaussian noise scaled to per-cluster std.

*Simple, interpretable.*

::middle::

### Alternative
**Diagonal-covariance GMM**

Fit `sklearn.GaussianMixture(covariance_type="diag")`
on 100k sample. Sample from the mixture.

*Captures per-dimension variance.*

::right::

### Reference
**cuVS Fingerprint + blur**

Fit PCA-compressed k-means fingerprint.
Generate in 32-d subspace, blur cluster
boundaries with τ = 1.0.

*Best understood against LAION (prior work).*

---

# Hypotheses

**H1 (over-separation):**  
Perturbation baseline over-estimates ρ(q) by > 0.02 — clean Gaussian clusters mirror unblurred cuVS behaviour.

**H2 (GMM captures variance):**  
Diagonal GMM reproduces per-bit recall@100 more accurately than the baseline because per-dimension variance is a first-order predictor of quantisation difficulty.

**H3 (cuVS best ρ, worst hubs):**  
cuVS + blur achieves the smallest ρ shift but introduces hub artefacts (high N₁₀ skewness) that the other methods avoid.

**H4 (all fail at low bits):**  
At 1–3 bits, all methods underestimate LAION difficulty because none captures its continuous intrinsic dimensionality (~24).

---

# Data

<div class="grid grid-cols-2 gap-4">
<div>

**Real data (ground truth)**

| Property | Value |
|----------|-------|
| Dataset | LAION-10M |
| Embedding model | CLIP ViT-L/14 |
| Dimensionality | 768 |
| Scale | ~10M vectors |
| Intrinsic dim (LID₁₀₀) | 24.3 |
| Hub skewness N₁₀ | 3.5 |

</div>
<div>

**Generator inputs**

- 100k-row sample for fitting (same sample for all three)
- Synthetic output: 1M and 10M vectors
- Ground truth computed with Faiss brute-force (k = 200)
- Two query sets: *heldout* (points not in base) and *jitter* (base point + noise)

</div>
</div>

---

# Evaluation: Routing Recall ρ(q)

ρ(q) measures what fraction of a query's k nearest neighbours land in the shard routed to.

**Setup:** balanced k-means, N = 16 shards, r = 100 neighbours

<div class="text-center my-6 text-gray-400 italic">
  [ ρ(q) curves — centroid router and oracle — will appear here ]
  <br/>
  x-axis: k (1 → 100), y-axis: ρ̄, one line per method
</div>

A systematic shift > 0.02 means benchmark routing scores don't transfer to real data.

---

# Evaluation: Quantisation (RaBitQ)

Minimum bits for recall@100 ≥ 0.99 in a flat protocol with top-L re-ranking.

<div class="text-center my-6 text-gray-400 italic">
  [ Recall@100 vs bits curves — 1–8 bits — will appear here ]
  <br/>
  One line per method + LAION ground truth
</div>

**Key numbers from prior work (cuVS + blur vs LAION-10M):**

| Bits | LAION recall | cuVS+blur recall | Δ |
|------|-------------|-----------------|---|
| 4 | — | — | ≤ 0.005 |
| 7 | 0.990 | 0.991 | 0.001 |
| 1 | 0.719 | 0.754 | 0.035 |

Baseline and GMM results: TBD.

---

# Statistical Properties

<div class="text-center my-6 text-gray-400 italic">
  [ Table: LAION vs all three generators — LID, loc_d90, hub skewness, C2ST ]
</div>

**Reference numbers (LAION-10M vs cuVS variants):**

| | LAION 10M | cuVS 10M | cuVS+blur 10M |
|--|--|--|--|
| C2ST | — | 0.887 | 0.865 |
| LID₁₀₀ | 24.3 | 20.4 | 21.9 |
| loc_d90 | 51 | 23 | 25 |
| hub skewness | 3.5 | 2.3 | 23.2 |

---

# Results Summary

<div class="text-center my-8 text-gray-400 italic">
  [ Summary table — all three methods × all metrics — will appear here ]
</div>

**Expected outcome:**  
- cuVS + blur meets both ρ and quantisation thresholds at 4+ bits  
- Perturbation baseline likely fails ρ threshold (H1)  
- GMM likely improves quantisation accuracy over baseline (H2)  
- All three likely fail at 1–3 bits (H4)

---

# Conclusions

> **Which method should you use to generate synthetic data for ANN benchmarks?**

<div class="mt-6">

- **DisturbGen perturbation** is the simplest but its over-separated clusters likely bias routing metrics
- **Diagonal GMM** better captures per-dimension variance and is likely more accurate for quantisation
- **cuVS + blur** is the most carefully validated option: meets ρ and quantisation thresholds at ≥ 4 bits, but introduces hub artefacts
- None of the three reliably reproduces LAION at 1–3 bits

</div>

---

# Limitations & Future Work

**Limitations**
- Results are LAION/CLIP-specific; other embedding models may differ
- 1M and 10M scales only; behaviour may change at 100M+
- GMM fitting on 100k sample with 1000 components is underdetermined

**Future directions**
- Flow-based generators (normalising flows) for better distributional fidelity
- Conditional generation matching query difficulty distribution
- Validation on graph indexes (HNSW, NSG), not just flat protocols

---
layout: center
---

# Thank you

**Repository:** `coursework/`

**References**

- Gottesbüren et al., *Scaling Graph-Based ANNS*, VLDB 2025 · arXiv:2403.01797
- Gao & Long, *RaBitQ*, SIGMOD 2024 · arXiv:2405.12497
- NVIDIA cuVS Benchmarking Guide
- Lopez-Paz & Oquab, *Classifier Two-Sample Tests*, ICLR 2017
- Facco et al., *TwoNN intrinsic dimensionality*, Scientific Reports 2017
- Radovanović et al., *Hubs in Space*, JMLR 2010
