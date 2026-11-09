"""Metrics for comparing real and synthetic datasets."""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


# ---------------------------------------------------------------------------
# Routing recall ρ(q)
# ---------------------------------------------------------------------------

def routing_recall_centroid(
    queries: np.ndarray,
    base: np.ndarray,
    gt_ids: np.ndarray,
    labels: np.ndarray,
    k_nn: int = 100,
) -> float:
    """Fraction of ground-truth neighbours that share the query's nearest centroid shard."""
    import faiss

    n_shards = int(labels.max()) + 1
    centroids = np.vstack([
        base[labels == s].mean(axis=0) for s in range(n_shards)
    ]).astype(np.float32)

    index = faiss.IndexFlatL2(centroids.shape[1])
    index.add(centroids)
    _, q_shards = index.search(queries.astype(np.float32), 1)
    q_shards = q_shards.ravel()

    recalls = []
    for i, q_shard in enumerate(q_shards):
        nn_ids = gt_ids[i, :k_nn]
        nn_shards = labels[nn_ids]
        recalls.append((nn_shards == q_shard).mean())
    return float(np.mean(recalls))


def oracle_routing_recall(
    gt_ids: np.ndarray,
    labels: np.ndarray,
    k_nn: int = 100,
) -> float:
    """Oracle: fraction of k_nn neighbours reachable if we always route to the best shard."""
    recalls = []
    for nn_ids in gt_ids[:, :k_nn]:
        nn_shards = labels[nn_ids]
        best = np.bincount(nn_shards).max()
        recalls.append(best / k_nn)
    return float(np.mean(recalls))


def rho_curve(
    queries: np.ndarray,
    base: np.ndarray,
    gt_ids: np.ndarray,
    labels: np.ndarray,
    k_values: list[int] | None = None,
) -> dict:
    """Compute ρ(q) for centroid router and oracle at each k in k_values."""
    if k_values is None:
        k_values = [1, 2, 4, 8, 16, 32, 64, 100]
    centroid_rho = []
    oracle_rho = []
    for k in k_values:
        centroid_rho.append(routing_recall_centroid(queries, base, gt_ids, labels, k_nn=k))
        oracle_rho.append(oracle_routing_recall(gt_ids, labels, k_nn=k))
    return {"k": k_values, "centroid": centroid_rho, "oracle": oracle_rho}


# ---------------------------------------------------------------------------
# 1-NN C2ST (classifier two-sample test)
# ---------------------------------------------------------------------------

def c2st_1nn(real: np.ndarray, synth: np.ndarray, n_sub: int = 5000, seed: int = 0) -> float:
    """Estimate 1-NN C2ST accuracy. Score near 0.5 means indistinguishable."""
    rng = np.random.default_rng(seed)
    n = min(n_sub, len(real), len(synth))
    idx_r = rng.choice(len(real), n, replace=False)
    idx_s = rng.choice(len(synth), n, replace=False)
    X = np.vstack([real[idx_r], synth[idx_s]])
    y = np.array([0] * n + [1] * n)

    nn = NearestNeighbors(n_neighbors=2, algorithm="ball_tree", n_jobs=-1)
    nn.fit(X)
    distances, indices = nn.kneighbors(X)
    # for each point, nearest neighbour (index 1, since index 0 is itself)
    nn_labels = y[indices[:, 1]]
    return float((nn_labels == y).mean())


# ---------------------------------------------------------------------------
# Hub skewness (N_k skewness)
# ---------------------------------------------------------------------------

def hub_skewness(vectors: np.ndarray, k: int = 10, n_sub: int = 10000, seed: int = 0) -> float:
    """Skewness of the N_k (k-occurrence) distribution over a random subset."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(vectors), min(n_sub, len(vectors)), replace=False)
    sub = vectors[idx].astype(np.float32)
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree", n_jobs=-1)
    nn.fit(sub)
    _, indices = nn.kneighbors(sub)
    counts = np.bincount(indices[:, 1:].ravel(), minlength=len(sub))
    from scipy.stats import skew
    return float(skew(counts))


# ---------------------------------------------------------------------------
# Intrinsic dimensionality (TwoNN estimator)
# ---------------------------------------------------------------------------

def twonn_lid(vectors: np.ndarray, n_sub: int = 20000, seed: int = 0) -> float:
    """Global intrinsic dimensionality estimate via the TwoNN method (Facco et al. 2017)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(vectors), min(n_sub, len(vectors)), replace=False)
    sub = vectors[idx].astype(np.float32)
    nn = NearestNeighbors(n_neighbors=3, algorithm="ball_tree", n_jobs=-1)
    nn.fit(sub)
    distances, _ = nn.kneighbors(sub)
    r1 = distances[:, 1]
    r2 = distances[:, 2]
    mu = r2 / np.maximum(r1, 1e-10)
    # MLE estimate
    return float(1.0 / np.mean(np.log(mu)))
