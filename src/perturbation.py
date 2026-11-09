"""DisturbGen-style perturbation generator.

Fits a flat k-means on a sample; generates new points by sampling a cluster
centroid and adding Gaussian noise scaled to that cluster's intra-cluster std.
"""

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PerturbationGenerator:
    n_clusters: int = 1000
    seed: int = 42
    n_iter: int = 25

    # fitted attributes
    centroids_: np.ndarray | None = field(default=None, repr=False)
    stds_: np.ndarray | None = field(default=None, repr=False)
    weights_: np.ndarray | None = field(default=None, repr=False)

    def fit(self, X: np.ndarray) -> "PerturbationGenerator":
        import faiss

        X = X.astype(np.float32)
        d = X.shape[1]
        kmeans = faiss.Kmeans(d, self.n_clusters, niter=self.n_iter, seed=self.seed, verbose=False)
        kmeans.train(X)
        self.centroids_ = kmeans.centroids.copy()

        _, labels = kmeans.index.search(X, 1)
        labels = labels.ravel()

        counts = np.bincount(labels, minlength=self.n_clusters)
        self.weights_ = counts / counts.sum()

        stds = []
        for c in range(self.n_clusters):
            mask = labels == c
            if mask.sum() < 2:
                stds.append(np.ones(d, dtype=np.float32) * 1e-4)
            else:
                stds.append(X[mask].std(axis=0).astype(np.float32) + 1e-8)
        self.stds_ = np.vstack(stds)
        return self

    def generate(self, n: int, seed: int | None = None) -> np.ndarray:
        if self.centroids_ is None:
            raise RuntimeError("Call fit() first.")
        rng = np.random.default_rng(seed if seed is not None else self.seed + 1)
        cluster_ids = rng.choice(self.n_clusters, size=n, p=self.weights_)
        noise = rng.standard_normal((n, self.centroids_.shape[1])).astype(np.float32)
        points = self.centroids_[cluster_ids] + noise * self.stds_[cluster_ids]
        return points

    def save(self, path: str | Path) -> None:
        np.savez(
            str(path),
            centroids=self.centroids_,
            stds=self.stds_,
            weights=self.weights_,
            n_clusters=np.array([self.n_clusters]),
        )

    @classmethod
    def load(cls, path: str | Path) -> "PerturbationGenerator":
        data = np.load(str(path))
        obj = cls(n_clusters=int(data["n_clusters"][0]))
        obj.centroids_ = data["centroids"]
        obj.stds_ = data["stds"]
        obj.weights_ = data["weights"]
        return obj
