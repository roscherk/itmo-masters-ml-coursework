"""Diagonal-covariance GMM generator using scikit-learn.

Fits GaussianMixture(covariance_type="diag") on a sample; generates new
points by sampling from the fitted mixture.
"""

import numpy as np
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GMMGenerator:
    n_components: int = 1000
    seed: int = 42
    max_iter: int = 100
    n_init: int = 1

    # fitted attributes
    gmm_: object | None = None

    def fit(self, X: np.ndarray) -> "GMMGenerator":
        from sklearn.mixture import GaussianMixture

        self.gmm_ = GaussianMixture(
            n_components=self.n_components,
            covariance_type="diag",
            max_iter=self.max_iter,
            n_init=self.n_init,
            random_state=self.seed,
            verbose=0,
        )
        self.gmm_.fit(X.astype(np.float64))
        return self

    def generate(self, n: int, seed: int | None = None) -> np.ndarray:
        if self.gmm_ is None:
            raise RuntimeError("Call fit() first.")
        rng_seed = seed if seed is not None else self.seed + 1
        samples, _ = self.gmm_.sample(n)
        return samples.astype(np.float32)

    def save(self, path: str | Path) -> None:
        import pickle

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.gmm_, f)

    @classmethod
    def load(cls, path: str | Path) -> "GMMGenerator":
        import pickle

        with open(path, "rb") as f:
            gmm = pickle.load(f)
        obj = cls(n_components=gmm.n_components)
        obj.gmm_ = gmm
        return obj
