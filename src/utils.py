"""Shared I/O and utility functions."""

import struct
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# fbin / ibin format (big-ann-benchmarks)
# header: two uint32 (n, d), then n*d values row-major
# ---------------------------------------------------------------------------

def read_fbin(path: str | Path) -> np.ndarray:
    path = Path(path)
    with open(path, "rb") as f:
        n, d = struct.unpack("<II", f.read(8))
        data = np.frombuffer(f.read(n * d * 4), dtype=np.float32).reshape(n, d)
    return data


def write_fbin(path: str | Path, data: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n, d = data.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<II", n, d))
        f.write(data.astype(np.float32).tobytes())


def read_ibin(path: str | Path) -> np.ndarray:
    path = Path(path)
    with open(path, "rb") as f:
        n, d = struct.unpack("<II", f.read(8))
        data = np.frombuffer(f.read(n * d * 4), dtype=np.int32).reshape(n, d)
    return data


def write_ibin(path: str | Path, data: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n, d = data.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<II", n, d))
        f.write(data.astype(np.int32).tobytes())


# ---------------------------------------------------------------------------
# Balanced k-means (thin wrapper around faiss)
# ---------------------------------------------------------------------------

def balanced_kmeans(
    vectors: np.ndarray,
    n_shards: int = 16,
    seed: int = 0,
    n_iter: int = 25,
) -> np.ndarray:
    """Return per-vector shard assignment from balanced k-means.

    Uses faiss for the centroid fit, then assigns each vector to its nearest
    centroid.  The result is not perfectly balanced but approximates it.
    """
    import faiss

    d = vectors.shape[1]
    kmeans = faiss.Kmeans(d, n_shards, niter=n_iter, seed=seed, verbose=False)
    kmeans.train(vectors.astype(np.float32))
    _, labels = kmeans.index.search(vectors.astype(np.float32), 1)
    return labels.ravel()
