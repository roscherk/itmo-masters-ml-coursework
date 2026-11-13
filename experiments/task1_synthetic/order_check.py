"""Похожи ли первые строки базы на случайные? cuVS берёт для отпечатка первые sample_size строк.

    python experiments/task1_synthetic/order_check.py laion10m --m 100000

Сравниваем первые m строк со случайными m: ‖Δ среднего‖, доли точек по 256 кластерам k-means,
обученного на третьей случайной выборке (полная вариация), и 1-NN C2ST между выборками.
Для контроля то же для двух независимых случайных выборок.
"""

from __future__ import annotations

import argparse

import numpy as np

from t1lib import gather, open_base, random_rows, set_threads


def compare(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> dict:
    import faiss
    from stats import knn_self
    km = faiss.Kmeans(A.shape[1], 256, niter=20, seed=1)
    km.train(C)
    ha = np.bincount(km.index.search(A, 1)[1][:, 0], minlength=256) / len(A)
    hb = np.bincount(km.index.search(B, 1)[1][:, 0], minlength=256) / len(B)
    m = 20_000
    X = np.concatenate([A[:m], B[:m]])
    lab = np.r_[np.zeros(m), np.ones(m)]
    _, I = knn_self(X, 1)
    return {"mean_diff": float(np.linalg.norm(A.mean(0) - B.mean(0))),
            "tv_256": float(0.5 * np.abs(ha - hb).sum()), "c2st_1nn": float((lab[I[:, 0]] == lab).mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--m", type=int, default=100_000)
    a = ap.parse_args()
    set_threads()
    base = open_base(a.name)
    n = base.shape[0]
    first = np.array(base[:a.m])
    idx = np.random.default_rng(5).permutation(np.arange(a.m, n))[: 3 * a.m]
    R1, R2, R3 = (gather(base, np.sort(idx[i * a.m:(i + 1) * a.m])) for i in range(3))
    print("первые против случайных:", compare(first, R1, R3))
    print("случайные против случайных:", compare(R2, R1, R3))


if __name__ == "__main__":
    main()
