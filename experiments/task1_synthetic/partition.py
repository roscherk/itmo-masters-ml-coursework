"""Balanced k-means на N шардов: каждый шард не больше (1 + ε)·n/N точек (gp-ann: ε = 0,05).

    python experiments/task1_synthetic/partition.py laion1m --N 16 --seed 0

Схема как в gp-ann (Gottesbüren et al., VLDB 2025, src/kmeans.cpp: KMeans, затем BalancedKMeans):
сначала обычный k-means, затем балансировка с ограничением на размер. Балансировку делаем
векторно: у каждого шарда своя «цена» λ_j, точка идёт в argmin_j (‖x − c_j‖² + λ_j); цену
переполненного шарда поднимаем ровно настолько, чтобы из него ушёл избыток (аукцион),
затем пересчитываем центроиды по итоговому разбиению; так несколько внешних раундов. В gp-ann
штраф общий, λ·|шард| (BKM+), итог тот же — разбиение с ограничением размера и малым объективом.
k-means обучается на случайной выборке (по умолчанию 1M), назначение — по всей базе кусками,
поэтому база может не помещаться в RAM.

Пишет data/<name>/partitions/bkm<N>_s<seed>.npz: labels (n,) int16, centroids (N, d) — средние
шардов (по ним роутит rho_curves.py), sizes, objective.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from utils import read_fbin


def ds_path(name: str):
    return Path(os.environ.get("VSBENCH_DATA", ROOT / "data")) / name


def open_base(name: str):
    return read_fbin(ds_path(name) / "base.fbin")


def set_threads():
    import faiss
    faiss.omp_set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "6")))


def random_rows(n: int, m: int, seed: int):
    return np.sort(np.random.default_rng(seed).choice(n, size=min(m, n), replace=False))


def gather(base: np.ndarray, idx: np.ndarray, chunk: int = 200_000):
    flat = np.asarray(idx).ravel()
    order = np.argsort(flat, kind="stable")
    out = np.empty((len(flat), base.shape[1]), dtype=np.float32)
    for start in range(0, len(flat), chunk):
        sel = order[start:start + chunk]
        out[sel] = base[flat[sel]]
    return out.reshape(*np.shape(idx), base.shape[1])


def iter_chunks(base: np.ndarray, chunk: int = 1_000_000):
    for start in range(0, base.shape[0], chunk):
        yield start, np.ascontiguousarray(base[start:start + chunk], dtype=np.float32)


def dists_to(base: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Квадраты расстояний всех точек базы до центроидов, (n, N) float32"""
    import faiss
    out = np.empty((base.shape[0], len(C)), dtype=np.float32)
    idx = faiss.IndexFlatL2(C.shape[1])
    idx.add(np.ascontiguousarray(C, dtype=np.float32))
    for s, X in iter_chunks(base):
        D, I = idx.search(X, len(C))
        np.put_along_axis(out[s:s + len(X)], I, D, axis=1)
    return out


def balanced_assign(D: np.ndarray, cap: int, max_iter: int = 200) -> np.ndarray:
    n, N = D.shape
    D = D.astype(np.float64)        # в float32 мелкие приращения цен теряются, цикл «пинг-понгует»
    lam = np.zeros(N, dtype=np.float64)
    for _ in range(max_iter):
        S = D + lam
        a = S.argmin(1)
        sizes = np.bincount(a, minlength=N)
        over = np.flatnonzero(sizes > cap)
        if len(over) == 0:
            return a
        if (sizes - cap)[over].sum() <= max(1, n // 1000):
            break                   # остаток избытка мал — его добьёт жадный проход ниже
        for j in over:
            pts = np.flatnonzero(a == j)
            Sj = S[pts]
            own = Sj[:, j].copy()
            Sj[:, j] = np.inf
            regret = Sj.min(1) - own               # на сколько дороже уйти во второй по выгоде шард
            excess = sizes[j] - cap
            lam[j] += float(np.partition(regret, excess - 1)[excess - 1]) * (1 + 1e-6) + 1e-12
    # не сошлось — дожимаем жадно: лишние точки (с наименьшим regret) в ближайший неполный шард
    S = D + lam
    a = S.argmin(1)
    sizes = np.bincount(a, minlength=N)
    for j in np.flatnonzero(sizes > cap):
        pts = np.flatnonzero(a == j)
        Sj = S[pts].copy()
        Sj[:, j] = np.inf
        order = np.argsort(Sj.min(1) - S[pts, j])        # сначала те, кому уйти дешевле всего
        for p in pts[order][: sizes[j] - cap]:
            free = np.flatnonzero(sizes < cap)
            t = free[np.argmin(D[p, free])]
            a[p] = t
            sizes[j] -= 1
            sizes[t] += 1
    return a


def shard_means(base: np.ndarray, labels: np.ndarray, N: int) -> np.ndarray:
    sums = np.zeros((N, base.shape[1]), dtype=np.float64)
    for s, X in iter_chunks(base):
        lab = labels[s:s + len(X)]
        for j in range(N):
            sums[j] += X[lab == j].sum(0, dtype=np.float64)
    return (sums / np.bincount(labels, minlength=N)[:, None]).astype(np.float32)


def balanced_kmeans(base: np.ndarray, N: int, eps: float, seed: int, n_train: int,
                    rounds: int, verbose: bool = True) -> dict:
    import faiss
    n, d = base.shape
    cap = int(n * (1 + eps) / N)
    t0 = time.time()
    X = gather(base, random_rows(n, n_train, seed))
    km = faiss.Kmeans(d, N, niter=20, seed=seed, max_points_per_centroid=1 << 30)
    km.train(X)
    C = km.centroids.copy()
    del X
    fit_seconds = time.time() - t0
    labels = None
    assignment_start = time.time()
    for r in range(rounds):
        D = dists_to(base, C)
        if r == 0:
            km_sizes = np.bincount(D.argmin(1), minlength=N)
        labels = balanced_assign(D, cap)
        obj = float(D[np.arange(n), labels].mean())
        C = shard_means(base, labels, N)
        if verbose:
            sz = np.bincount(labels, minlength=N)
            print(f"  раунд {r}: объектив {obj:.5g}, шарды {sz.min()}–{sz.max()} (cap {cap}), "
                  f"{time.time() - t0:.0f} с", flush=True)
        del D
    return {"labels": labels.astype(np.int16), "centroids": C,
            "sizes": np.bincount(labels, minlength=N), "kmeans_sizes": km_sizes,
            "objective": obj, "cap": cap, "eps": eps, "seed": seed,
            "training_seconds": fit_seconds,
            "assignment_seconds": time.time() - assignment_start}


def emean_kmeans(base: np.ndarray, N: int, seed: int, n_train: int,
                 epochs: float, batch_size: int | None, verbose: bool = True) -> dict:
    """Fit mini-batch E-means on a deterministic sample, then assign all rows."""
    import faiss
    import math

    n, d = base.shape
    t0 = time.time()
    X = gather(base, random_rows(n, n_train, seed))
    initial = X[np.random.default_rng(seed).choice(len(X), N, replace=False)].copy()
    if batch_size is None:
        raw = 0.035134 * N**0.134749 * len(X)**0.876473 * epochs**0.813344
        batch_size = min(2 ** round(math.log2(max(1, raw))), max(1, len(X) // 2))
    batch_size = min(max(1, int(batch_size)), len(X))
    alpha0 = min(1.0, max(1e-4, 0.8244 * N**-0.4617 * batch_size**0.9003
                          * len(X)**-0.4498 * epochs**-0.1057))
    idx = faiss.IndexFlatL2(d)
    C = initial
    mass = np.full(N, 2.0**-16, dtype=np.float32)
    fit_start = time.time()
    for epoch in range(math.ceil(epochs)):
        rows = len(X) if epoch < int(epochs) else int(round((epochs % 1) * len(X)))
        if rows == 0:
            continue
        order = np.random.default_rng(seed + epoch).permutation(len(X))[:rows]
        steps = math.ceil(rows / batch_size)
        for step, start in enumerate(range(0, rows, batch_size)):
            batch = np.ascontiguousarray(X[order[start:start + batch_size]])
            idx.reset()
            idx.add(C)
            D, I = idx.search(batch, 1)
            labels = I.ravel()
            counts = np.bincount(labels, minlength=N).astype(np.float32)
            sums = np.zeros_like(C)
            np.add.at(sums, labels, batch)
            fraction = (step + 1) / steps
            alpha = alpha0 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * fraction)))
            new_mass = (1 - alpha) * mass + alpha * counts
            updated = ((1 - alpha) * mass[:, None] * C + alpha * sums)
            nonzero = new_mass > 0
            C[nonzero] = updated[nonzero] / new_mass[nonzero, None]
            mass = new_mass
        if verbose:
            print(f"  E-means epoch {epoch + 1}/{epochs:g}: batch={batch_size}, "
                  f"alpha0={alpha0:.4g}, {time.time() - t0:.0f} s", flush=True)
    fit_seconds = time.time() - fit_start
    del X
    idx.reset()
    idx.add(np.ascontiguousarray(C, dtype=np.float32))
    labels = np.empty(n, dtype=np.int32)
    objective = 0.0
    assignment_start = time.time()
    for s, X in iter_chunks(base):
        D, I = idx.search(X, 1)
        labels[s:s + len(X)] = I.ravel()
        objective += float(D.sum(dtype=np.float64))
    sizes = np.bincount(labels, minlength=N)
    if verbose:
        print(f"  final assignment: objective {objective / n:.5g}, "
              f"clusters {sizes.min()}–{sizes.max()}, {time.time() - t0:.0f} s",
              flush=True)
    return {"labels": labels.astype(np.int16), "centroids": C,
            "sizes": sizes, "kmeans_sizes": sizes.copy(),
            "objective": objective / n, "seed": seed, "method": "e-means",
            "epochs": epochs, "batch_size": batch_size, "alpha0": alpha0,
            "elapsed_seconds": fit_seconds + time.time() - assignment_start,
            "training_seconds": fit_seconds,
            "full_assignment_seconds": time.time() - assignment_start}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name")
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_train", type=int, default=1_000_000)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--method", choices=["balanced", "e-means"], default="balanced")
    ap.add_argument("--epochs", type=float, default=5)
    ap.add_argument("--batch_size", type=int)
    a = ap.parse_args(argv)
    set_threads()
    base = open_base(a.name)
    print(f"{a.name}: balanced k-means N={a.N}, ε={a.eps}, seed={a.seed}")
    if a.method == "e-means":
        res = emean_kmeans(base, a.N, a.seed, a.n_train, a.epochs, a.batch_size)
        prefix = "emeans"
    else:
        res = balanced_kmeans(base, a.N, a.eps, a.seed, a.n_train, a.rounds)
        prefix = "bkm"
    out = ds_path(a.name) / "partitions" / f"{prefix}{a.N}_s{a.seed}.npz"
    out.parent.mkdir(exist_ok=True)
    np.savez(out, **res)
    print(f"→ {out}; clusters {res['sizes'].min()}–{res['sizes'].max()}, objective {res['objective']:.6g}")


if __name__ == "__main__":
    main()
