"""Статистики «реальные против синтетики» для одного датасета: глобальные по базе и по запросам.

    python experiments/task1_synthetic/stats.py laion1m --qsets heldout jitter

Пишет results/task1/stats/<name>.json (глобальные и сводки по наборам запросов) и
<name>__<qset>.npz (значения по каждому запросу, для KS/W1 в compare.py).

Глобальные (выборка базы, одинакового размера на всех датасетах, поэтому сравнимы):
  norm_*            нормы векторов
  mean_norm         ‖среднего вектора‖ — анизотропия (у CLIP большая)
  mean_cos          средний косинус случайных пар
  pr, d50/d90/d95   спектр ковариации: participation ratio (Σλ)²/Σλ² и число компонент на 50/90/95% дисперсии
  twonn_id          глобальная внутренняя размерность, TwoNN (Facco et al., 2017)
  hub_skew10        скошенность распределения N₁₀ — числа вхождений точки в 10-NN других (Radovanović et al., 2010)
По запросам (из точного gt):
  d1, d10, d100     расстояния до 1-го, 10-го, 100-го соседа
  ratio101          d101/d100 (vsbench.metrics.distance_ratio), задача 2
  lid10, lid100     LID, MLE Левины–Бикеля по k соседям (Amsaleg et al., 2015; Aumüller, Ceccarello, 2019)
  rc10, rc100       relative contrast d_mean / d_k (He et al., 2012); лучший предиктор трудности
                    запроса по Ceccarello et al., KDD 2025; d_mean — по фиксированной выборке базы
  exp20_10, exp200_100  expansion d_2k / d_k
  loc_pr100, loc_d90    локальная размерность: спектр 100 ближайших соседей запроса (PR и число компонент на 90%).
                    У синтетики cuVS внутри кластера ранг ≈ ncomp, это видно здесь
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from t1lib import RESULTS, gather, load_qset, open_base, random_rows, set_threads
from vsbench.metrics import distance_ratio

N_GLOBAL = 100_000     # выборка базы для спектра, норм
N_TWONN = 20_000
N_HUB = 50_000
N_DMEAN = 10_000
N_LOCAL = 1_000        # запросов для локального спектра


def spectrum_stats(lam: np.ndarray) -> dict:
    lam = np.sort(np.maximum(lam, 0))[::-1]
    c = np.cumsum(lam) / lam.sum()
    return {"pr": float(lam.sum() ** 2 / (lam ** 2).sum()),
            "d50": int(np.searchsorted(c, 0.50) + 1), "d90": int(np.searchsorted(c, 0.90) + 1),
            "d95": int(np.searchsorted(c, 0.95) + 1)}


def knn_self(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """k ближайших в X для каждой точки X, без неё самой; евклидовы расстояния"""
    import faiss
    D, I = faiss.knn(X, X, k + 1)
    # самой точкой считаем столбец, где I == номер строки; если её вытеснил дубликат — первый столбец
    own = I == np.arange(len(X))[:, None]
    drop = np.where(own.any(1), own.argmax(1), 0)
    keep = np.ones_like(I, dtype=bool)
    keep[np.arange(len(X)), drop] = False
    I = I[keep].reshape(len(X), k)
    D = D[keep].reshape(len(X), k)
    return np.sqrt(np.maximum(D, 0)), I


def twonn(X: np.ndarray) -> float:
    D, _ = knn_self(X, 2)
    ok = D[:, 0] > 1e-6
    mu = D[ok, 1] / D[ok, 0]
    mu = np.sort(mu)[: int(0.9 * len(mu))]       # Facco et al.: отбрасываем 10% самых больших μ
    return float(len(mu) / np.log(mu).sum())


def hub_skew(X: np.ndarray, k: int = 10) -> float:
    _, I = knn_self(X, k)
    N = np.bincount(I.ravel(), minlength=len(X)).astype(np.float64)
    return float(((N - N.mean()) ** 3).mean() / N.std() ** 3)


def lid_mle(D: np.ndarray, k: int) -> np.ndarray:
    """MLE LID по первым k расстояниям; нулевые (дубликаты) не учитываем"""
    r = D[:, :k].astype(np.float64)
    rk = r[:, -1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        lg = np.log(r / rk)
    ok = r > 1e-6
    m = np.where(ok, lg, 0.0).sum(1) / np.maximum(ok.sum(1), 1)
    with np.errstate(divide="ignore"):
        return np.where(m < 0, -1.0 / m, np.inf)


def local_spectrum(base: np.ndarray, gt: np.ndarray, rows: np.ndarray, k: int = 100) -> tuple[np.ndarray, np.ndarray]:
    nb = gather(base, gt[rows, :k])               # (m, k, d)
    pr = np.empty(len(rows))
    d90 = np.empty(len(rows))
    for i in range(len(rows)):
        X = nb[i].astype(np.float64)
        X -= X.mean(0)
        s = np.linalg.svd(X, compute_uv=False) ** 2
        st = spectrum_stats(s)
        pr[i], d90[i] = st["pr"], st["d90"]
    return pr, d90


def global_stats(base: np.ndarray, seed: int = 0) -> dict:
    t0 = time.time()
    S = gather(base, random_rows(base.shape[0], N_GLOBAL, seed))
    norms = np.linalg.norm(S, axis=1)
    mu = S.mean(0)
    rng = np.random.default_rng(seed)
    a, b = rng.integers(0, len(S), 20_000), rng.integers(0, len(S), 20_000)
    Sn = S / np.maximum(norms[:, None], 1e-12)
    cos = (Sn[a] * Sn[b]).sum(1)[a != b]
    C = np.cov(S.astype(np.float64), rowvar=False)
    lam = np.linalg.eigvalsh(C)[::-1]
    out = {"n": int(base.shape[0]), "d": int(base.shape[1]),
           "norm_mean": float(norms.mean()), "norm_std": float(norms.std()),
           "norm_p01": float(np.quantile(norms, 0.01)), "norm_p99": float(np.quantile(norms, 0.99)),
           "mean_norm": float(np.linalg.norm(mu)), "mean_cos": float(cos.mean()),
           "total_var": float(lam.sum()), **spectrum_stats(lam),
           "twonn_id": twonn(S[:N_TWONN]), "hub_skew10": hub_skew(S[:N_HUB]),
           "eig": (lam / lam.sum()).astype(np.float32).tolist()}
    print(f"  глобальные: {time.time() - t0:.0f} с", flush=True)
    return out


def query_stats(base: np.ndarray, Q: np.ndarray, I: np.ndarray, D: np.ndarray, seed: int = 0) -> dict[str, np.ndarray]:
    S = gather(base, random_rows(base.shape[0], N_DMEAN, seed + 1)).astype(np.float64)
    q = Q.astype(np.float64)
    d2 = (q ** 2).sum(1)[:, None] - 2 * q @ S.T + (S ** 2).sum(1)[None, :]
    dmean = np.sqrt(np.maximum(d2, 0)).mean(1)
    rows = random_rows(len(Q), N_LOCAL, seed + 2)
    pr, d90 = local_spectrum(base, I, rows)
    lp = np.full(len(Q), np.nan)
    ld = np.full(len(Q), np.nan)
    lp[rows], ld[rows] = pr, d90
    return {**purity(base, I), "d1": D[:, 0], "d10": D[:, 9], "d100": D[:, 99], "ratio101": distance_ratio(D),
            "lid10": lid_mle(D, 10), "lid100": lid_mle(D, 100),
            "rc10": dmean / np.maximum(D[:, 9], 1e-12), "rc100": dmean / np.maximum(D[:, 99], 1e-12),
            "exp20_10": D[:, 19] / np.maximum(D[:, 9], 1e-12), "exp200_100": D[:, 199] / np.maximum(D[:, 99], 1e-12),
            "dup1": (D[:, 0] < 1e-6).astype(np.float64), "dmean": dmean,
            "loc_pr100": lp, "loc_d90": ld}


def purity(base: np.ndarray, I: np.ndarray, Ns=(64, 256, 1024), seed: int = 0, r: int = 100) -> dict[str, np.ndarray]:
    """Доля r соседей запроса в самом частом кластере k-means на N кластеров (обучен на выборке
    этой же базы): оракульный ρ(k = 1) без балансировки на нескольких масштабах. У синтетики из
    гауссовых кластеров соседи обычно лежат в одном кластере генератора — видно при N ~ числу кластеров"""
    import faiss
    S = gather(base, random_rows(base.shape[0], N_GLOBAL, seed + 3))
    ids, inv = np.unique(I[:, :r], return_inverse=True)
    V = gather(base, ids)
    out = {}
    for N in Ns:
        km = faiss.Kmeans(base.shape[1], N, niter=20, seed=seed, max_points_per_centroid=1 << 30)
        km.train(S)
        lab = km.index.search(V, 1)[1][:, 0][inv.reshape(I.shape[0], r)]
        out[f"pur{N}"] = np.array([np.bincount(row).max() / r for row in lab])
    return out


def summarize(v: dict[str, np.ndarray]) -> dict:
    out = {}
    for k, a in v.items():
        a = a[np.isfinite(a)]
        out[k] = {"mean": float(a.mean()), "median": float(np.median(a)),
                  "p10": float(np.quantile(a, 0.1)), "p90": float(np.quantile(a, 0.9))}
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name")
    ap.add_argument("--qsets", nargs="+", default=["heldout", "jitter"])
    ap.add_argument("--skip_global", action="store_true", help="взять глобальные из готового json")
    a = ap.parse_args(argv)
    set_threads()
    out_dir = RESULTS / "stats"
    out_dir.mkdir(parents=True, exist_ok=True)
    js = out_dir / f"{a.name}.json"
    res = json.loads(js.read_text()) if js.exists() else {}
    base = open_base(a.name)
    print(f"{a.name}: база {base.shape}")
    if not a.skip_global or "global" not in res:
        res["global"] = global_stats(base)
    res.setdefault("qsets", {})
    for qs in a.qsets:
        t0 = time.time()
        Q, I, D = load_qset(a.name, qs)
        v = query_stats(base, Q, I, D)
        np.savez(out_dir / f"{a.name}__{qs}.npz", **v)
        res["qsets"][qs] = {"nq": int(len(Q)), **summarize(v)}
        print(f"  {qs}: {len(Q)} запросов, {time.time() - t0:.0f} с; медианы "
              + ", ".join(f"{k}={res['qsets'][qs][k]['median']:.3g}" for k in
                          ("d1", "d100", "lid100", "rc100", "loc_d90", "pur256", "pur1024")), flush=True)
    js.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"→ {js}")


if __name__ == "__main__":
    main()
