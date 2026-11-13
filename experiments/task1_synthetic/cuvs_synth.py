# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Изменённый порт cuvs_bench.synthesize_dataset (github.com/rapidsai/cuvs,
# python/cuvs_bench/cuvs_bench/synthesize_dataset, коммит 2450691e6c65) на CPU:
# cupy → numpy, cuvs kmeans → faiss.Kmeans, cuvs PCA → тонкий SVD, cuvs brute_force → faiss.
"""Синтетика cuVS на CPU: fit → fingerprint.npz → generate → data/<name>/.

Алгоритм (как в cuVS):
1. fit: k-means на выборке реальных данных; для каждого кластера — PCA ранга ncomp по остаткам
   (точка − центроид), дисперсия по каждой координате, доля точек (плотность), сетка из 256
   квантилей норм. Если нормы ≈ 1 (|mean−1| ≤ 0,02 и CV ≤ 0,05), вместо сеток ставится флаг norm_unit.
2. generate: кластер c порождается как centroid_c + z @ components_c + шум,
   z ~ N(0, diag(explained_var_c)), шум ~ N(0, noise_var_c·I); затем по каждой координате
   масштаб sqrt(target_var / max(actual_var, noise_var)), не больше 5; затем нормы: L2-нормировка
   (unit) или целевая норма из обратной ФР кластера (percentile).
   Каждый кластер порождается целиком своим генератором с сидом seed·10⁶ + c, поэтому его можно
   породить заново для запросов и ground truth.
3. Запросы: равномерно выбранные точки базы + гауссов шум с σ = 0,1·std(всех координат запросов).

Особенности cuVS, которые сохранены намеренно (задача 1 меряет смещение именно этой синтетики):
- noise_var = max(var(всех элементов остатков) − Σ explained_var, 1e-6). Первое слагаемое ≈ след/d,
  второе ≥ (ncomp/d)·след, поэтому при ncomp ≥ 2 почти всегда noise_var = 1e-6 и внутри кластера
  синтетика имеет ранг ncomp (масштабирование по координатам ранг не меняет). Схема норм percentile
  добавляет одно радиальное направление (норма берётся независимо от положения точки), ранг ncomp + 1;
  схема unit (L2-нормировка) ранг не повышает.
- Запросы — точки базы с шумом: ближайший сосед запроса почти всегда его исходная точка
  (query_source.ibin), что завышает ρ(q) по сравнению с настоящими отложенными запросами.
  Опция --noise_mode consistent (не по умолчанию) берёт (Σ var по координатам − Σ explained_var)/(d − k).
- sample_size берёт первые строки файла без перемешивания.

Чем порт отличается от cuVS:
- ГСЧ кластеров numpy PCG64 вместо cupy: точки не совпадают побитно, совпадает распределение.
  Индексы исходных точек запросов и шум запросов идут из того же numpy-генератора, что в cuVS,
  поэтому индексы совпадают побитно;
- k-means — faiss (случайная инициализация, niter задаётся), в cuVS — k-means++ до 300 итераций;
- точный ground truth в CLI считается по записанному base.fbin (vsbench.gt), k = 200, евклидовы
  (не квадратные) расстояния — как у остальных датасетов проекта; cuVS пишет квадраты.

Формат fingerprint.npz совпадает с cuVS: отпечаток, снятый на GPU, читается здесь и наоборот.

    python experiments/task1_synthetic/cuvs_synth.py fit --dataset data/laion10m/base.fbin \\
        --sample_size 1000000 --n_clusters 10000 --pca_components 32 --output data/fp/laion_nc10k.npz
    python experiments/task1_synthetic/cuvs_synth.py verify --fingerprint data/fp/laion_nc10k.npz \\
        --total_rows 1000000 --n_queries 1000 --k 100 --nprobes 500
    python experiments/task1_synthetic/cuvs_synth.py generate --fingerprint data/fp/laion_nc10k.npz \\
        --total_rows 10000000 --n_queries 10000 --name synth-laion10m
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from utils import BinWriter, read_fbin, write_fbin, write_ibin  # noqa: E402


def data_root() -> Path:
    return Path(os.environ.get("VSBENCH_DATA", ROOT / "data"))

CUVS_COMMIT = "2450691e6c658172d7f1b269a2b1b6b05ab2acbb"
NORM_QUANTILE_COUNT = 256
NORM_UNIT_MEAN_TOL = 0.02
NORM_UNIT_CV_TOL = 0.05
QUERY_SEED_OFFSET = 999_999
_BATCH = 1_000_000


# --------------------------------------------------------------------------- fingerprint

@dataclass
class Fingerprint:
    """Отпечаток датасета, поля как в cuvs_bench.synthesize_dataset.Fingerprint"""

    nclusters: int
    ncols: int
    seed: int
    cluster_centers: np.ndarray            # (nc, d) float32
    cluster_variances: np.ndarray          # (nc, d) float32, дисперсия по координатам
    cluster_densities: np.ndarray          # (nc,) доли, нормируются до суммы 1
    pca_components_list: list[Optional[np.ndarray]]      # (k, d) или None (диагональный запасной путь)
    pca_explained_var_list: list[Optional[np.ndarray]]   # (k,) или None
    pca_noise_var: np.ndarray              # (nc,)
    pca_n_components: int
    norm_quantiles: Optional[np.ndarray] = None   # (nc, 256) или None
    norm_unit: bool = False
    # ручки генерации задачи 1, не входят в отпечаток; значения по умолчанию — поведение cuVS
    tail: str = "fp"            # fp — шум из отпечатка (cuVS); iso — изотропный, равный остатку дисперсии;
                                # diag — по координатам: var_dim − дисперсия, объяснённая PCA
    tail_mult: float = 1.0      # множитель дисперсии хвоста (для iso/diag)
    var_scale: bool = True      # масштаб по координатам к var_dim (cuVS)
    blur: float = 0.0           # сдвиг центра точки к соседнему центроиду на t·(μ_c' − μ_c), t ~ U(0, blur)
    blur_k: int = 8             # из скольких ближайших центроидов выбирать c'
    blur_mix: float = 0.0       # α: шум смешивается, √(1−w)·ε_c + √w·ε_c', w = α·t; при α = 1 и t = 1 точка устроена как точка c'
    _nbr: Optional[np.ndarray] = None   # (nc, blur_k) ближайшие центроиды, считается лениво

    def __post_init__(self):
        dens = np.asarray(self.cluster_densities, dtype=np.float64)
        if dens.shape != (self.nclusters,):
            raise ValueError(f"cluster_densities {dens.shape} != ({self.nclusters},)")
        if not np.isfinite(dens).all() or (dens < 0).any() or dens.sum() <= 0:
            raise ValueError("cluster_densities: нужны конечные неотрицательные с положительной суммой")
        self.cluster_densities = dens / dens.sum()


def resolve_norm_scheme(fp: Fingerprint) -> str:
    if fp.norm_unit:
        return "unit"
    return "percentile" if fp.norm_quantiles is not None else "off"


def save_fingerprint(path: str | Path, fp: dict[str, Any]) -> None:
    """NPZ в схеме cuVS: пустой массив в pca_components_arr — кластер без PCA"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nc = len(fp["centroids"])
    comps = np.empty(nc, dtype=object)
    evs = np.empty(nc, dtype=object)
    empty = np.array([], dtype=np.float32)
    for i in range(nc):
        c, e = fp["pca_components_list"][i], fp["pca_explained_var_list"][i]
        comps[i], evs[i] = (c, e) if c is not None else (empty, empty)
    out = dict(
        centroids=fp["centroids"], densities=fp["densities"],
        variances_per_dim=fp["variances_per_dim"],
        pca_components_arr=comps, pca_explained_var_arr=evs,
        pca_noise_var=fp["pca_noise_var"],
        pca_n_components=np.array([fp["pca_n_components"]]),
        norm_unit=np.array([bool(fp.get("norm_unit", False))]),
    )
    if fp.get("norm_quantiles") is not None:
        out["norm_quantiles"] = np.asarray(fp["norm_quantiles"], dtype=np.float32)
    np.savez(path, **out)


def load_fingerprint(path: str | Path, seed: int) -> Fingerprint:
    data = np.load(path, allow_pickle=True)
    nc = len(data["centroids"])
    comps, evs = [], []
    # NpzFile читает массив заново при каждом data[key]: берём их один раз, иначе O(nc²)
    comps_arr, ev_arr = data["pca_components_arr"], data["pca_explained_var_arr"]
    for i in range(nc):
        c = comps_arr[i]
        if len(c) > 0:
            comps.append(np.asarray(c, dtype=np.float32))
            evs.append(np.asarray(ev_arr[i], dtype=np.float32))
        else:
            comps.append(None)
            evs.append(None)
    cent = data["centroids"]
    return Fingerprint(
        nclusters=int(cent.shape[0]), ncols=int(cent.shape[1]), seed=seed,
        cluster_centers=np.asarray(cent, dtype=np.float32),
        cluster_variances=np.asarray(data["variances_per_dim"], dtype=np.float32),
        cluster_densities=data["densities"].astype(np.float64),
        pca_components_list=comps, pca_explained_var_list=evs,
        pca_noise_var=np.asarray(data["pca_noise_var"], dtype=np.float32),
        pca_n_components=int(data["pca_n_components"][0]),
        norm_quantiles=(np.asarray(data["norm_quantiles"], dtype=np.float32)
                        if "norm_quantiles" in data.files else None),
        norm_unit=bool(data["norm_unit"][0]) if "norm_unit" in data.files else False,
    )


# --------------------------------------------------------------------------- fit

def _kmeans(data: np.ndarray, n_clusters: int, niter: int, seed: int,
            verbose: bool) -> tuple[np.ndarray, np.ndarray]:
    import faiss

    d = data.shape[1]
    # max_points_per_centroid велик: faiss по умолчанию берёт ≤256 точек на центроид, cuVS — все
    km = faiss.Kmeans(d, n_clusters, niter=niter, seed=seed, verbose=verbose,
                      max_points_per_centroid=1 << 30, min_points_per_centroid=1)
    t0 = time.perf_counter()
    km.train(data)
    if verbose:
        print(f"  k-means: {time.perf_counter() - t0:.1f} с, объектив {km.obj[-1]:.4g}", flush=True)
    centroids = km.centroids.astype(np.float32)
    index = faiss.IndexFlatL2(d)
    index.add(centroids)
    labels = np.empty(len(data), dtype=np.int64)
    for s in range(0, len(data), _BATCH):
        _, I = index.search(np.ascontiguousarray(data[s:s + _BATCH]), 1)
        labels[s:s + _BATCH] = I[:, 0]
    return labels, centroids


def _pca(residuals: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """PCA как в cuVS/cuML: центрирование внутри, дисперсия с делителем n−1"""
    R = residuals.astype(np.float64)
    R -= R.mean(axis=0)
    _, s, vt = np.linalg.svd(R, full_matrices=False)
    ev = s[:k] ** 2 / max(len(R) - 1, 1)
    return vt[:k].astype(np.float32), ev.astype(np.float32)


def fit_fingerprint(data: np.ndarray, n_clusters: int, pca_components: int,
                    kmeans_iter: int = 25, seed: int = 1234, noise_mode: str = "cuvs",
                    verbose: bool = True) -> dict[str, Any]:
    """Отпечаток по выборке data (n, d); ключи как у cuvs fit_fingerprint"""
    if noise_mode not in ("cuvs", "consistent"):
        raise ValueError("noise_mode: 'cuvs' или 'consistent'")
    data = np.ascontiguousarray(data, dtype=np.float32)
    n, d = data.shape
    levels = np.linspace(0.0, 1.0, NORM_QUANTILE_COUNT)

    norms = np.empty(n, dtype=np.float32)
    for s in range(0, n, _BATCH):
        norms[s:s + _BATCH] = np.linalg.norm(data[s:s + _BATCH], axis=1)
    n64 = norms.astype(np.float64)
    norm_mean = float(n64.mean())
    norm_cv = float(n64.std() / max(norm_mean, 1e-12))
    norm_unit = abs(norm_mean - 1.0) <= NORM_UNIT_MEAN_TOL and norm_cv <= NORM_UNIT_CV_TOL
    global_q = np.quantile(norms, levels).astype(np.float32)
    if verbose:
        print(f"  нормы: mean={norm_mean:.4f}, CV={norm_cv:.4f}"
              + (" → единичные, флаг norm_unit" if norm_unit else " → сетки квантилей по кластерам"))

    labels, centroids = _kmeans(data, n_clusters, kmeans_iter, seed, verbose)

    densities = np.zeros(n_clusters, dtype=np.float64)
    var_dim = np.zeros((n_clusters, d), dtype=np.float32)
    comps: list = [None] * n_clusters
    evs: list = [None] * n_clusters
    noise = np.zeros(n_clusters, dtype=np.float32)
    nq = None if norm_unit else np.empty((n_clusters, NORM_QUANTILE_COUNT), dtype=np.float32)

    order = np.argsort(labels, kind="stable")
    bounds = np.searchsorted(labels[order], np.arange(n_clusters + 1))
    t0 = time.perf_counter()
    for i in range(n_clusters):
        idx = order[bounds[i]:bounds[i + 1]]
        pts = data[idx]
        m = len(idx)
        densities[i] = m / n
        if nq is not None:
            nq[i] = np.quantile(norms[idx], levels).astype(np.float32) if m >= 1 else global_q

        if m < pca_components + 1:
            # мало точек для PCA ранга ncomp — только диагональ
            var_dim[i] = np.var(pts, axis=0) if m > 1 else np.full(d, 0.01, dtype=np.float32)
            noise[i] = float(np.mean(var_dim[i]))
            continue

        var_dim[i] = np.var(pts, axis=0)
        res = pts - centroids[i]
        k = min(pca_components, m - 1, d)
        comps[i], evs[i] = _pca(res, k)
        if noise_mode == "cuvs":
            total = float(np.var(res))          # дисперсия всех элементов ≈ след/d — как в cuVS
            noise[i] = max(total - float(evs[i].sum()), 1e-6)
        else:
            tail = float(np.var(res, axis=0, ddof=1).sum()) - float(evs[i].sum())
            noise[i] = max(tail / max(d - k, 1), 1e-6)
        if verbose and (i + 1) % max(1, n_clusters // 10) == 0:
            print(f"  PCA: {i + 1}/{n_clusters} кластеров, {time.perf_counter() - t0:.0f} с", flush=True)

    n_valid = sum(c is not None for c in comps)
    if verbose:
        print(f"  PCA ранга ncomp у {n_valid}/{n_clusters} кластеров, "
              f"{n_clusters - n_valid} — диагональ (мало точек)")
    return {
        "centroids": centroids, "densities": densities, "variances_per_dim": var_dim,
        "pca_components_list": comps, "pca_explained_var_list": evs,
        "pca_noise_var": noise, "pca_n_components": int(pca_components),
        "norm_quantiles": nq, "norm_unit": norm_unit,
    }


# --------------------------------------------------------------------------- generate

def get_cluster_seed(base_seed: int, cluster_id: int) -> int:
    return base_seed * 1_000_000 + int(cluster_id)


def get_num_points_per_cluster(total: int, fp: Fingerprint, min_per_cluster: int = 1) -> np.ndarray:
    nc = fp.nclusters
    need = nc * min_per_cluster
    if total < need:
        raise ValueError(f"total_points={total} < nclusters·min_points_per_cluster = {need}")
    ppc = np.full(nc, min_per_cluster, dtype=np.int64)
    rem = total - need
    if rem > 0:
        ppc += (fp.cluster_densities * rem).astype(np.int64)
        diff = total - int(ppc.sum())
        # остаток по кругу, как в cuVS
        if diff:
            np.add.at(ppc, np.arange(abs(diff)) % nc, 1 if diff > 0 else -1)
    return ppc


def tail_variance(fp: Fingerprint, cluster_id: int) -> np.ndarray:
    """Дисперсия вне PCA-подпространства по координатам, (d,): то, что cuVS теряет (шум 1e-6)"""
    comps = fp.pca_components_list[cluster_id]
    ev = fp.pca_explained_var_list[cluster_id]
    var = fp.cluster_variances[cluster_id].astype(np.float64)
    left = np.maximum(var - (comps.astype(np.float64) ** 2 * ev[:, None]).sum(0), 0.0)
    if fp.tail == "iso":
        left = np.full_like(left, left.sum() / max(fp.ncols - len(ev), 1))
    elif fp.tail != "diag":
        raise ValueError(f"tail: fp, iso или diag, а не {fp.tail}")
    return np.maximum(left * fp.tail_mult, 1e-6)


def cluster_neighbors(fp: Fingerprint) -> np.ndarray:
    if fp._nbr is None:
        import faiss
        C = np.ascontiguousarray(fp.cluster_centers, dtype=np.float32)
        _, I = faiss.knn(C, C, fp.blur_k + 1)
        fp._nbr = I[:, 1:]
    return fp._nbr


def borrowed_noise(cluster_id: int, n_points: int, fp: Fingerprint, rng: np.random.Generator) -> np.ndarray:
    """n_points векторов шума кластера cluster_id для точек, которые blur_mix ведёт к нему из другого
    кластера. Масштаб по координатам аналитический: модельная дисперсия (PCA + хвост) против var_dim,
    в gen_cluster он берётся по выборке целого кластера, а здесь кластер целиком не порождается"""
    comps = fp.pca_components_list[cluster_id]
    if comps is None:
        x = rng.standard_normal((n_points, fp.ncols), dtype=np.float32)
        return x * np.sqrt(fp.cluster_variances[cluster_id]).astype(np.float32)
    ev = np.maximum(fp.pca_explained_var_list[cluster_id], 0.0)
    if fp.tail == "fp":
        tvar = np.full(fp.ncols, max(float(fp.pca_noise_var[cluster_id]), 0.0))
        floor = float(fp.pca_noise_var[cluster_id])
    else:
        tvar = tail_variance(fp, cluster_id)
        floor = float(tvar.min())
    z = rng.standard_normal((n_points, len(ev)), dtype=np.float32) * np.sqrt(ev).astype(np.float32)
    x = rng.standard_normal((n_points, fp.ncols), dtype=np.float32) * np.sqrt(tvar).astype(np.float32)
    x += z @ comps
    if fp.var_scale:
        model = (comps.astype(np.float64) ** 2 * ev[:, None]).sum(0) + tvar
        scale = np.minimum(np.sqrt(fp.cluster_variances[cluster_id] / np.maximum(model, floor)), 5.0)
        x *= scale.astype(np.float32)
    return x


def gen_cluster(cluster_id: int, n_points: int, fp: Fingerprint) -> np.ndarray:
    """Все n_points точек кластера. Чистая функция (seed, cluster_id, n_points): кластер нельзя
    порождать по частям — масштабирование дисперсии считается по всему кластеру"""
    rng = np.random.default_rng(get_cluster_seed(fp.seed, cluster_id))
    center = fp.cluster_centers[cluster_id]
    comps = fp.pca_components_list[cluster_id]
    if comps is not None:
        ev = fp.pca_explained_var_list[cluster_id]
        noise_var = float(fp.pca_noise_var[cluster_id])
        z = rng.standard_normal((n_points, len(ev)), dtype=np.float32)
        z *= np.sqrt(np.maximum(ev, 0.0))
        x = rng.standard_normal((n_points, fp.ncols), dtype=np.float32)
        if fp.tail == "fp":
            x *= np.float32(np.sqrt(max(noise_var, 0.0)))
        else:
            tvar = tail_variance(fp, cluster_id)
            x *= np.sqrt(tvar).astype(np.float32)
            noise_var = float(tvar.min())
        x += z @ comps
        if n_points >= 10 and fp.var_scale:
            target = fp.cluster_variances[cluster_id]
            actual = np.var(x, axis=0)
            scale = np.minimum(np.sqrt(target / np.maximum(actual, noise_var)), 5.0)
            x *= scale.astype(np.float32)
        x += center
    else:
        x = rng.standard_normal((n_points, fp.ncols), dtype=np.float32)
        x *= np.sqrt(fp.cluster_variances[cluster_id])
        x += center
    x = x.astype(np.float32, copy=False)
    if fp.blur > 0 and n_points > 0:
        # размываем границы кластеров: в реальных данных соседство запроса пересекает кластеры,
        # у гауссовых кластеров cuVS — нет. Отдельный генератор, чтобы при blur = 0 точки не менялись
        brng = np.random.default_rng(get_cluster_seed(fp.seed, cluster_id) + 7_777_777)
        nbr = cluster_neighbors(fp)[cluster_id]
        pick = brng.integers(0, len(nbr), n_points)
        other = fp.cluster_centers[nbr[pick]]
        t = brng.random((n_points, 1), dtype=np.float32) * np.float32(fp.blur)
        if fp.blur_mix > 0:
            # x − center — шум ε_c; к нему подмешивается шум кластера c' с весами √(1−w), √w, w = α·t,
            # ковариация переходит от Σ_c к Σ_c'. Свой ГСЧ: c' и t те же, что без смешивания,
            # и при любом α те же ε_c'; при α = 1 точки совпадают с прежним флагом --blur_mix
            mrng = np.random.default_rng(get_cluster_seed(fp.seed, cluster_id) + 9_999_999)
            eps2 = np.empty_like(x)
            for k in range(len(nbr)):
                m = pick == k
                if m.any():
                    eps2[m] = borrowed_noise(int(nbr[k]), int(m.sum()), fp, mrng)
            w = t * np.float32(fp.blur_mix)
            x = center + np.sqrt(1.0 - w) * (x - center) + np.sqrt(w) * eps2
            x = x.astype(np.float32, copy=False)
        x += t * (other - center)

    scheme = resolve_norm_scheme(fp)
    if scheme != "off" and n_points > 0:
        cur = np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
        if scheme == "unit":
            x /= cur
        else:
            q = fp.norm_quantiles[cluster_id]
            u = rng.random(n_points, dtype=np.float32)
            target = np.interp(u, np.linspace(0.0, 1.0, len(q)), q).astype(np.float32)[:, None]
            x *= target / cur
    return x


def _iter_clusters(fp: Fingerprint, ppc: np.ndarray, cluster_ids, workers: int):
    """(cluster_id, точки) по порядку; кластеры порождаются параллельно потоками"""
    ids = [int(c) for c in cluster_ids if ppc[c] > 0]
    if workers <= 1:
        for c in ids:
            yield c, gen_cluster(c, int(ppc[c]), fp)
        return
    with ThreadPoolExecutor(workers) as ex:
        win = 4 * workers
        for s in range(0, len(ids), win):
            chunk = ids[s:s + win]
            for c, pts in zip(chunk, ex.map(lambda c: gen_cluster(c, int(ppc[c]), fp), chunk)):
                yield c, pts


def sample_query_sources(nqueries: int, total_rows: int, fp: Fingerprint) -> tuple[np.ndarray, np.random.Generator]:
    """Глобальные индексы исходных точек запросов; генератор потом нужен для шума"""
    rng = np.random.default_rng(fp.seed + QUERY_SEED_OFFSET)
    return rng.choice(total_rows, size=nqueries, replace=False), rng


def add_jitter(queries: np.ndarray, rng: np.random.Generator, normalize: bool) -> np.ndarray:
    sigma = float(np.std(queries)) * 0.1
    q = queries + rng.normal(0, sigma, queries.shape).astype(np.float32)
    if normalize:
        q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-8)
    return q.astype(np.float32)


def generate_synthetic_dataset(fp: Fingerprint, total_points: int, workers: int = 1) -> np.ndarray:
    ppc = get_num_points_per_cluster(total_points, fp)
    out = np.empty((total_points, fp.ncols), dtype=np.float32)
    pos = 0
    for _, pts in _iter_clusters(fp, ppc, range(fp.nclusters), workers):
        out[pos:pos + len(pts)] = pts
        pos += len(pts)
    return out


def generate_to_file(fp: Fingerprint, total_points: int, path: str | Path,
                     query_sources: Optional[np.ndarray] = None, workers: int = 1,
                     verbose: bool = True, exclude: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Пишет base.fbin кластер за кластером. Если даны query_sources — заодно собирает эти точки
    (запросы до шума), чтобы не порождать их кластеры второй раз"""
    ppc = get_num_points_per_cluster(total_points, fp)
    starts = np.concatenate([[0], np.cumsum(ppc)[:-1]])
    picked = None
    if query_sources is not None:
        picked = np.empty((len(query_sources), fp.ncols), dtype=np.float32)
        q_order = np.argsort(query_sources)
        q_sorted = query_sources[q_order]
    held = None
    if exclude is not None:
        # точки с этими глобальными номерами не пишем в базу, а возвращаем (отложенные запросы)
        exclude = np.sort(exclude)
        held = np.empty((len(exclude), fp.ncols), dtype=np.float32)
    t0 = time.perf_counter()
    with BinWriter(path, fp.ncols) as w:
        for c, pts in _iter_clusters(fp, ppc, range(fp.nclusters), workers):
            if held is not None:
                lo, hi = np.searchsorted(exclude, [starts[c], starts[c] + len(pts)])
                if hi > lo:
                    loc = exclude[lo:hi] - starts[c]
                    held[lo:hi] = pts[loc]
                    pts = np.delete(pts, loc, axis=0)
            w.write(pts)
            if picked is not None:
                lo, hi = np.searchsorted(q_sorted, [starts[c], starts[c] + len(pts)])
                for j in range(lo, hi):
                    picked[q_order[j]] = pts[q_sorted[j] - starts[c]]
            if verbose and (c + 1) % max(1, fp.nclusters // 20) == 0:
                print(f"  base: {w.n:,}/{total_points:,}, {time.perf_counter() - t0:.0f} с", flush=True)
    return held if held is not None else picked


def generate_queries(nqueries: int, total_rows: int, fp: Fingerprint, workers: int = 1) -> np.ndarray:
    """Запросы как в cuVS: точка базы (кластер порождается заново) + шум"""
    src, rng = sample_query_sources(nqueries, total_rows, fp)
    ppc = get_num_points_per_cluster(total_rows, fp)
    cum = np.cumsum(ppc)
    cids = np.searchsorted(cum, src, side="right")
    q = np.zeros((nqueries, fp.ncols), dtype=np.float32)
    for c, pts in _iter_clusters(fp, ppc, np.unique(cids), workers):
        sel = np.nonzero(cids == c)[0]
        q[sel] = pts[src[sel] - (cum[c] - ppc[c])]
    return add_jitter(q, rng, normalize=resolve_norm_scheme(fp) == "unit")


# --------------------------------------------------------------------------- ground truth

def _knn(base: np.ndarray, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    import faiss
    D, I = faiss.knn(np.ascontiguousarray(queries, dtype=np.float32),
                     np.ascontiguousarray(base, dtype=np.float32), k)
    return D, I


def compute_groundtruth_exact(queries: np.ndarray, total_rows: int, fp: Fingerprint, k: int,
                              workers: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Точный GT по всем кластерам, порождённым заново; квадраты расстояний, как в cuVS"""
    import faiss

    if not 1 <= k <= total_rows:
        raise ValueError(f"k должно быть в [1, total_rows], k={k}")
    ppc = get_num_points_per_cluster(total_rows, fp)
    starts = np.concatenate([[0], np.cumsum(ppc)[:-1]])
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    heap = faiss.ResultHeap(len(queries), k)
    buf, buf_start, buf_n = [], 0, 0

    def flush():
        nonlocal buf, buf_n
        if buf_n:
            block = np.concatenate(buf)
            D, I = _knn(block, queries, min(k, len(block)))
            heap.add_result(D, I + buf_start)
        buf, buf_n = [], 0

    for c, pts in _iter_clusters(fp, ppc, range(fp.nclusters), workers):
        if buf_n == 0:
            buf_start = int(starts[c])
        buf.append(pts)
        buf_n += len(pts)
        if buf_n >= _BATCH:
            flush()
    flush()
    heap.finalize()
    return heap.I.astype(np.int64), heap.D.astype(np.float32)


def compute_groundtruth_nprobe(queries: np.ndarray, total_rows: int, fp: Fingerprint, k: int,
                               nprobes: int, workers: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Приближённый GT: каждому запросу только nprobes ближайших по центроиду кластеров"""
    if not 1 <= k <= total_rows:
        raise ValueError(f"k должно быть в [1, total_rows], k={k}")
    if not 1 <= nprobes <= fp.nclusters:
        raise ValueError(f"nprobes должно быть в [1, {fp.nclusters}], nprobes={nprobes}")
    ppc = get_num_points_per_cluster(total_rows, fp)
    starts = np.concatenate([[0], np.cumsum(ppc)[:-1]])
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    _, near = _knn(fp.cluster_centers, queries, nprobes)

    flat_q = np.repeat(np.arange(len(queries)), nprobes)
    flat_c = near.ravel()
    order = np.argsort(flat_c, kind="stable")
    flat_q, flat_c = flat_q[order], flat_c[order]
    cids, first = np.unique(flat_c, return_index=True)
    groups = dict(zip(cids.tolist(), np.split(flat_q, first[1:])))

    gt_i = np.full((len(queries), k), -1, dtype=np.int64)
    gt_d = np.full((len(queries), k), np.inf, dtype=np.float32)
    for c, pts in _iter_clusters(fp, ppc, cids, workers):
        qi = groups[c]
        D, I = _knn(pts, queries[qi], min(k, len(pts)))
        md = np.concatenate([gt_d[qi], D], axis=1)
        mi = np.concatenate([gt_i[qi], I.astype(np.int64) + starts[c]], axis=1)
        o = np.argsort(md, axis=1, kind="stable")[:, :k]
        gt_d[qi] = np.take_along_axis(md, o, axis=1)
        gt_i[qi] = np.take_along_axis(mi, o, axis=1)
    return gt_i, gt_d


def recall(found: np.ndarray, truth: np.ndarray, k: int) -> float:
    hits = sum(len(np.intersect1d(f[:k], t[:k])) for f, t in zip(found, truth))
    return hits / (len(found) * k)


def verify_groundtruth(fp: Fingerprint, total_rows: int, nqueries: int, k: int, nprobes: int,
                       workers: int = 1) -> dict:
    q = generate_queries(nqueries, total_rows, fp, workers)
    t0 = time.perf_counter()
    ni, _ = compute_groundtruth_nprobe(q, total_rows, fp, k, nprobes, workers)
    t1 = time.perf_counter()
    ei, _ = compute_groundtruth_exact(q, total_rows, fp, k, workers)
    t2 = time.perf_counter()
    return {"recall": recall(ni, ei, k), "nprobe_time": t1 - t0, "exact_time": t2 - t1,
            "total_rows": total_rows, "nqueries": nqueries, "k": k, "nprobes": nprobes}


# --------------------------------------------------------------------------- CLI

def _load_sample(path: str, sample_size: Optional[int], mode: str = "first", seed: int = 0) -> np.ndarray:
    if path.endswith(".npy"):
        x = np.load(path, mmap_mode="r")
    else:
        x = read_fbin(path)
    if sample_size is not None and mode == "random":
        x = x[np.sort(np.random.default_rng(seed).choice(len(x), min(sample_size, len(x)), replace=False))]
    elif sample_size is not None:
        x = x[:sample_size]       # первые строки, без перемешивания — как в cuVS
    return np.ascontiguousarray(x, dtype=np.float32)


def _default_nprobes(fp: Fingerprint, nprobes: Optional[int]) -> int:
    if nprobes is None:
        nprobes = max(1, round(fp.nclusters * 0.05))
        print(f"  nprobes не задан: 5% от {fp.nclusters} кластеров = {nprobes}")
    if nprobes > fp.nclusters:
        sys.exit(f"--nprobes ({nprobes}) больше числа кластеров ({fp.nclusters})")
    return nprobes


def cmd_fit(a) -> None:
    print(f"чтение {a.dataset}")
    x = _load_sample(a.dataset, a.sample_size, a.sample_mode, a.kmeans_seed)
    print(f"  выборка {x.shape}")
    t0 = time.perf_counter()
    fp = fit_fingerprint(x, a.n_clusters, a.pca_components, kmeans_iter=a.kmeans_iter,
                         seed=a.kmeans_seed, noise_mode=a.noise_mode)
    out = a.output or (f"{Path(a.dataset).stem}_nc{a.n_clusters}_ncomp{a.pca_components}"
                       + ("" if a.sample_size is None else f"_ss{a.sample_size}") + ".npz")
    save_fingerprint(out, fp)
    side = {k: v for k, v in vars(a).items() if k != "func"}
    side.update(n_sample=int(x.shape[0]), d=int(x.shape[1]), fit_seconds=round(time.perf_counter() - t0, 1),
                cuvs_commit=CUVS_COMMIT, port="experiments/task1_synthetic/cuvs_synth.py")
    Path(out).with_suffix(".json").write_text(json.dumps(side, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"отпечаток: {out}")


def cmd_generate(a) -> None:
    from _common import finalize_dataset

    fp = load_fingerprint(a.fingerprint, seed=a.seed)
    fp.tail, fp.tail_mult, fp.var_scale = a.tail, a.tail_mult, not a.no_var_scale
    fp.blur, fp.blur_k, fp.blur_mix = a.blur, a.blur_k, a.blur_mix
    if not 0 <= fp.blur_mix <= 1:
        sys.exit("--blur_mix α от 0 до 1")
    if fp.blur_mix > 0 and fp.blur <= 0:
        sys.exit("--blur_mix только вместе с --blur > 0")
    if fp.blur_mix * fp.blur > 1:
        sys.exit("--blur_mix: нужно α·blur ≤ 1, вес √(1 − α·t)")
    if fp.blur > 0:
        cluster_neighbors(fp)       # до запуска потоков
    print(f"  {fp.nclusters} кластеров, d={fp.ncols}, ncomp={fp.pca_n_components}, "
          f"нормы: {resolve_norm_scheme(fp)}, хвост {fp.tail}×{fp.tail_mult}, масштаб {fp.var_scale}")
    if a.n_queries > a.total_rows:
        sys.exit("--n_queries больше --total_rows")
    out = Path(a.output_dir) if a.output_dir else data_root() / a.name
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    knobs = {"tail": fp.tail, "tail_mult": fp.tail_mult, "var_scale": fp.var_scale,
             "blur": fp.blur, "blur_k": fp.blur_k, "blur_mix": fp.blur_mix}

    if a.holdout:
        # отложенные запросы: порождаем total_rows + holdout точек и убираем holdout случайных из базы,
        # как prepare_parquet откладывает запросы реального датасета. Запросы jitter к этой базе
        # строит scripts/make_query_pool.py build — тот же код, что для реальных данных
        if a.gt_mode != "exact":
            sys.exit("--holdout только с --gt_mode exact")
        n_gen = a.total_rows + a.holdout
        excl = np.random.default_rng(fp.seed + 2 * QUERY_SEED_OFFSET).choice(n_gen, a.holdout, replace=False)
        held = generate_to_file(fp, n_gen, out / "base.fbin", workers=a.workers, exclude=excl)
        held = held[np.random.default_rng(fp.seed).permutation(a.holdout)]   # не по кластерам подряд
        meta = {"source": "experiments/task1_synthetic/cuvs_synth.py generate --holdout",
                "cuvs_commit": CUVS_COMMIT, "fingerprint": str(a.fingerprint),
                "nclusters": fp.nclusters, "pca_n_components": fp.pca_n_components,
                "norm_scheme": resolve_norm_scheme(fp), "seed": a.seed, **knobs,
                "queries": "heldout: точки того же генератора, удалённые из базы",
                "generate_seconds": round(time.perf_counter() - t0, 1)}
        finalize_dataset(out, out / "base.fbin", held, metric="l2", k=a.k, meta=meta)
        return

    src, rng = sample_query_sources(a.n_queries, a.total_rows, fp)
    picked = generate_to_file(fp, a.total_rows, out / "base.fbin", query_sources=src, workers=a.workers)
    queries = add_jitter(picked, rng, normalize=resolve_norm_scheme(fp) == "unit")
    write_ibin(out / "query_source.ibin", src.astype(np.int32)[:, None])
    gen_s = time.perf_counter() - t0

    fp_side = Path(a.fingerprint).with_suffix(".json")
    meta = {
        "source": "experiments/task1_synthetic/cuvs_synth.py generate",
        "cuvs_commit": CUVS_COMMIT, "fingerprint": str(a.fingerprint),
        "fingerprint_fit": json.loads(fp_side.read_text(encoding="utf-8")) if fp_side.exists() else None,
        "nclusters": fp.nclusters, "pca_n_components": fp.pca_n_components,
        "norm_scheme": resolve_norm_scheme(fp), "seed": a.seed, "gt_mode": a.gt_mode,
        "generate_seconds": round(gen_s, 1), **knobs,
        "query_source": "query_source.ibin — индекс точки базы, из которой получен запрос (до шума)",
    }
    if a.gt_mode == "exact":
        finalize_dataset(out, out / "base.fbin", queries, metric="l2", k=a.k, meta=meta)
        return
    nprobes = _default_nprobes(fp, a.nprobes)
    write_fbin(out / "query.fbin", queries)
    ids, d2 = compute_groundtruth_nprobe(queries, a.total_rows, fp, a.k, nprobes, a.workers)
    write_ibin(out / "gt.ibin", ids.astype(np.int32))
    write_fbin(out / "gt.fbin", np.sqrt(np.maximum(d2, 0.0)))
    meta.update(nprobes=nprobes, n=a.total_rows, d=fp.ncols, nq=a.n_queries, k_gt=a.k, metric="l2")
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"готово: {out}")


def cmd_verify(a) -> None:
    fp = load_fingerprint(a.fingerprint, seed=a.seed)
    nprobes = _default_nprobes(fp, a.nprobes)
    r = verify_groundtruth(fp, a.total_rows, a.n_queries, a.k, nprobes, a.workers)
    print(f"recall nprobe-GT против точного: {r['recall']:.4f} "
          f"(nprobe {r['nprobe_time']:.0f} с, точный {r['exact_time']:.0f} с)")
    print(f"nprobes={nprobes} " + ("подходит" if r["recall"] >= 0.999 else "мало, recall < 0,999 — увеличьте"))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    workers = max(1, (os.cpu_count() or 1) // 2)

    p = sub.add_parser("fit", help="отпечаток по реальной выборке")
    p.add_argument("--dataset", required=True, help=".fbin или .npy")
    p.add_argument("--output")
    p.add_argument("--sample_size", type=int, help="первые строки файла, без перемешивания")
    p.add_argument("--n_clusters", type=int, required=True)
    p.add_argument("--pca_components", type=int, required=True)
    p.add_argument("--kmeans_iter", type=int, default=25, help="в cuVS до 300")
    p.add_argument("--kmeans_seed", type=int, default=1234)
    p.add_argument("--noise_mode", choices=["cuvs", "consistent"], default="cuvs")
    p.add_argument("--sample_mode", choices=["first", "random"], default="first",
                   help="first — первые строки файла (cuVS), random — случайные")
    p.set_defaults(func=cmd_fit)

    p = sub.add_parser("generate", help="base/query/gt в формате data/<name>/")
    p.add_argument("--fingerprint", required=True)
    p.add_argument("--total_rows", type=int, required=True)
    p.add_argument("--n_queries", type=int, default=10_000)
    p.add_argument("--k", type=int, default=200)
    p.add_argument("--gt_mode", choices=["exact", "nprobe"], default="exact")
    p.add_argument("--nprobes", type=int)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--name", help="папка в $VSBENCH_DATA")
    g.add_argument("--output_dir")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=workers)
    p.add_argument("--holdout", type=int, default=0,
                   help="вместо запросов cuVS (jitter) отложить столько точек генератора, их нет в базе")
    p.add_argument("--tail", choices=["fp", "iso", "diag"], default="fp",
                   help="дисперсия вне PCA: fp — из отпечатка, как cuVS (≈1e-6); iso/diag — вернуть остаток")
    p.add_argument("--tail_mult", type=float, default=1.0)
    p.add_argument("--no_var_scale", action="store_true", help="без масштаба по координатам к var_dim")
    p.add_argument("--blur", type=float, default=0.0,
                   help="сдвиг центра точки к соседнему центроиду, t ~ U(0, blur); 0 — как cuVS")
    p.add_argument("--blur_k", type=int, default=8)
    p.add_argument("--blur_mix", type=float, nargs="?", const=1.0, default=0.0, metavar="α",
                   help="вместе со сдвигом смешивать шум: √(1−w)·ε_c + √w·ε_c', w = α·t; "
                        "без значения α = 1, 0 — без смешивания")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("verify", help="хватает ли nprobes: nprobe-GT против точного")
    p.add_argument("--fingerprint", required=True)
    p.add_argument("--total_rows", type=int, default=1_000_000)
    p.add_argument("--n_queries", type=int, default=1_000)
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--nprobes", type=int)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=workers)
    p.set_defaults(func=cmd_verify)

    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
