"""C2ST и относительное расстояние Фреше для пилота three_generators (1M) против laion1m.

Считает те же две величины, что `compare.py` → `base_distances`, но без зависимости от
`t1lib`/`vsbench`: базы берутся из каталога `data/` в формате fbin.

    python experiments/task1_synthetic/tg_c2st.py

Пишет `results/task1/three_generators/c2st.csv` (по строке на генератор) и печатает таблицу.

Протокол (как в compare.py): выборка 100k случайных строк из каждой базы, первые 20k идут в
1-NN классификатор «реальная или синтетическая» (0,5 — неразличимы), ковариации для Фреше
считаются по всем 100k.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
from scipy import linalg

DATA = Path(os.environ.get("VSBENCH_DATA", Path(__file__).resolve().parents[2] / "data"))
RESULTS = Path(__file__).resolve().parents[2] / "results" / "task1" / "three_generators"
REAL = "laion1m"
SYNTH = ["tg1m-perturb-heldout-v2", "tg1m-gmm-heldout-v2", "tg1m-cuvs-heldout-v2"]
SAMPLE = 100_000
M = 20_000
SEED = 10


def open_base(name: str) -> np.ndarray:
    path = DATA / name / "base.fbin"
    with open(path, "rb") as f:
        n, d = np.frombuffer(f.read(8), dtype=np.uint32)
    return np.memmap(path, dtype=np.float32, mode="r", offset=8, shape=(int(n), int(d)))


def sample(base: np.ndarray, m: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(base.shape[0], size=min(m, base.shape[0]), replace=False))
    return np.asarray(base[idx], dtype=np.float32)


def c2st_and_frechet(real: str, synth: str) -> dict:
    import faiss

    faiss.omp_set_num_threads(int(os.environ.get("T1_THREADS", "6")))
    R = sample(open_base(real), SAMPLE, SEED)
    S = sample(open_base(synth), SAMPLE, SEED)
    X = np.concatenate([R[:M], S[:M]])
    lab = np.r_[np.zeros(M), np.ones(M)]
    _, I = faiss.knn(X, X, 2)
    # самой точкой считаем столбец, где I == номер строки; иначе (дубликат) — первый
    own = I == np.arange(len(X))[:, None]
    drop = np.where(own.any(1), own.argmax(1), 0)
    keep = np.ones_like(I, dtype=bool)
    keep[np.arange(len(X)), drop] = False
    nearest = I[keep].reshape(len(X), 1)[:, 0]
    acc = float((lab[nearest] == lab).mean())

    R64 = np.asarray(R, dtype=np.float64)
    S64 = np.asarray(S, dtype=np.float64)
    mr, ms = R64.mean(0), S64.mean(0)
    Cr, Cs = np.cov(R64, rowvar=False), np.cov(S64, rowvar=False)
    cs = linalg.sqrtm(Cr @ Cs).real
    fd = float(((mr - ms) ** 2).sum() + np.trace(Cr + Cs - 2 * cs))
    return {"c2st_1nn": acc, "fd_rel": fd / float(np.trace(Cr))}


def main() -> None:
    rows = []
    for name in SYNTH:
        res = c2st_and_frechet(REAL, name)
        rows.append({"dataset": name, "real": REAL, **res})
        print(f"{name}: C2ST {res['c2st_1nn']:.3f}  FD_rel {res['fd_rel']:.4f}", flush=True)
    out = RESULTS / "c2st.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "real", "c2st_1nn", "fd_rel"])
        w.writeheader()
        w.writerows(rows)
    print(f"→ {out}")


if __name__ == "__main__":
    main()
