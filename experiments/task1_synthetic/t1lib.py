"""Общие функции задачи 1: наборы запросов, выборки строк базы, потоки Faiss.

Набор запросов датасета data/<name>/ задаётся именем qset:
    root      — query.fbin/gt.* в корне датасета (как их сделал скрипт подготовки);
    <другое>  — пул data/<name>/pools/<qset>/ (формат vsbench.pool).
В задаче 1 у каждого датасета два набора: heldout (точки того же распределения, которых нет в базе)
и jitter (точка базы + шум, схема cuVS random-jitter).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from vsbench.datasets import data_root  # noqa: E402
from vsbench.io import read_fbin, read_ibin  # noqa: E402

THREADS = int(os.environ.get("T1_THREADS", "6"))   # на гибридных Intel 6 потоков быстрее 16
RESULTS = Path(os.environ.get("T1_RESULTS_ROOT", ROOT / "results" / "task1"))


def set_threads(n: int = THREADS) -> None:
    import faiss
    faiss.omp_set_num_threads(n)


def ds_path(name: str) -> Path:
    return data_root() / name


def qset_path(name: str, qset: str) -> Path:
    return ds_path(name) if qset == "root" else ds_path(name) / "pools" / qset


def load_qset(name: str, qset: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(запросы, gt id, gt евклидовы расстояния) набора qset датасета name"""
    p = qset_path(name, qset)
    if not (p / "query.fbin").exists():
        raise FileNotFoundError(f"нет набора запросов {p}")
    return (np.array(read_fbin(p / "query.fbin")), np.array(read_ibin(p / "gt.ibin")),
            np.array(read_fbin(p / "gt.fbin")))


def open_base(name: str) -> np.ndarray:
    return read_fbin(ds_path(name) / "base.fbin")


def random_rows(n: int, m: int, seed: int) -> np.ndarray:
    """m различных номеров строк из n по возрастанию"""
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=min(m, n), replace=False))


def gather(base: np.ndarray, idx: np.ndarray, chunk: int = 200_000) -> np.ndarray:
    """Строки base[idx] (idx любой формы) в RAM; из memmap читаем по отсортированным номерам"""
    flat = np.asarray(idx).ravel()
    order = np.argsort(flat, kind="stable")
    out = np.empty((len(flat), base.shape[1]), dtype=np.float32)
    for s in range(0, len(flat), chunk):
        sel = order[s:s + chunk]
        out[sel] = base[flat[sel]]
    return out.reshape(*np.shape(idx), base.shape[1])


def iter_chunks(base: np.ndarray, chunk: int = 1_000_000):
    for s in range(0, base.shape[0], chunk):
        yield s, np.ascontiguousarray(base[s:s + chunk], dtype=np.float32)
