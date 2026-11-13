"""Квантизация в плоском протоколе: recall@100 по кодам и с переранжированием top-L.

    python experiments/task1_synthetic/quant.py laion1m --qsets heldout jitter --nq 1000

Методы (Faiss, плоские индексы — без влияния графа или роутинга):
  rabitq-b{1..8}  IndexRaBitQ, b бит на координату (1 — исходный RaBitQ, >1 — extended RaBitQ),
                  запрос квантуется в 8 бит (qb = 8), чтобы мерить ошибку кодов базы;
  sq4, sq6, sq8   IndexScalarQuantizer, равномерная по координатам;
  pq-m{96,192}    IndexPQ, 8 бит на подвектор (1 и 2 бита на координату при d = 768).
Обучение — на случайных 200k строк базы, коды — для всей базы.
По кодам ищем top-Lmax, затем переранжируем первые L точными расстояниями.
В results/task1/quant.csv: recall@100 по кодам, recall@100 после переранжирования для каждого L
(JSON) и L99 — минимальное L из сетки, при котором средний recall@100 ≥ 0,99.
recall100_se — SE среднего по запросам, sd / √nq (то же, что бутстреп по запросам); recall@100 по кодам
каждого запроса — в data/<name>/quant/<qset>__<method>.npy, чтобы потом сравнивать наборы бутстрепом.
«Минимум бит» для задачи 1 — наименьшее b у rabitq, при котором recall@100 по кодам ≥ 0,99
(и отдельно при фиксированном L); считает compare.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import time

import numpy as np

from t1lib import RESULTS, ds_path, gather, iter_chunks, load_qset, open_base, random_rows, set_threads

L_GRID = [100, 120, 150, 200, 300, 400, 600, 800, 1000, 1500, 2000]
FIELDS = ["dataset", "qset", "method", "bits_per_dim", "bytes_per_vec", "nq", "recall100", "recall100_se",
          "recall100_p10", "L99", "rerank", "train_s", "search_s"]
METHODS = [f"rabitq-b{b}" for b in range(1, 9)] + ["sq4", "sq6", "sq8", "pq-m96", "pq-m192"]


def make_index(method: str, d: int):
    import faiss
    if method.startswith("rabitq-b"):
        b = int(method.split("b")[-1])
        idx = faiss.IndexRaBitQ(d, faiss.METRIC_L2, b)
        idx.qb = 8
        return idx, float(b)
    if method.startswith("sq"):
        b = int(method[2:])
        qt = {4: faiss.ScalarQuantizer.QT_4bit, 6: faiss.ScalarQuantizer.QT_6bit,
              8: faiss.ScalarQuantizer.QT_8bit}[b]
        return faiss.IndexScalarQuantizer(d, qt, faiss.METRIC_L2), float(b)
    if method.startswith("pq-m"):
        m = int(method[4:])
        return faiss.IndexPQ(d, m, 8, faiss.METRIC_L2), 8.0 * m / d
    raise ValueError(method)


def recall100(found: np.ndarray, gt: np.ndarray) -> np.ndarray:
    g = gt[:, :100]
    return np.array([np.intersect1d(f, t).size / 100 for f, t in zip(found, g)])


def run_method(method: str, base: np.ndarray, Xtr: np.ndarray, qsets: dict, l_max: int,
               perq_dir=None) -> list[dict]:
    d = base.shape[1]
    idx, bits = make_index(method, d)
    t0 = time.time()
    idx.train(Xtr)
    for _, X in iter_chunks(base):
        idx.add(X)
    train_s = time.time() - t0
    rows = []
    for qs, (Q, gt) in qsets.items():
        t0 = time.time()
        _, I = idx.search(Q, l_max)
        search_s = time.time() - t0
        r_codes = recall100(I[:, :100], gt)
        if perq_dir is not None:
            perq_dir.mkdir(parents=True, exist_ok=True)
            np.save(perq_dir / f"{qs}__{method}.npy", r_codes.astype(np.float32))
        # точные расстояния до кандидатов: один раз для всех L
        De = np.empty(I.shape, dtype=np.float32)
        for s in range(0, len(Q), 100):              # блоками: 100 × Lmax × d float32 ≈ 0,6 ГБ
            V = gather(base, np.maximum(I[s:s + 100], 0))
            De[s:s + 100] = ((V - Q[s:s + 100, None, :]) ** 2).sum(-1)
        De[I < 0] = np.inf
        rer = {}
        for L in L_GRID:
            if L > l_max:
                break
            o = np.argsort(De[:, :L], axis=1)[:, :100]
            rer[L] = float(recall100(np.take_along_axis(I[:, :L], o, 1), gt).mean())
        L99 = next((L for L in sorted(rer) if rer[L] >= 0.99), None)
        rows.append({"qset": qs, "method": method, "bits_per_dim": bits, "bytes_per_vec": idx.sa_code_size(),
                     "nq": len(Q), "recall100": float(r_codes.mean()),
                     "recall100_se": float(r_codes.std(ddof=1) / np.sqrt(len(Q))),
                     "recall100_p10": float(np.quantile(r_codes, 0.1)), "L99": L99,
                     "rerank": json.dumps(rer), "train_s": round(train_s, 1), "search_s": round(search_s, 1)})
        print(f"  {method:10s} {qs:8s} {idx.sa_code_size():4d} Б: recall@100 по кодам {r_codes.mean():.4f} "
              f"± {rows[-1]['recall100_se']:.4f}, "
              f"L99={L99}, {train_s:.0f}+{search_s:.0f} с", flush=True)
    return rows


def write_rows(rows: list[dict], name: str) -> None:
    path = RESULTS / "quant.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = {(name, r["qset"], r["method"]) for r in rows}
    old = []
    if path.exists():
        with open(path, newline="") as f:
            old = [r for r in csv.DictReader(f) if (r["dataset"], r["qset"], r["method"]) not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(old)
        w.writerows({**r, "dataset": name} for r in rows)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name")
    ap.add_argument("--qsets", nargs="+", default=["heldout", "jitter"])
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--nq", type=int, default=1000)
    ap.add_argument("--l_max", type=int, default=2000)
    ap.add_argument("--n_train", type=int, default=200_000)
    a = ap.parse_args(argv)
    set_threads()
    base = open_base(a.name)
    Xtr = gather(base, random_rows(base.shape[0], a.n_train, 7))
    qsets = {}
    for qs in a.qsets:
        Q, gt, _ = load_qset(a.name, qs)
        rows = random_rows(len(Q), a.nq, 3)
        qsets[qs] = (np.ascontiguousarray(Q[rows]), gt[rows])
    print(f"{a.name}: {base.shape}, запросов {a.nq} на набор")
    for m in a.methods:
        write_rows(run_method(m, base, Xtr, qsets, a.l_max, ds_path(a.name) / "quant"), a.name)


if __name__ == "__main__":
    main()
