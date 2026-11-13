"""Сколько потоков Faiss выгодно на этой машине: брутфорс kNN (как в gt) и плоский RaBitQ (как в quant.py).

    python experiments/task1_synthetic/thread_sweep.py laion1m
"""

from __future__ import annotations

import sys
import time

import numpy as np

from t1lib import gather, load_qset, open_base, random_rows


def main() -> None:
    import faiss
    name = sys.argv[1]
    base = open_base(name)
    X = gather(base, random_rows(base.shape[0], 500_000, 1))
    Q = np.ascontiguousarray(load_qset(name, "heldout")[0][:2000])
    rq = faiss.IndexRaBitQ(X.shape[1], faiss.METRIC_L2, 4)
    rq.qb = 8
    rq.train(X[:100_000])
    rq.add(X)
    for t in (4, 6, 8, 12, 16):
        faiss.omp_set_num_threads(t)
        t0 = time.perf_counter()
        idx = faiss.IndexFlatL2(X.shape[1])
        idx.add(X)
        idx.search(Q, 200)
        t1 = time.perf_counter()
        rq.search(Q[:500], 1000)
        t2 = time.perf_counter()
        print(f"потоков {t:2d}: kNN {t1 - t0:5.2f} с, RaBitQ-4 бита {t2 - t1:5.2f} с", flush=True)


if __name__ == "__main__":
    main()
