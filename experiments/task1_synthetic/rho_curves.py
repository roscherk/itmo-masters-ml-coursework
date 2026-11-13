"""ρ(q) от числа опрошенных шардов k: роутер по центроидам шардов и оракул.

    python experiments/task1_synthetic/rho_curves.py laion1m --qsets heldout jitter --partition bkm16_s0

Роутер по центроидам: шарды по возрастанию ‖q − c_j‖, c_j — среднее шарда (partition.py).
Оракул: k шардов, где больше всего истинных соседей (vsbench.metrics.oracle_routing_recall).
r = 100. Строки пишутся в results/task1/rho.csv; строки с тем же (dataset, qset, partition)
перезаписываются.
"""

from __future__ import annotations

import argparse
import csv

import numpy as np

from t1lib import RESULTS, ds_path, load_qset, set_threads
from vsbench.metrics import oracle_routing_recall, routing_recall

FIELDS = ["dataset", "qset", "partition", "router", "k", "r", "rho_mean", "rho_median", "rho_p10",
          "frac_ge_0.9", "frac_ge_0.99", "nq"]


def centroid_order(Q: np.ndarray, C: np.ndarray) -> np.ndarray:
    import faiss
    _, I = faiss.knn(np.ascontiguousarray(Q, dtype=np.float32), np.ascontiguousarray(C, dtype=np.float32), len(C))
    return I


def curves(labels: np.ndarray, C: np.ndarray, Q: np.ndarray, gt: np.ndarray, r: int) -> list[dict]:
    order = centroid_order(Q, C)
    rows = []
    for k in range(1, len(C) + 1):
        for router, rho in (("centroid", routing_recall(gt, labels, order[:, :k], r)),
                            ("oracle", oracle_routing_recall(gt, labels, k, r))):
            rows.append({"router": router, "k": k, "r": r, "rho_mean": rho.mean(),
                         "rho_median": float(np.median(rho)), "rho_p10": float(np.quantile(rho, 0.1)),
                         "frac_ge_0.9": float((rho >= 0.9).mean()), "frac_ge_0.99": float((rho >= 0.99).mean()),
                         "nq": len(Q)})
    return rows


def write_rows(rows: list[dict], key: tuple) -> None:
    path = RESULTS / "rho.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    old = []
    if path.exists():
        with open(path, newline="") as f:
            old = [r for r in csv.DictReader(f) if (r["dataset"], r["qset"], r["partition"]) != key]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(old)
        w.writerows(rows)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name")
    ap.add_argument("--qsets", nargs="+", default=["heldout", "jitter"])
    ap.add_argument("--partition", nargs="+", default=["bkm16_s0"])
    ap.add_argument("--r", type=int, default=100)
    a = ap.parse_args(argv)
    set_threads()
    for part in a.partition:
        P = np.load(ds_path(a.name) / "partitions" / f"{part}.npz")
        labels, C = P["labels"].astype(np.int64), P["centroids"]
        for qs in a.qsets:
            Q, gt, _ = load_qset(a.name, qs)
            rows = curves(labels, C, Q, gt, a.r)
            for r in rows:
                r.update(dataset=a.name, qset=qs, partition=part)
            write_rows(rows, (a.name, qs, part))
            c = {r["k"]: r["rho_mean"] for r in rows if r["router"] == "centroid"}
            o = {r["k"]: r["rho_mean"] for r in rows if r["router"] == "oracle"}
            print(f"{a.name}/{qs}/{part}: ρ центроиды k=1,2,4: {c[1]:.3f} {c[2]:.3f} {c[4]:.3f}; "
                  f"оракул k=1,2,4: {o[1]:.3f} {o[2]:.3f} {o[4]:.3f}")


if __name__ == "__main__":
    main()
