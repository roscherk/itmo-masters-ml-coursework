"""Compare E-means and Faiss Lloyd routing on cuVS synthetic and LAION 1M."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "task1_synthetic"))

from utils import read_fbin, read_ibin
from partition import emean_kmeans, gather, random_rows


def recall_curves(labels: np.ndarray, centroids: np.ndarray, queries: np.ndarray,
                  gt: np.ndarray, r: int) -> list[dict]:
    import faiss

    labels_gt = labels[gt[:, :r]]
    counts = np.zeros((len(gt), len(centroids)), dtype=np.int16)
    rows = np.arange(len(gt))[:, None]
    np.add.at(counts, (rows.repeat(r, axis=1), labels_gt), 1)
    oracle_order = np.argsort(-counts, axis=1)

    index = faiss.IndexFlatL2(centroids.shape[1])
    index.add(np.ascontiguousarray(centroids, dtype=np.float32))
    _, centroid_order = index.search(np.ascontiguousarray(queries, dtype=np.float32), len(centroids))

    results = []
    for router, order in (("centroid", centroid_order), ("oracle", oracle_order)):
        selected = np.zeros_like(counts, dtype=bool)
        for k in range(1, len(centroids) + 1):
            selected[np.arange(len(gt))[:, None], order[:, :k]] = True
            per_query = (counts * selected).sum(axis=1) / r
            results.append({"router": router, "k": k, "mean": float(per_query.mean()),
                            "median": float(np.median(per_query)),
                            "p10": float(np.quantile(per_query, .1)),
                            "frac_ge_0.9": float(np.mean(per_query >= .9)),
                            "nq": len(gt), "r": r})
    return results


def run_method(base: np.ndarray, method: str, clusters: int, seed: int,
               train_size: int, epochs: float, batch_size: int | None,
               lloyd_iterations: int) -> dict:
    import faiss
    from partition import iter_chunks

    n, d = base.shape
    start = time.perf_counter()
    training = gather(base, random_rows(n, train_size, seed))
    if method == "e-means":
        fit = emean_kmeans(base, clusters, seed, train_size, epochs, batch_size,
                           verbose=False)
        labels, centroids = fit["labels"].astype(np.int64), fit["centroids"]
        fit_seconds = fit["training_seconds"]
    else:
        km = faiss.Kmeans(d, clusters, niter=lloyd_iterations, seed=seed,
                          verbose=False, max_points_per_centroid=1 << 30)
        km.train(training)
        centroids = km.centroids.copy()
        index = faiss.IndexFlatL2(d)
        index.add(centroids)
        labels = np.empty(n, dtype=np.int64)
        objective = 0.0
        assignment_start = time.perf_counter()
        for offset, chunk in iter_chunks(base):
            dist, ids = index.search(chunk, 1)
            labels[offset:offset + len(chunk)] = ids.ravel()
            objective += float(dist.sum(dtype=np.float64))
        fit_seconds = time.perf_counter() - start
        fit = {"objective": objective / n,
               "assignment_seconds": time.perf_counter() - assignment_start}
    del training
    if method == "e-means":
        fit["assignment_seconds"] = fit.pop("full_assignment_seconds")
    sizes = np.bincount(labels, minlength=clusters)
    return {"labels": labels, "centroids": centroids,
            "metrics": {"objective": fit["objective"], "sizes_min": int(sizes.min()),
                        "sizes_max": int(sizes.max()), "training_seconds": fit_seconds,
                        "assignment_seconds": fit["assignment_seconds"],
                        "epochs": epochs if method == "e-means" else None,
                        "iterations": lloyd_iterations if method == "faiss-kmeans" else None}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--laion", default="/home/heapof/uni/thesis/itmo-vector-search/data/laion1m")
    ap.add_argument("--synthetic", default="data/tg1m-cuvs")
    ap.add_argument("--output", default="results/task1/emeans_routing_comparison.json")
    ap.add_argument("--clusters", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-size", type=int, default=100_000)
    ap.add_argument("--epochs", type=float, default=5)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--lloyd-iterations", type=int, default=20)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--r", type=int, default=100)
    args = ap.parse_args()

    import faiss
    faiss.omp_set_num_threads(args.threads)
    datasets = {"cuVS synthetic": Path(args.synthetic), "LAION 1M": Path(args.laion)}
    results = {"protocol": {"clusters": args.clusters, "seed": args.seed,
                             "train_size": args.train_size, "threads": args.threads,
                             "r": args.r, "query_set": "root heldout"},
               "datasets": {}}

    for name, path in datasets.items():
        base = read_fbin(path / "base.fbin")
        queries = read_fbin(path / "query.fbin")
        gt = read_ibin(path / "gt.ibin")
        per_method = {}
        for method in ("faiss-kmeans", "e-means"):
            model = run_method(base, method, args.clusters, args.seed, args.train_size,
                               args.epochs, args.batch_size, args.lloyd_iterations)
            curves = recall_curves(model["labels"], model["centroids"], queries, gt, args.r)
            per_method[method] = {"metrics": model["metrics"], "routing": curves}
            print(f"{name} {method}: objective={model['metrics']['objective']:.6g} "
                  f"fit={model['metrics']['training_seconds']:.2f}s", flush=True)
            del model
        results["datasets"][name] = {"shape": list(base.shape), "nq": len(queries),
                                      "methods": per_method}

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
