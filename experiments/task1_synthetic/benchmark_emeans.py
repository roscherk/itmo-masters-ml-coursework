"""Compare E-means-style mini-batch updates with Faiss Lloyd on a dataset."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from partition import emean_kmeans, gather, open_base, random_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="dataset directory under data/")
    parser.add_argument("--clusters", type=int, default=16)
    parser.add_argument("--seed", type=int, default=81027)
    parser.add_argument("--train-size", type=int, default=100_000)
    parser.add_argument("--epochs", type=float, default=5)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lloyd-iterations", type=int, default=20)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--output", default="results/task1/emeans_1m.json")
    args = parser.parse_args()

    import faiss
    faiss.omp_set_num_threads(args.threads)
    base = open_base(args.name)
    results = {"dataset": args.name, "shape": list(base.shape),
               "clusters": args.clusters, "seed": args.seed,
               "train_size": args.train_size, "threads": args.threads}
    for method in ("faiss-kmeans", "e-means"):
        if method == "faiss-kmeans":
            training_start = time.perf_counter()
            rows = gather(base, random_rows(len(base), args.train_size, args.seed))
            model = faiss.Kmeans(base.shape[1], args.clusters,
                                 niter=args.lloyd_iterations, seed=args.seed,
                                 verbose=False, max_points_per_centroid=1 << 30)
            model.train(rows)
            centroids = model.centroids.copy()
            result = {"training_seconds": time.perf_counter() - training_start}
            assignment_start = time.perf_counter()
            index = faiss.IndexFlatL2(base.shape[1])
            index.add(centroids)
            labels = np.empty(len(base), dtype=np.int32)
            objective = 0.0
            for start in range(0, len(base), 100_000):
                stop = min(start + 100_000, len(base))
                distances, ids = index.search(np.ascontiguousarray(base[start:stop]), 1)
                labels[start:stop] = ids.ravel()
                objective += float(distances.sum(dtype=np.float64))
            sizes = np.bincount(labels, minlength=args.clusters)
            result.update(objective=objective / len(base), sizes=sizes,
                          assignment_seconds=time.perf_counter() - assignment_start,
                          seed=args.seed, iterations=args.lloyd_iterations)
        else:
            result = emean_kmeans(base, args.clusters, args.seed, args.train_size,
                                  args.epochs, args.batch_size, verbose=False)
            result["assignment_seconds"] = result.pop("full_assignment_seconds")
        results[method] = {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in result.items() if key not in ("labels", "centroids")
        }
        print(f"{method}: objective={result['objective']:.6g}, "
              f"fit={results[method]['training_seconds']:.2f}s, "
              f"assign={results[method]['assignment_seconds']:.2f}s", flush=True)

    import os
    from pathlib import Path
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
