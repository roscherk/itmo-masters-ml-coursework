"""Run the coursework perturbation baseline, diagonal GMM and cuVS on one sample."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
COURSEWORK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(COURSEWORK / "src"))

from experiments.task1_synthetic.cuvs_synth import (
    fit_fingerprint, save_fingerprint, load_fingerprint, generate_to_file,
    gen_cluster,
)
from vsbench.datasets import data_root
from vsbench.gt import compute_gt
from vsbench.io import BinWriter, read_fbin, write_fbin, write_ibin
from perturbation import PerturbationGenerator
from gmm_generator import GMMGenerator


def cuvs_heldout_queries(fp, n: int, seed: int) -> np.ndarray:
    """Draw independently from the fitted cuVS mixture, not from base jitter."""
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n, fp.cluster_densities)
    chunks = []
    for c in range(fp.nclusters):
        if counts[c]:
            old_seed = fp.seed
            fp.seed = seed
            try:
                chunks.append(gen_cluster(c, int(counts[c]), fp))
            finally:
                fp.seed = old_seed
    Q = np.concatenate(chunks, axis=0)
    rng.shuffle(Q, axis=0)
    return Q


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--fit-size", type=int, default=100_000)
    ap.add_argument("--queries", type=int, default=1_000)
    ap.add_argument("--seed", type=int, default=20261115)
    ap.add_argument("--clusters", type=int, default=1_000)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--methods", nargs="+", default=["perturb", "gmm", "cuvs"])
    a = ap.parse_args()

    import faiss
    faiss.omp_set_num_threads(a.threads)
    real = read_fbin(data_root() / "laion1m" / "base.fbin")
    if a.fit_size > len(real):
        raise ValueError("fit sample exceeds LAION-1M")
    fit_ids = np.sort(np.random.default_rng(a.seed).choice(len(real), a.fit_size, replace=False))
    X = np.ascontiguousarray(real[fit_ids], dtype=np.float32)
    out_root = ROOT / "results" / "task1" / "three_generators"
    out_root.mkdir(parents=True, exist_ok=True)
    np.save(out_root / "fit_rows.npy", fit_ids)
    protocol = {"seed": a.seed, "fit_size": a.fit_size, "fit_source": "laion1m",
                "fit_sample_file": "fit_rows.npy", "n": a.n, "d": int(X.shape[1]),
                "dtype": "float32", "normalize": "L2 unit norm after generation",
                "queries": a.queries, "gt_k": 200, "threads": a.threads,
                "n_clusters": a.clusters,
                "query_seed": a.seed + 91,
                "disturbgen_note": "Coursework perturbation baseline (Faiss k-means, weighted cluster draw, per-coordinate Gaussian std, then L2 normalize); the VecBench paper's DisturbGen perturbs vector-scalar records and does not specify this embedding-only algorithm.",
                "gmm_note": "sklearn GaussianMixture(covariance_type=diag, n_components=n_clusters, max_iter=100, n_init=1, reg_covar=1e-6); L2-normalized after sampling.",
                "cuvs_note": "CPU cuVS port fingerprint fit on this common sample, 32 PCA components, 25 Faiss k-means iterations, cuVS tail and L2-normalization; independent heldout draws from fitted mixture."}
    (out_root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")

    for method in a.methods:
        name = f"tg1m-{method}-heldout-v2"
        ds = data_root() / name
        if (ds / "base.fbin").exists():
            raise FileExistsError(f"refusing to overwrite existing dataset: {ds}")
        ds.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        if method == "perturb":
            gen = PerturbationGenerator(a.clusters, a.seed).fit(X)
            with BinWriter(ds / "base.fbin", X.shape[1]) as w:
                rng = np.random.default_rng(a.seed + 1)
                left = a.n
                while left:
                    size = min(left, 20_000)
                    z = gen.generate(size, seed=int(rng.integers(0, 2**31)))
                    z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
                    w.write(z)
                    left -= size
        elif method == "gmm":
            gen = GMMGenerator(a.clusters, a.seed).fit(X)
            rng = np.random.default_rng(a.seed + 1)
            with BinWriter(ds / "base.fbin", X.shape[1]) as w:
                left = a.n
                while left:
                    size = min(left, 20_000)
                    z = gen.generate(size, seed=int(rng.integers(0, 2**31)))
                    z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
                    w.write(z)
                    left -= size
        elif method == "cuvs":
            fp = fit_fingerprint(X, a.clusters, 32, seed=a.seed, verbose=True)
            save_fingerprint(out_root / "cuvs_fingerprint.npz", fp)
            loaded = load_fingerprint(out_root / "cuvs_fingerprint.npz", seed=a.seed)
            generate_to_file(loaded, a.n, ds / "base.fbin", workers=1)
            Q = cuvs_heldout_queries(loaded, a.queries, a.seed + 91)
        else:
            raise ValueError(method)

        if method in ("perturb", "gmm"):
            Q = gen.generate(a.queries, seed=a.seed + 91)
        Q = np.asarray(Q, dtype=np.float32)
        Q /= np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-12)
        base = read_fbin(ds / "base.fbin")
        ids, dist = compute_gt(base, Q, k=200, chunk=100_000)
        write_fbin(ds / "query.fbin", Q)
        write_ibin(ds / "gt.ibin", ids)
        write_fbin(ds / "gt.fbin", dist)
        meta = {**protocol, "dataset": name, "generator": method,
                "generated_at_seconds": round(time.time() - t0, 2), "query_protocol": "independent synthetic heldout"}
        (ds / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(f"{name}: done in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
