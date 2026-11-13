"""Attach one common heldout LAION query set and exact per-base GT."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from t1lib import ds_path, open_base, set_threads
from vsbench.datasets import data_root
from vsbench.gt import compute_gt
from vsbench.io import read_fbin, write_fbin, write_ibin


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nq", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=81027)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--datasets", nargs="+", required=True)
    args = ap.parse_args()
    set_threads(args.threads)

    Qall = read_fbin(data_root() / "laion1m" / "query.fbin")
    rng = np.random.default_rng(args.seed)
    rows = np.sort(rng.choice(len(Qall), min(args.nq, len(Qall)), replace=False))
    Q = np.ascontiguousarray(Qall[rows])
    for name in args.datasets:
        out = ds_path(name) / "pools" / "laion-heldout"
        out.mkdir(parents=True, exist_ok=True)
        if (out / "gt.ibin").exists():
            raise FileExistsError(f"refusing to overwrite existing query set: {out}")
        base = open_base(name)
        ids, dist = compute_gt(base, Q, k=200, chunk=100_000)
        write_fbin(out / "query.fbin", Q)
        write_ibin(out / "gt.ibin", ids)
        write_fbin(out / "gt.fbin", dist)
        np.save(out / "laion_query_rows.npy", rows)
        print(f"{name}: common queries={len(Q)}, exact GT k=200", flush=True)


if __name__ == "__main__":
    main()
