"""Наборы данных задачи 1 поверх готовых датасетов.

subset   — случайные n строк базы src как новый датасет (те же отложенные запросы, gt пересчитан):
    python experiments/task1_synthetic/make_views.py subset laion10m laion1m --n 1000000
heldout  — пул pools/heldout из корневых query/gt (симлинки), чтобы у всех датасетов было одно имя:
    python experiments/task1_synthetic/make_views.py heldout laion1m
Запросы jitter строит scripts/make_query_pool.py build <name> --pool jitter --n 10000 --normalize.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from t1lib import ds_path, open_base, random_rows, set_threads
from vsbench.io import BinWriter, read_fbin


def cmd_subset(a) -> None:
    from _common import finalize_dataset

    set_threads()
    src = open_base(a.src)
    rows = random_rows(src.shape[0], a.n, a.seed)
    out = ds_path(a.name)
    out.mkdir(parents=True, exist_ok=True)
    with BinWriter(out / "base.fbin", src.shape[1]) as w:
        for s in range(0, len(rows), 200_000):
            w.write(np.asarray(src[rows[s:s + 200_000]], dtype=np.float32))
    np.save(out / "rows_in_src.npy", rows.astype(np.int64))
    Q = np.array(read_fbin(ds_path(a.src) / "query.fbin"))
    src_meta = json.loads((ds_path(a.src) / "meta.json").read_text(encoding="utf-8"))
    finalize_dataset(out, out / "base.fbin", Q, metric="l2", k=a.k, meta={
        "source": f"experiments/task1_synthetic/make_views.py subset {a.src}",
        "subset_of": a.src, "subset_seed": a.seed, "rows": "rows_in_src.npy", "src_meta": src_meta})
    link_heldout(a.name)


def link_heldout(name: str) -> None:
    p = ds_path(name) / "pools" / "heldout"
    p.mkdir(parents=True, exist_ok=True)
    for f in ("query.fbin", "gt.ibin", "gt.fbin"):
        if (p / f).is_symlink() or (p / f).exists():
            (p / f).unlink()
        os.symlink(os.path.join("..", "..", f), p / f)
    (p / "meta.json").write_text(json.dumps({"kind": "heldout", "from": "root query.fbin"}), encoding="utf-8")
    print(f"→ {p}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("subset")
    p.add_argument("src")
    p.add_argument("name")
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k", type=int, default=200)
    p.set_defaults(func=cmd_subset)
    p = sub.add_parser("heldout")
    p.add_argument("name")
    p.set_defaults(func=lambda a: link_heldout(a.name))
    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
