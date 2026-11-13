"""Таблица «реальные против синтетики»: метрики из stats.py, rho.csv, quant.csv плюс расстояния
между распределениями баз.

    python experiments/task1_synthetic/compare.py --real laion1m --synth s1m-cuvs s1m-diag --qset heldout

Пишет results/task1/compare_<real>_<qset>.md и печатает его.

Критерии, заданные до экспериментов (синтетика «годится», если):
  ρ:      max_k |Δ ρ̄(k)| ≤ 0,02 и для роутера по центроидам, и для оракула (N = 16, r = 100);
  биты:   то же минимальное число бит RaBitQ для recall@100 ≥ 0,99 по кодам, |Δ recall@100| ≤ 0,01
          при каждом b, L99 совпадает с точностью до шага сетки;
  диагностика (объясняет расхождения, не критерий): медианы LID, RC, d_k — в пределах 10%, KS ≤ 0,1.
Расстояния между базами (выборки по 20k / 100k строк):
  c2st_1nn  — точность 1-NN классификатора «реальная или синтетическая» с исключением самой точки
              (Lopez-Paz, Oquab, 2017; Friedman–Rafsky): 0,5 — неразличимы, 1,0 — разделимы полностью;
  fd_rel    — расстояние Фреше между гауссовыми приближениями (как FID), делённое на след ковариации реальных
"""

from __future__ import annotations

import argparse
import csv
import json

import numpy as np
from scipy import linalg, stats

from t1lib import RESULTS, gather, open_base, random_rows, set_threads

QMETRICS = ["pur64", "pur256", "pur1024", "d1", "d10", "d100", "ratio101", "lid10", "lid100", "rc10", "rc100", "exp20_10",
            "exp200_100", "loc_pr100", "loc_d90", "dup1"]
GMETRICS = ["norm_mean", "norm_std", "mean_norm", "mean_cos", "total_var", "pr", "d50", "d90", "d95",
            "twonn_id", "hub_skew10"]


def load_stats(name: str, qset: str):
    js = json.loads((RESULTS / "stats" / f"{name}.json").read_text())
    npz = np.load(RESULTS / "stats" / f"{name}__{qset}.npz")
    return js, npz


def base_distances(real: str, synth: str, seed: int = 0) -> dict:
    from stats import knn_self
    R = gather(open_base(real), random_rows(open_base(real).shape[0], 100_000, seed + 10))
    S = gather(open_base(synth), random_rows(open_base(synth).shape[0], 100_000, seed + 10))
    m = 20_000
    X = np.concatenate([R[:m], S[:m]])
    lab = np.r_[np.zeros(m), np.ones(m)]
    _, I = knn_self(X, 1)
    acc = float((lab[I[:, 0]] == lab).mean())
    mr, ms = R.mean(0).astype(np.float64), S.mean(0).astype(np.float64)
    Cr, Cs = np.cov(R.astype(np.float64), rowvar=False), np.cov(S.astype(np.float64), rowvar=False)
    cs = linalg.sqrtm(Cr @ Cs).real
    fd = float(((mr - ms) ** 2).sum() + np.trace(Cr + Cs - 2 * cs))
    return {"c2st_1nn": acc, "fd_rel": fd / float(np.trace(Cr))}


def read_csv(path):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def rho_table(name: str, qset: str, part: str) -> dict:
    out = {}
    for r in read_csv(RESULTS / "rho.csv"):
        if (r["dataset"], r["qset"], r["partition"]) == (name, qset, part):
            out[(r["router"], int(r["k"]))] = float(r["rho_mean"])
    return out


def quant_table(name: str, qset: str) -> dict:
    return {r["method"]: r for r in read_csv(RESULTS / "quant.csv") if (r["dataset"], r["qset"]) == (name, qset)}


def min_bits(q: dict, thr: float = 0.99):
    bs = sorted(int(m.split("b")[-1]) for m in q if m.startswith("rabitq-b") and float(q[m]["recall100"]) >= thr)
    return bs[0] if bs else None


def near_thr(q: dict, thr: float, z: float = 1.96) -> list[str]:
    """Методы RaBitQ, у которых порог thr внутри recall ± z·SE (SE по запросам, если записан)"""
    return [m for m in q if m.startswith("rabitq-b") and q[m].get("recall100_se")
            and abs(float(q[m]["recall100"]) - thr) < z * float(q[m]["recall100_se"])]


def rec_cell(q: dict) -> str:
    se = f" ± {1.96 * float(q['recall100_se']):.4f}" if q.get("recall100_se") else ""
    return f"{float(q['recall100']):.4f}{se} / {q['L99'] or '>'}"


def fmt(x):
    if x is None:
        return "—"
    if isinstance(x, str):
        return x
    if abs(x) >= 100:
        return f"{x:.0f}"
    if abs(x) >= 10:
        return f"{x:.1f}"
    return f"{x:.3g}"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", required=True)
    ap.add_argument("--synth", nargs="+", required=True)
    ap.add_argument("--qset", default="heldout")
    ap.add_argument("--synth_qset", default=None, help="набор запросов синтетики, если отличается")
    ap.add_argument("--partition", default="bkm16_s0")
    ap.add_argument("--no_basedist", action="store_true")
    a = ap.parse_args(argv)
    set_threads()
    sq = a.synth_qset or a.qset
    rj, rq = load_stats(a.real, a.qset)
    syn = {s: load_stats(s, sq) for s in a.synth}
    names = [a.real] + a.synth
    head = "| метрика | " + " | ".join(names) + " |\n|" + "---|" * (len(names) + 1) + "\n"
    lines = [f"# {a.real} ({a.qset}) против синтетики ({sq})\n", "## Глобальные (по базе)\n", head]
    for m in GMETRICS:
        lines.append(f"| {m} | " + " | ".join(fmt(x["global"][m]) for x in [rj] + [syn[s][0] for s in a.synth]) + " |\n")
    if not a.no_basedist:
        bd = {s: base_distances(a.real, s) for s in a.synth}
        for m in ("c2st_1nn", "fd_rel"):
            lines.append(f"| {m} | — | " + " | ".join(fmt(bd[s][m]) for s in a.synth) + " |\n")
    lines += ["\n## По запросам: медиана (KS к реальным)\n", head]
    for m in QMETRICS:
        if m not in rq.files or any(m not in syn[s][1].files for s in a.synth):
            continue
        r = rq[m][np.isfinite(rq[m])]
        cells = [fmt(float(np.median(r)))]
        for s in a.synth:
            v = syn[s][1][m]
            v = v[np.isfinite(v)]
            ks = stats.ks_2samp(r, v).statistic if m != "dup1" else abs(r.mean() - v.mean())
            cells.append(f"{fmt(float(np.median(v)) if m != 'dup1' else float(v.mean()))} ({ks:.2f})")
        if m == "dup1":
            cells[0] = fmt(float(r.mean()))
        lines.append(f"| {m} | " + " | ".join(cells) + " |\n")
    # routing
    rr = rho_table(a.real, a.qset, a.partition)
    if rr:
        lines += [f"\n## ρ(q), {a.partition}, r = 100: среднее при k шардах\n", head]
        for router in ("centroid", "oracle"):
            for k in (1, 2, 3, 4, 8):
                cells = [fmt(rr.get((router, k)))]
                for s in a.synth:
                    st = rho_table(s, sq, a.partition)
                    cells.append(fmt(st.get((router, k))))
                lines.append(f"| {router} k={k} | " + " | ".join(cells) + " |\n")
            cells = ["0"]
            for s in a.synth:
                st = rho_table(s, sq, a.partition)
                ks = [k for (ro, k) in rr if ro == router and (ro, k) in st]
                cells.append(fmt(max(abs(rr[(router, k)] - st[(router, k)]) for k in ks)) if ks else "—")
            lines.append(f"| **{router}: max_k Δρ̄** | " + " | ".join(cells) + " |\n")
    # quantization
    qr = quant_table(a.real, a.qset)
    if qr:
        qs = {s: quant_table(s, sq) for s in a.synth}
        lines += ["\n## Квантизация, плоский протокол: recall@100 по кодам (± 1,96·SE по запросам, если есть) / L99\n",
                  head]
        for m in qr:
            cells = [rec_cell(qr[m])]
            for s in a.synth:
                q = qs[s].get(m)
                cells.append(rec_cell(q) if q else "—")
            lines.append(f"| {m} ({qr[m]['bytes_per_vec']} Б) | " + " | ".join(cells) + " |\n")
        def mb(q, thr):
            near = near_thr(q, thr)
            return fmt(min_bits(q, thr)) + (f" (у порога: {', '.join(near)})" if near else "")
        for thr in (0.95, 0.99):
            lines.append(f"| **мин. бит RaBitQ для recall@100 ≥ {thr}** | {mb(qr, thr)} | "
                         + " | ".join(mb(qs[s], thr) for s in a.synth) + " |\n")
        cells = ["0"]
        for s in a.synth:
            common = [m for m in qr if m in qs[s]]
            cells.append(fmt(max(abs(float(qr[m]["recall100"]) - float(qs[s][m]["recall100"])) for m in common))
                         if common else "—")
        lines.append("| **max Δ recall@100** | " + " | ".join(cells) + " |\n")
    text = "".join(lines)
    out = RESULTS / f"compare_{a.real}_{a.qset}{'_vs_' + sq if sq != a.qset else ''}.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"→ {out}")


if __name__ == "__main__":
    main()
