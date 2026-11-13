"""Δρ̄ «реальные против синтетики» с интервалом: шум сидов balanced k-means, запросов и генератора.

    python experiments/task1_synthetic/rho_ci.py --real laion1m --synth s1m-fp-blur1.0 \
        --gens s1m-fp-blur1.0-g1 s1m-fp-blur1.0-g2 --extra s1m-fp s1m-nc10k-fp-blur1.0 --qsets heldout jitter

ρ(q) по каждому запросу пересчитывается из data/<name>/partitions/bkm16_s*.npz и gt (база не нужна),
берутся все сиды, которые есть на диске. Для датасета с n сидами и nq запросами, R[s, q] — ρ запроса q
при разбиении s:
    ρ̄       = R.mean()
    SE_seed = sd(R.mean(1)) / √n         — разброс между разбиениями, запросы общие;
    SE_q    = sd(R.mean(0)) / √nq        — то же, что бутстреп по запросам (новые запросы того же
                                           распределения сдвинут ρ̄ на столько же);
    SE      = √(SE_seed² + SE_q²).
С --gens синтетика — G реплик генератора (основной датасет и сиды генератора): ρ̄ — среднее средних
реплик, SE = sd(средних реплик) / √G. В этот SE уже входят сиды k-means и запросы каждой реплики,
потому что у реплик свои запросы. sd_gen — разброс генератора сверх них: √(var(средних) − mean(SE²)).
Δ = ρ̄_синт − ρ̄_реал, SE_Δ = √(SE_р² + SE_с²), 95% интервал Δ ± 1,96·SE_Δ (нормальное приближение;
при G ≤ 5 репликах оно оптимистично, sd по 4 степеням свободы сама неточна).

Критерий задачи 1: max_k |Δρ̄| ≤ 0,02. Вердикт по каждому роутеру:
    «с запасом»        max_k (|Δ| + 1,645·SE_Δ) ≤ 0,02 — односторонняя 95% граница внутри порога;
    «по точечной»      max_k |Δ| ≤ 0,02, но граница выходит за порог;
    «нет»              max_k |Δ| > 0,02.
Пишет results/task1/rho_ci_<real>.md и печатает его. В rho_ci_<real>.csv — ρ̄, SE и его части по всем k
для real, synth, реплик, датасетов из --extra и реплик вместе (dataset = "<synth>+gens"), роутеры centroid,
oracle и gap = oracle − centroid (потеря роутера по центроидам, SE считается по разности на запрос).
По этому CSV строится plot_rho.py
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass

import numpy as np

from t1lib import RESULTS, ds_path, load_qset, qset_path, set_threads
from rho_curves import centroid_order

THR = 0.02
Z95, Z1 = 1.96, 1.645


def partitions(name: str, N: int = 16) -> list[str]:
    ps = sorted(ds_path(name).glob(f"partitions/bkm{N}_s*.npz"),
                key=lambda p: int(re.search(r"_s(\d+)\.npz$", p.name).group(1)))
    if not ps:
        raise FileNotFoundError(f"нет разбиений {ds_path(name)}/partitions/bkm{N}_s*.npz")
    return [p.stem for p in ps]


def per_query(name: str, qset: str, part: str, r: int = 100) -> dict[str, np.ndarray]:
    """{"centroid", "oracle", "gap"} → (N, nq): ρ(q) при k = 1..N, gap = oracle − centroid"""
    P = np.load(ds_path(name) / "partitions" / f"{part}.npz")
    labels, C = P["labels"].astype(np.int64), P["centroids"]
    Q, gt, _ = load_qset(name, qset)
    N = len(C)
    nn = labels[gt[:, :r]]                                            # (nq, r): шард каждого соседа
    counts = np.zeros((len(Q), N), dtype=np.int32)
    np.add.at(counts, (np.arange(len(Q))[:, None], nn), 1)
    oracle = np.cumsum(-np.sort(-counts, axis=1), axis=1).T / r       # k шардов с наибольшим числом соседей
    order = centroid_order(Q, C)
    by_rank = np.take_along_axis(counts, order, axis=1)               # соседи в шардах по возрастанию ‖q − c‖
    centroid = np.cumsum(by_rank, axis=1).T / r
    return {"centroid": centroid, "oracle": oracle, "gap": oracle - centroid}


@dataclass
class Est:
    mean: np.ndarray        # (N,) по k
    se: np.ndarray
    se_seed: np.ndarray
    se_q: np.ndarray
    sd_seed: np.ndarray
    n: int                  # сидов (или реплик)
    sd_gen: np.ndarray | None = None
    seed_means: np.ndarray | None = None    # (n, N)


def estimate(name: str, qset: str, router: str, cache: dict) -> Est:
    parts = partitions(name)
    R = np.stack([cache.setdefault((name, qset, p), per_query(name, qset, p))[router] for p in parts])  # (n, N, nq)
    n, _, nq = R.shape
    seed_means = R.mean(2)                                           # (n, N)
    sd_seed = seed_means.std(0, ddof=1) if n > 1 else np.full(R.shape[1], np.nan)
    se_seed = sd_seed / np.sqrt(n) if n > 1 else np.zeros(R.shape[1])
    se_q = R.mean(0).std(1, ddof=1) / np.sqrt(nq)
    return Est(seed_means.mean(0), np.sqrt(se_seed ** 2 + se_q ** 2), se_seed, se_q, sd_seed, n,
               seed_means=seed_means)


def estimate_gens(names: list[str], qset: str, router: str, cache: dict) -> Est:
    """Реплики генератора. Дисперсия сидов k-means общая для реплик и оценивается по всем вместе
    (у реплики с одним сидом своей оценки нет); средняя реплики g имеет дисперсию
    w_g = σ²_seed / n_g + SE_q,g² + σ²_gen. Отсюда σ²_gen = max(0, var(средних) − mean(w_g без σ²_gen))
    и SE² среднего по G репликам = Σ w_g / G² (w_g с σ²_gen)."""
    es = [estimate(nm, qset, router, cache) for nm in names]
    G = len(es)
    M = np.stack([e.mean for e in es])
    dfs = np.array([e.n - 1 for e in es])
    if dfs.sum() == 0:
        raise ValueError("нужен хотя бы один датасет с двумя сидами k-means")
    var_seed = sum((e.seed_means.var(0, ddof=1) * (e.n - 1) for e in es if e.n > 1)) / dfs.sum()
    within = np.stack([var_seed / e.n + e.se_q ** 2 for e in es])    # (G, N)
    var_gen = np.maximum(M.var(0, ddof=1) - within.mean(0), 0)
    se = np.sqrt((within + var_gen).sum(0)) / G
    e0 = es[0]
    return Est(M.mean(0), se, e0.se_seed, e0.se_q, e0.sd_seed, G, np.sqrt(var_gen))


def f(x: float, p: int = 4) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{x:.{p}f}"


def verdict(d: np.ndarray, se: np.ndarray) -> tuple[str, int, int]:
    up = np.abs(d) + Z1 * se
    k_pt, k_up = int(np.argmax(np.abs(d))), int(np.argmax(up))
    if up[k_up] <= THR:
        v = "с запасом"
    elif abs(d[k_pt]) <= THR:
        v = "по точечной"
    else:
        v = "нет"
    return v, k_pt, k_up


def table(real: str, synth: str, gens: list[str], qset: str, cache: dict, ks=(1, 2, 3, 4, 8)) -> list[str]:
    reps = [synth] + gens
    out = [f"\n## {qset}: {real} против {synth}"
           + (f" и {len(gens)} сидов генератора" if gens else "") + "\n\n"]
    out.append(f"Сидов k-means: {real} — {len(partitions(real))}, "
               + ", ".join(f"{s} — {len(partitions(s))}" for s in reps) + "\n\n")
    hdr = ["k", "ρ̄ реал ± SE", "sd сидов", "ρ̄ синт ± SE", "sd сидов", "Δ", "95% интервал Δ", "SE сиды / запросы"]
    if gens:
        hdr += [f"ρ̄ по {len(reps)} репликам ± SE", "Δ с генератором", "95% интервал", "sd_gen"]
    summary = []
    for router in ("centroid", "oracle"):
        er = estimate(real, qset, router, cache)
        es = estimate(synth, qset, router, cache)
        eg = estimate_gens(reps, qset, router, cache) if gens else None
        out.append(f"\n**{router}**\n\n| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) + "\n")
        d = es.mean - er.mean
        se = np.sqrt(er.se ** 2 + es.se ** 2)
        dg = seg = None
        if gens:
            dg = eg.mean - er.mean
            seg = np.sqrt(er.se ** 2 + eg.se ** 2)
        for k in ks:
            i = k - 1
            se_seed = np.sqrt(er.se_seed[i] ** 2 + es.se_seed[i] ** 2)
            se_q = np.sqrt(er.se_q[i] ** 2 + es.se_q[i] ** 2)
            row = [str(k), f"{f(er.mean[i])} ± {f(er.se[i])}", f(er.sd_seed[i]),
                   f"{f(es.mean[i])} ± {f(es.se[i])}", f(es.sd_seed[i]), f"{d[i]:+.4f}",
                   f"[{d[i] - Z95 * se[i]:+.4f}, {d[i] + Z95 * se[i]:+.4f}]", f"{f(se_seed)} / {f(se_q)}"]
            if gens:
                row += [f"{f(eg.mean[i])} ± {f(eg.se[i])}", f"{dg[i]:+.4f}",
                        f"[{dg[i] - Z95 * seg[i]:+.4f}, {dg[i] + Z95 * seg[i]:+.4f}]", f(eg.sd_gen[i])]
            out.append("| " + " | ".join(row) + " |\n")
        v, kp, ku = verdict(d, se)
        line = (f"| **{router}, max_k** | | | | | {abs(d[kp]):.4f} (k = {kp + 1}) | "
                f"граница {abs(d[ku]) + Z1 * se[ku]:.4f} (k = {ku + 1}) | **{v}** |")
        sline = f"{router}: max_k |Δ| {abs(d[kp]):.4f} (k={kp + 1}), граница {abs(d[ku]) + Z1 * se[ku]:.4f} — {v}"
        if gens:
            vg, kpg, kug = verdict(dg, seg)
            line += (f" | {abs(dg[kpg]):.4f} (k = {kpg + 1}) | граница {abs(dg[kug]) + Z1 * seg[kug]:.4f} "
                     f"(k = {kug + 1}) | **{vg}** |")
            sline += (f"; с генератором {abs(dg[kpg]):.4f} (k={kpg + 1}), "
                      f"граница {abs(dg[kug]) + Z1 * seg[kug]:.4f} — {vg}")
        out.append(line + "\n")
        summary.append(sline)
    return out, summary


def csv_rows(real: str, synth: str, gens: list[str], extra: list[str], qset: str, cache: dict) -> list[dict]:
    rows = []
    def add(ds: str, router: str, e: Est) -> None:
        for i in range(len(e.mean)):
            rows.append({"dataset": ds, "qset": qset, "router": router, "k": i + 1, "n": e.n,
                         "rho_mean": e.mean[i], "se": e.se[i], "se_seed": e.se_seed[i], "se_q": e.se_q[i],
                         "sd_seed": e.sd_seed[i],
                         "sd_gen": e.sd_gen[i] if e.sd_gen is not None else np.nan})
    for router in ("centroid", "oracle", "gap"):
        for ds in dict.fromkeys([real, synth, *gens, *extra]):
            add(ds, router, estimate(ds, qset, router, cache))
        if gens:
            add(f"{synth}+gens", router, estimate_gens([synth] + gens, qset, router, cache))
    return rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", required=True)
    ap.add_argument("--synth", required=True)
    ap.add_argument("--gens", nargs="*", default=[], help="реплики синтетики с другим сидом генератора")
    ap.add_argument("--extra", nargs="*", default=[], help="ещё датасеты только в CSV (для графика)")
    ap.add_argument("--qsets", nargs="+", default=["heldout", "jitter"])
    ap.add_argument("--out", default=None, help="имя файла в results/task1/ (по умолчанию rho_ci_<real>.md)")
    a = ap.parse_args(argv)
    set_threads()
    cache: dict = {}
    lines = [f"# ρ(q): {a.real} против {a.synth}, интервалы по сидам и запросам\n\n",
             "Порог max_k |Δρ̄| ≤ 0,02; граница — max_k (|Δ| + 1,645·SE_Δ). Как считается — rho_ci.py\n"]
    summ, rows = [], []
    for qs in a.qsets:
        t, s = table(a.real, a.synth, a.gens, qs, cache)
        lines += t
        summ += [f"{qs} {x}" for x in s]
        gens = [g for g in a.gens if (qset_path(g, qs) / "query.fbin").exists()]
        rows += csv_rows(a.real, a.synth, gens, a.extra, qs, cache)
    text = "".join(lines)
    out = RESULTS / (a.out or f"rho_ci_{a.real}.md")
    out.write_text(text, encoding="utf-8")
    print(text)
    print("\n".join(summ))
    csv = out.with_suffix(".csv")
    import pandas as pd
    pd.DataFrame(rows).to_csv(csv, index=False, float_format="%.6g")
    print(f"→ {out}\n→ {csv}")


if __name__ == "__main__":
    main()
