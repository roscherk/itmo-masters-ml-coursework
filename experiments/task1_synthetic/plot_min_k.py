"""Один график ρ̄(k), четыре кривые (LAION и синтетика × роутер и оракул), и минимальное k для ρ̄ ≥ 0,99 (пункт 1.4).

    conda run -n vector-thesis python experiments/task1_synthetic/plot_min_k.py [blur|base]

Данные: results/task1/rho.csv, heldout, balanced k-means N = 16, r = 100, ρ̄ — среднее по всем сидам k-means
(как в rho_ci.py; число сидов — в колонке n_seeds).
Пишет results/task1/rho_four_curves.png и min_k.csv (по каждому датасету, масштабу и роутеру).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from t1lib import RESULTS

VARIANTS = {"blur": ("s10m-fp-blur1.0", "s1m-fp-blur1.0", "cuVS + blur", "#1baf7a", "rho_four_curves", "min_k"),
            "base": ("s10m-fp", "s1m-fp", "cuVS", "#eb6834", "rho_four_curves_base", "min_k_base")}
INK, MUTED, GRID, BG = "#1f1f1e", "#6b6a64", "#e4e3de", "#fcfcfb"
TH = 0.99


def main() -> None:
    import sys
    s10, s1, slab, scol, png, csv = VARIANTS[sys.argv[1] if len(sys.argv) > 1 else "blur"]
    PAIRS = {"10M": ("laion10m", s10), "1M": ("laion1m", s1)}
    LINES = [("LAION", 0, "centroid", "#2a78d6", "-", "o"), ("LAION", 0, "oracle", "#2a78d6", "--", "o"),
             (slab, 1, "centroid", scol, "-", "^"), (slab, 1, "oracle", scol, "--", "^")]
    d = pd.read_csv(RESULTS / "rho.csv")
    d = d[(d.qset == "heldout") & (d.r == 100) & d.partition.str.startswith("bkm16_s")]
    d = d.groupby(["dataset", "router", "k"], as_index=False).agg(rho_mean=("rho_mean", "mean"),
                                                                  n_seeds=("partition", "nunique"))
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=BG, sharey=True)
    out = []
    for a, (scale, names) in zip(ax, PAIRS.items()):
        a.set_facecolor(BG)
        a.grid(True, color=GRID, lw=0.6)
        for lab, i, router, c, ls, mk in LINES:
            s = d[(d.dataset == names[i]) & (d.router == router)].sort_values("k")
            ok = s[s.rho_mean >= TH]
            mk_k = int(ok.k.min()) if len(ok) else None
            out.append((scale, names[i], router, mk_k, s.k.max(), float(s.rho_mean.iloc[-1]), int(s.n_seeds.min())))
            a.plot(s.k, s.rho_mean, color=c, ls=ls, lw=2, marker=mk, ms=6, mec=BG, mew=1.2,
                   label=f"{lab}, {'роутер' if router == 'centroid' else 'оракул'}, k* = {mk_k}")
        a.axhline(TH, color=MUTED, lw=0.8, ls=":")
        a.set_title(f"{scale}: ρ̄(k), N = 16, r = 100, среднее по сидам", color=INK, loc="left")
        a.set_xlabel("k, число опрошенных шардов")
        a.set_xticks(range(1, 17))
        a.legend(frameon=False, fontsize=8, loc="lower right")
    ax[0].set_ylabel("ρ̄")
    fig.tight_layout()
    fig.savefig(RESULTS / f"{png}.png", dpi=150, facecolor=BG)
    df = pd.DataFrame(out, columns=["scale", "dataset", "router", "min_k_rho_ge_0.99", "k_max", "rho_at_kmax", "n_seeds"])
    df.to_csv(RESULTS / f"{csv}.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
