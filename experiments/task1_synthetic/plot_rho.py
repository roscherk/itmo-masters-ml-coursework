"""Кривые ρ(q) от k: LAION против синтетики на 1M и 10M, запросы heldout, среднее по всем сидам k-means.

    python experiments/task1_synthetic/rho_ci.py --real laion1m --synth s1m-fp-blur1.0 \
        --gens s1m-fp-blur1.0-g1 s1m-fp-blur1.0-g2 s1m-fp-blur1.0-g3 s1m-fp-blur1.0-g4 --extra s1m-fp s1m-nc10k-fp-blur1.0
    python experiments/task1_synthetic/rho_ci.py --real laion10m --synth s10m-fp-blur1.0 --extra s10m-fp s10m-nc10k-fp-blur1.0
    python experiments/task1_synthetic/plot_rho.py

Данные — results/task1/rho_ci_laion1m.csv и rho_ci_laion10m.csv (их пишет rho_ci.py с --extra для
cuVS и 10k кл. + blur), так что точки и интервалы совпадают с таблицами rho_ci_*.md. Каждая кривая —
среднее по 10 сидам balanced k-means (колонка n в CSV), сиды догнаны logs/iter16.sh

Ряды: ρ̄(k) для роутера по центроидам; Δρ̄ = синтетика − LAION по центроидам; то же для оракула;
потеря роутера ρ̄_oracle − ρ̄_centroid. Полоса — 95% интервал (сиды k-means + запросы) для cuVS + blur,
в нижнем ряду ещё и для LAION. Сиды генератора на графике не показаны: интервал с ними почти тот же
(таблицы rho_ci_*.md с --gens). Пунктир — критерий ±0,02. Пишет results/task1/rho_curves.png
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from t1lib import RESULTS

# категориальные цвета по порядку (палитра dataviz, проверена validate_palette.js), у каждой линии свой маркер
SERIES = [("LAION", "#2a78d6", "o"), ("cuVS", "#eb6834", "s"),
          ("cuVS + blur", "#1baf7a", "^"), ("10k кл. + blur", "#eda100", "D")]
SCALES = {"1M": {"LAION": "laion1m", "cuVS": "s1m-fp", "cuVS + blur": "s1m-fp-blur1.0",
                 "10k кл. + blur": "s1m-nc10k-fp-blur1.0"},
          "10M": {"LAION": "laion10m", "cuVS": "s10m-fp", "cuVS + blur": "s10m-fp-blur1.0",
                  "10k кл. + blur": "s10m-nc10k-fp-blur1.0"}}
BAND = "cuVS + blur"
INK, MUTED, GRID, BG = "#1f1f1e", "#6b6a64", "#e4e3de", "#fcfcfb"
KMAX, Z95, QSET = 8, 1.96, "heldout"


def main() -> None:
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, ax = plt.subplots(4, 2, figsize=(9, 12.5), sharex=True, facecolor=BG)
    mk_kw = dict(lw=2, ms=6, mec=BG, mew=1.5)
    for j, (scale, names) in enumerate(SCALES.items()):
        real = names["LAION"]
        d = pd.read_csv(RESULTS / f"rho_ci_{real}.csv")
        d = d[(d.qset == QSET) & (d.k <= KMAX)]

        def est(ds: str, router: str) -> pd.DataFrame:
            s = d[(d.dataset == ds) & (d.router == router)].sort_values("k")
            if s.empty:
                raise KeyError(f"нет {ds}/{router} в rho_ci_{real}.csv — перезапустить rho_ci.py с --extra")
            return s

        k = est(real, "centroid").k.to_numpy()
        for a in ax[:, j]:
            a.set_facecolor(BG)
            a.grid(True, color=GRID, lw=0.6)
            a.set_axisbelow(True)

        a = ax[0, j]
        for lab, col, mk in SERIES:
            a.plot(k, est(names[lab], "centroid").rho_mean, color=col, marker=mk, label=lab, **mk_kw)
        a.set_title(f"{scale}: ρ̄(k), роутер по центроидам", color=INK, loc="left")
        a.set_ylabel("ρ̄")
        a.legend(frameon=False, fontsize=8, loc="lower right")

        for i, router in ((1, "centroid"), (2, "oracle")):
            a = ax[i, j]
            er = est(real, router)
            a.axhline(0, color=MUTED, lw=0.8)
            for y in (-0.02, 0.02):
                a.axhline(y, color=MUTED, lw=0.8, ls="--")
            es = est(names[BAND], router)
            dlt = es.rho_mean.to_numpy() - er.rho_mean.to_numpy()
            se = np.hypot(es.se.to_numpy(), er.se.to_numpy())
            a.fill_between(k, dlt - Z95 * se, dlt + Z95 * se, color=SERIES[2][1], alpha=0.18, lw=0,
                           label=f"{BAND}: 95% интервал Δ (сиды + запросы)")
            for lab, col, mk in SERIES[1:]:
                a.plot(k, est(names[lab], router).rho_mean.to_numpy() - er.rho_mean.to_numpy(),
                       color=col, marker=mk, label=lab, **mk_kw)
            a.set_title(f"{scale}: Δρ̄ к LAION, {'по центроидам' if router == 'centroid' else 'оракул'}",
                        color=INK, loc="left")
            a.set_ylabel("Δρ̄")
            a.set_ylim(-0.035, 0.06)
            if i == 1 and j == 0:
                h, l = a.get_legend_handles_labels()
                a.legend([x for x, y in zip(h, l) if "интервал" in y], [y for y in l if "интервал" in y],
                         frameon=False, fontsize=8, loc="upper right")

        a = ax[3, j]
        for lab, col, mk in SERIES:
            e = est(names[lab], "gap")
            if lab in ("LAION", BAND):
                a.fill_between(k, e.rho_mean - Z95 * e.se, e.rho_mean + Z95 * e.se, color=col, alpha=0.18, lw=0)
            a.plot(k, e.rho_mean, color=col, marker=mk, label=lab, **mk_kw)
        a.set_title(f"{scale}: потеря роутера ρ̄_oracle − ρ̄_centroid", color=INK, loc="left")
        a.set_ylabel("ρ̄_oracle − ρ̄_centroid")
        a.set_ylim(0, 0.08)
        a.set_xlabel("k, число опрошенных шардов из 16")
        a.set_xticks(range(1, KMAX + 1))
    ax[3, 0].text(0.98, 0.95, "Δ по центроидам = Δ оракула − Δ потери роутера\n"
                  "полоса — 95% интервал для LAION и cuVS + blur",
                  transform=ax[3, 0].transAxes, ha="right", va="top", fontsize=8, color=MUTED)
    fig.suptitle(f"ρ(q) при balanced k-means, N = 16, r = 100, запросы {QSET}, среднее по сидам",
                 color=INK, x=0.01, ha="left")
    fig.tight_layout()
    out = RESULTS / "rho_curves.png"
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    print(f"→ {out}")


if __name__ == "__main__":
    main()
