"""Графики «реальные против синтетики» для статистик (пункт 1.3) и таблица.

    conda run -n vector-thesis python experiments/task1_synthetic/plot_stats.py

Пишет results/task1/stats_real_vs_synth.png и stats_table.md. Запросы heldout, 10M.
Нормы: оба датасета единичные (std ~4e-8), поэтому гистограмма нормы вырождена, нормы только в таблице.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from t1lib import RESULTS

SERIES = [("LAION-10M", "laion10m", "#2a78d6"), ("cuVS", "s10m-fp", "#eb6834"),
          ("cuVS + blur", "s10m-fp-blur1.0", "#1baf7a")]
INK, MUTED, GRID, BG = "#1f1f1e", "#6b6a64", "#e4e3de", "#fcfcfb"


def style(a, title, xl, yl):
    a.set_facecolor(BG)
    a.grid(True, color=GRID, lw=0.6)
    a.set_axisbelow(True)
    a.set_title(title, color=INK, loc="left")
    a.set_xlabel(xl)
    a.set_ylabel(yl)


def main() -> None:
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    J = {n: json.loads((RESULTS / "stats" / f"{n}.json").read_text()) for _, n, _ in SERIES}
    Z = {n: np.load(RESULTS / "stats" / f"{n}__heldout.npz") for _, n, _ in SERIES}
    fig, ax = plt.subplots(2, 2, figsize=(10, 8), facecolor=BG)
    for lab, n, c in SERIES:
        lam = np.array(J[n]["global"]["eig"])
        i = np.arange(1, len(lam) + 1)
        ax[0, 0].loglog(i, np.maximum(lam, 1e-12), color=c, lw=1.8, label=lab)
        cum = np.cumsum(lam)
        ax[0, 1].plot(i, cum, color=c, lw=1.8, label=lab)
        for q in (0.5, 0.9):
            ax[0, 1].plot(int(np.searchsorted(cum, q)) + 1, q, "o", color=c, ms=5)
        for a, key in ((ax[1, 0], "d1"), (ax[1, 1], "d100")):
            a.hist(Z[n][key], bins=80, density=True, histtype="step", lw=1.8, color=c, label=lab)
    style(ax[0, 0], "Спектр ковариации (доля дисперсии)", "номер компоненты", "λ / Σλ")
    ax[0, 0].set_ylim(1e-7, 1)
    style(ax[0, 1], "Накопленная дисперсия, точки: 50% и 90%", "число компонент", "доля дисперсии")
    style(ax[1, 0], "Расстояние до 1-го соседа", "d₁ (евклидово)", "плотность")
    style(ax[1, 1], "Расстояние до 100-го соседа", "d₁₀₀ (евклидово)", "плотность")
    ax[0, 0].legend(frameon=False)
    fig.suptitle("LAION-10M против синтетики cuVS, запросы heldout, 10 000 шт", color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(RESULTS / "stats_real_vs_synth.png", dpi=150, facecolor=BG)

    rows = [("норма: среднее", lambda n: f"{J[n]['global']['norm_mean']:.6f}"),
            ("норма: std", lambda n: f"{J[n]['global']['norm_std']:.1e}"),
            ("PR (эфф. размерность)", lambda n: f"{J[n]['global']['pr']:.1f}"),
            ("d50 / d90 / d95", lambda n: "{} / {} / {}".format(*(J[n]['global'][k] for k in ("d50", "d90", "d95")))),
            ("TwoNN", lambda n: f"{J[n]['global']['twonn_id']:.1f}"),
            ("d₁ медиана", lambda n: f"{np.median(Z[n]['d1']):.3f}"),
            ("d₁₀₀ медиана", lambda n: f"{np.median(Z[n]['d100']):.3f}"),
            ("d₁/d₁₀₀ медиана (d₁₀₀ > 0)", lambda n: f"{np.median((Z[n]['d1'] / Z[n]['d100'])[Z[n]['d100'] > 0]):.3f}"),
            ("запросов с d₁₀₀ = 0 (≥ 100 дубликатов)", lambda n: str(int((Z[n]['d100'] == 0).sum()))),
            ("запросов с d₁ = 0", lambda n: str(int((Z[n]['d1'] == 0).sum())))]
    md = "| | " + " | ".join(s[0] for s in SERIES) + " |\n|---|" + "---|" * len(SERIES) + "\n"
    for name, f in rows:
        md += f"| {name} | " + " | ".join(f(n) for _, n, _ in SERIES) + " |\n"
    (RESULTS / "stats_table.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
