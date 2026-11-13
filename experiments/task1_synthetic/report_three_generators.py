"""Create isolated tables and plots for the three-generator pilot."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "task1" / "three_generators"
NAMES = ["tg1m-perturb-heldout-v2", "tg1m-gmm-heldout-v2", "tg1m-cuvs-heldout-v2"]
LABELS = ["Perturbation baseline", "Diagonal GMM", "cuVS"]


def read_csv(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    quant = read_csv(OUT / "quant.csv")
    fields = list(quant[0])
    with (OUT / "quant_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(quant)

    methods = [f"rabitq-b{i}" for i in range(1, 9)]
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for name, label in zip(NAMES, LABELS):
        rows = {r["method"]: r for r in quant if r["dataset"] == name}
        x = np.arange(1, 9)
        y = np.array([float(rows[m]["recall100"]) for m in methods])
        se = np.array([float(rows[m]["recall100_se"]) for m in methods])
        ax.plot(x, y, marker="o", label=label)
        ax.fill_between(x, y - 1.96 * se, y + 1.96 * se, alpha=0.15)
    ax.axhline(0.95, color="gray", linewidth=0.8, linestyle="--")
    ax.axhline(0.99, color="black", linewidth=0.8, linestyle=":")
    ax.set(xticks=range(1, 9), xlabel="RaBitQ bits per dimension", ylabel="Recall@100 (codes)",
           ylim=(0.3, 1.005))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "rabitq_recall.png", dpi=160)
    plt.close(fig)

    rho = read_csv(OUT / "rho.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, router in zip(axes, ("centroid", "oracle")):
        for name, label in zip(NAMES, LABELS):
            rows = [r for r in rho if r["dataset"] == name and r["router"] == router]
            rows.sort(key=lambda r: int(r["k"]))
            ax.plot([int(r["k"]) for r in rows], [float(r["rho_mean"]) for r in rows],
                    marker=".", label=label)
        ax.set(title=router.capitalize(), xlabel="Shards queried", xticks=[1, 2, 4, 8, 12, 16])
    axes[0].set_ylabel("Routing recall rho(q)")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "rho_curves.png", dpi=160)
    plt.close(fig)

    stats = {}
    for name, label in zip(NAMES, LABELS):
        stats[label] = json.loads((OUT / "stats" / f"{name}.json").read_text())
    fields = ["norm_mean", "mean_norm", "mean_cos", "pr", "d50", "d90", "d95",
              "twonn_id", "hub_skew10", "lid100", "rc100", "loc_d90", "dup1", "pur256", "pur1024"]
    with (OUT / "stats_summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", *LABELS])
        for metric in fields:
            values = []
            for label in LABELS:
                doc = stats[label]
                if metric in doc["global"]:
                    v = doc["global"][metric]
                else:
                    v = doc["qsets"]["laion-heldout"][metric]["median"]
                values.append(v)
            w.writerow([metric, *values])

    with (OUT / "routing_summary.csv").open("w", newline="") as f:
        rho = read_csv(OUT / "rho.csv")
        w = csv.writer(f)
        w.writerow(["dataset", "router", "k", "rho_mean", "rho_se_query"])
        for name in NAMES:
            for router in ("centroid", "oracle"):
                for r in sorted((x for x in rho if x["dataset"] == name and x["router"] == router),
                                key=lambda x: int(x["k"])):
                    w.writerow([name, router, r["k"], r["rho_mean"], ""])


if __name__ == "__main__":
    main()
