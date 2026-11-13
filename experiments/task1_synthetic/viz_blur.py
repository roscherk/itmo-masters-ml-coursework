"""Что делает --blur с распределением: картинки к хабам и к оракулу на 10M.

    python experiments/task1_synthetic/viz_blur.py

Скрытые переменные генератора (кластер, t, выбранный соседний центроид) восстанавливаются для каждой строки
базы и каждого запроса heldout без чтения базы: blur берёт числа из своего детерминированного ГСЧ, база пишется
кластер за кластером, отложенные строки выбираются тоже детерминированно (cuvs_synth.cmd_generate). Проверка:
gen_cluster побитно (до 6e-8) совпадает со строками базы s1m-fp-blur1.0.

Картинки в results/task1/:
    blur_geometry.png — кластер LAION, тот же кластер cuVS и cuVS + blur в плоскости соседних центроидов;
                        тугой кластер, вытянутый в иглы, с хабами на них
    blur_hubs.png     — распределение N₁₀ на равномерной выборке 50k; N₁₀ от разброса своего кластера
    blur_oracle.png   — оракул при k = 1 на 1M и 10M и его связь с долей кластера генератора в одном шарде
Числа печатаются в stdout
"""

from __future__ import annotations

import json
import sys

import numpy as np

import cuvs_synth as cs
from stats import knn_self
from t1lib import RESULTS, ROOT, ds_path, gather, load_qset, open_base, random_rows, set_threads

N_HUB = 50_000
SEEDS = range(10)
LAION, CUVS, BLUR = "#2a78d6", "#eb6834", "#1baf7a"      # порядок и цвета как в plot_rho.py
INK, MUTED, GRID, BG = "#1f1f1e", "#6b6a64", "#e4e3de", "#fcfcfb"


def fingerprint(name: str) -> tuple[cs.Fingerprint, dict]:
    meta = json.loads((ds_path(name) / "meta.json").read_text())
    fp = cs.load_fingerprint(ROOT / meta["fingerprint"], seed=meta["seed"])
    fp.tail, fp.tail_mult = meta.get("tail", "fp"), meta.get("tail_mult", 1.0)
    fp.var_scale = meta.get("var_scale", True)
    fp.blur, fp.blur_k = meta.get("blur", 0.0), meta.get("blur_k", 8)
    fp.blur_mix = float(meta.get("blur_mix", 0.0))      # в старых meta.json — True/False
    return fp, meta


def hidden(name: str) -> dict:
    """Кластер генератора, t и соседний центроид для строк базы (b_*) и запросов heldout (q_*)"""
    fp, meta = fingerprint(name)
    n_gen = meta["n"] + meta["nq"]
    excl = np.sort(np.random.default_rng(fp.seed + 2 * cs.QUERY_SEED_OFFSET).choice(n_gen, meta["nq"], replace=False))
    ppc = cs.get_num_points_per_cluster(n_gen, fp)
    cl = np.repeat(np.arange(fp.nclusters), ppc)
    t = np.zeros(n_gen, np.float32)
    nb = np.full(n_gen, -1)
    if fp.blur > 0:
        nbr = cs.cluster_neighbors(fp)
        s = 0
        for c in range(fp.nclusters):      # те же вызовы ГСЧ в том же порядке, что в gen_cluster
            m = int(ppc[c])
            brng = np.random.default_rng(cs.get_cluster_seed(fp.seed, c) + 7_777_777)
            nb[s:s + m] = nbr[c][brng.integers(0, len(nbr[c]), m)]
            t[s:s + m] = brng.random((m, 1), dtype=np.float32)[:, 0] * np.float32(fp.blur)
            s += m
    keep = np.ones(n_gen, bool)
    keep[excl] = False
    qg = excl[np.random.default_rng(fp.seed).permutation(meta["nq"])]
    return {"fp": fp, "ppc": ppc, "excl": excl, "b_cl": cl[keep], "b_t": t[keep], "b_nb": nb[keep],
            "q_cl": cl[qg], "q_t": t[qg]}


def check_bitwise(name: str, h: dict, c: int = 0) -> float:
    fp, ppc, excl = h["fp"], h["ppc"], h["excl"]
    starts = np.concatenate([[0], np.cumsum(ppc)])
    keep = np.ones(starts[-1], bool)
    keep[excl] = False
    row_of = np.cumsum(keep) - 1
    g = np.arange(starts[c], starts[c + 1])
    g = g[keep[g]]
    pts = cs.gen_cluster(c, int(ppc[c]), fp)
    return float(np.abs(pts[g - starts[c]] - np.asarray(open_base(name)[row_of[g]])).max())


def n10(X: np.ndarray) -> np.ndarray:
    _, I = knn_self(X, 10)
    return np.bincount(I.ravel(), minlength=len(X)).astype(float)


def skew(N: np.ndarray) -> float:
    return float(((N - N.mean()) ** 3).mean() / N.std() ** 3)


def style(plt) -> None:
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": BG, "axes.facecolor": BG,
                         "axes.titlelocation": "left", "axes.titlesize": 9.5, "legend.frameon": False})


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def fig_geometry(plt, fp: cs.Fingerprint, c: int, tight: int, laion_lab: np.ndarray, laion_base) -> None:
    """Плоскость: две главные оси звезды «центроид c → 8 соседей», центр в c. Всё на единичной сфере"""
    nbr = cs.cluster_neighbors(fp)
    Cn = unit(fp.cluster_centers.astype(np.float64))

    def basis(k):
        A = Cn[nbr[k]] - Cn[k]
        return np.linalg.svd(A, full_matrices=False)[2][:2]

    def gen(k, blur):
        fp.blur = blur
        return cs.gen_cluster(k, int(PPC1M[k]), fp).astype(np.float64)

    fig, ax = plt.subplots(2, 3, figsize=(12, 7.6))
    P = basis(c)
    proj = lambda X, k=c: (X - Cn[k]) @ P.T            # noqa: E731
    star = proj(Cn[nbr[c]])
    real = np.asarray(laion_base[np.where(laion_lab == c)[0]], dtype=np.float64)
    x0, x1 = gen(c, 0.0), gen(c, 1.0)
    fp.blur = 1.0                                       # настоящие t точек: тот же ГСЧ, что в gen_cluster
    brng = np.random.default_rng(cs.get_cluster_seed(fp.seed, c) + 7_777_777)
    brng.integers(0, fp.blur_k, len(x1))
    t1 = brng.random((len(x1), 1), dtype=np.float32)[:, 0]
    lim = np.abs(np.vstack([star, proj(x1)])).max() * 1.08
    for a, X, col, title in ((ax[0, 0], real, LAION, f"LAION-1M: точки кластера {c} ({len(real)})"),
                             (ax[0, 1], x0, CUVS, f"cuVS: кластер {c}, гаусс ранга 32"),
                             (ax[0, 2], x1, None, f"cuVS + blur 1.0: тот же кластер, цвет — t")):
        Y = proj(X)
        if col:
            a.scatter(Y[:, 0], Y[:, 1], s=4, c=col, alpha=0.5, lw=0)
        else:
            sc = a.scatter(Y[:, 0], Y[:, 1], s=4, c=t1, cmap="viridis", vmin=0, vmax=1, alpha=0.7, lw=0)
            fig.colorbar(sc, ax=a, fraction=0.04, pad=0.02, label="t")
        a.scatter(star[:, 0], star[:, 1], marker="x", s=40, c=INK, lw=1.2, label="8 соседних центроидов")
        a.scatter([0], [0], marker="+", s=80, c=INK, lw=1.5)
        a.set_xlim(-lim, lim)
        a.set_ylim(-lim, lim)
        a.set_aspect("equal")
        a.grid(True, color=GRID, lw=0.6)
        a.set_title(title)
    ax[0, 0].legend(loc="lower left", fontsize=8)

    # расстояние до своего центроида: в 768-d плоская картинка обманывает, это честная проекция «по радиусу»
    a = ax[1, 0]
    bins = np.linspace(0, max(np.linalg.norm(x1 - Cn[c], axis=1).max(), np.linalg.norm(real - Cn[c], axis=1).max()), 50)
    for X, col, lab in ((real, LAION, "LAION"), (x0, CUVS, "cuVS"), (x1, BLUR, "cuVS + blur")):
        a.hist(np.linalg.norm(X - Cn[c], axis=1), bins=bins, histtype="step", lw=2, color=col, label=lab, density=True)
    a.set_title(f"кластер {c}: ‖x − центроид‖ в 768-d")
    a.set_xlabel("расстояние до центроида")
    a.legend(fontsize=8)
    a.grid(True, color=GRID, lw=0.6)

    # тугой кластер: точки почти совпадают, blur вытягивает их в 8 игл, иглы протыкают соседей и становятся хабами
    P2 = basis(tight)
    proj2 = lambda X: (X - Cn[tight]) @ P2.T            # noqa: E731
    xt = gen(tight, 1.0)
    others = np.vstack([gen(k, 1.0) for k in nbr[tight]])
    U = np.vstack([xt, others])
    N = n10(U.astype(np.float32))
    nt = len(xt)
    for a, (lab, show) in zip(ax[1, 1:], (("все точки", None), ("N₁₀ точек иглы", True))):
        Y = proj2(others)
        a.scatter(Y[:, 0], Y[:, 1], s=3, c="#b9b8b2", alpha=0.35, lw=0, label="8 соседних кластеров (с blur)")
        Yt = proj2(xt)
        if show:
            o = np.argsort(N[:nt])
            sc = a.scatter(Yt[o, 0], Yt[o, 1], s=6, c=N[:nt][o], cmap="magma_r", norm=plt.matplotlib.colors.LogNorm(1, N.max()), lw=0)
            fig.colorbar(sc, ax=a, fraction=0.04, pad=0.02, label="N₁₀ (сколько точек считают её соседом)")
            a.set_title(f"N₁₀ точек иглы: до {N[:nt].max():.0f}; у иглы {N[:nt].sum() / N.sum():.0%} рёбер kNN при {nt / len(N):.0%} точек")
        else:
            a.scatter(Yt[:, 0], Yt[:, 1], s=5, c=BLUR, lw=0, label=f"тугой кластер {tight}")
            a.set_title(f"тугой кластер {tight}: разброс {fp.cluster_variances[tight].sum():.3f} "
                         f"(медиана {np.median(fp.cluster_variances.sum(1)):.2f})")
            a.legend(loc="lower left", fontsize=8, markerscale=3)
        st = proj2(Cn[nbr[tight]])
        a.scatter(st[:, 0], st[:, 1], marker="x", s=40, c=INK, lw=1.2)
        a.set_aspect("equal")
        a.grid(True, color=GRID, lw=0.6)
    print(f"иглы кластера {tight}: медиана N₁₀ {np.median(N[:nt]):.0f}, max {N[:nt].max():.0f}; "
          f"соседние кластеры: медиана {np.median(N[nt:]):.0f}; доля рёбер в иглу {N[:nt].sum() / N.sum():.2f} при доле точек {nt / len(N):.2f}")
    fig.suptitle("blur 1.0: каждая точка сдвигается к одному из 8 соседних центроидов на t·(μ_c′ − μ_c), t ~ U(0, 1)",
                 x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(RESULTS / "blur_geometry.png", dpi=150)
    plt.close(fig)


def fig_hubs(plt, hubs: dict, h1: dict) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    a = ax[0]
    for name, col, lab in (("laion1m", LAION, "LAION"), ("s1m-fp", CUVS, "cuVS"), ("s1m-fp-blur1.0", BLUR, "cuVS + blur")):
        N = hubs[name]["N"]
        x = np.sort(N)
        a.step(x, 1 - np.arange(len(x)) / len(x), color=col, lw=2, where="post", label=f"{lab}: скошенность {skew(N):.1f}")
    a.set_xscale("symlog", linthresh=10)
    a.set_yscale("log")
    a.set_xlabel("N₁₀ точки")
    a.set_ylabel("доля точек с N₁₀ ≥ x")
    a.set_title("1M, равномерная выборка 50k: хвост N₁₀")
    a.legend(fontsize=8)
    a.grid(True, color=GRID, lw=0.6)

    a = ax[1]
    fp = h1["fp"]
    tv = fp.cluster_variances.sum(1)
    b = hubs["s1m-fp-blur1.0"]
    v = tv[h1["b_cl"][b["rows"]]]
    edges = np.quantile(tv, np.linspace(0, 1, 11))
    idx = np.clip(np.searchsorted(edges, v, side="right") - 1, 0, 9)
    mean = [b["N"][idx == i].mean() for i in range(10)]
    p99 = [np.quantile(b["N"][idx == i], 0.99) for i in range(10)]
    xs = np.arange(10)
    a.bar(xs - 0.2, mean, 0.38, color=BLUR, label="среднее N₁₀")
    a.bar(xs + 0.2, p99, 0.38, color="#0d6b49", label="99-й процентиль N₁₀")
    a.set_xticks(xs, [f"{edges[i]:.2f}" for i in range(10)], rotation=45)
    a.set_xlabel("разброс своего кластера генератора (децили, нижняя граница)")
    a.set_title("cuVS + blur: хабы дают тугие кластеры")
    a.legend(fontsize=8)
    a.grid(True, axis="y", color=GRID, lw=0.6)
    fig.tight_layout()
    fig.savefig(RESULTS / "blur_hubs.png", dpi=150)
    plt.close(fig)


def oracle_stats(name: str, h: dict | None) -> dict:
    _, gt, _ = load_qset(name, "heldout")
    o1, pur = [], []
    for s in SEEDS:
        lab = np.load(ds_path(name) / "partitions" / f"bkm16_s{s}.npz")["labels"]
        L = lab[gt[:, :100]]
        o1.append(np.array([np.bincount(x, minlength=16).max() for x in L]) / 100)
        if h is not None:
            M = np.zeros((h["fp"].nclusters, 16))
            np.add.at(M, (h["b_cl"], lab), 1)
            pur.append((M.max(1) / M.sum(1))[h["q_cl"]])
    out = {"o1": np.mean(o1, 0)}
    if h is not None:
        out["pur"] = np.mean(pur, 0)
        out["own"] = (h["b_cl"][gt[:, :100]] == h["q_cl"][:, None]).mean(1)
    return out


def fig_oracle(plt, st: dict) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1.3, 1]})
    a = ax[0]
    groups = (("1M", "laion1m", "s1m-fp", "s1m-fp-blur1.0"), ("10M", "laion10m", "s10m-fp", "s10m-fp-blur1.0"))
    w = 0.16
    for g, (scale, *names) in enumerate(groups):
        bars = [(names[0], "o1", LAION, False), (names[1], "o1", CUVS, False), (names[1], "pur", CUVS, True),
                (names[2], "o1", BLUR, False), (names[2], "pur", BLUR, True)]
        for j, (n, key, col, hatch) in enumerate(bars):
            x = g + (j - 2) * w
            y = st[n][key].mean()
            a.bar(x, y, w * 0.92, color=col, alpha=0.45 if hatch else 1, hatch="//" if hatch else None, edgecolor=BG)
            a.text(x, y + 0.004, f"{y:.3f}", ha="center", fontsize=7)
    a.set_xticks([0, 1], ["1M", "10M"])
    a.set_ylim(0.6, 0.85)
    for n, col, lab in ((0, LAION, "LAION"), (1, CUVS, "cuVS"), (2, BLUR, "cuVS + blur")):
        a.bar([0], [0], color=col, label=lab)
    a.bar([0], [0], color=MUTED, alpha=0.45, hatch="//", edgecolor=BG, label="доля кластера генератора в главном шарде")
    a.legend(fontsize=7.5, loc="upper left", ncol=2)
    a.set_title("оракул k = 1 (сплошные), 10 сидов: blur снимает 0,026 на обоих масштабах;\n"
                "на 1M это гасит завышение cuVS (+0,021), на 10M завышения почти нет (+0,007)")
    a.grid(True, axis="y", color=GRID, lw=0.6)

    a = ax[1]
    bins = np.linspace(0.3, 1.0, 8)
    for n, col, ls, lab in (("s10m-fp", CUVS, "-", "cuVS 10M"), ("s10m-fp-blur1.0", BLUR, "-", "cuVS + blur 10M"),
                            ("s1m-fp-blur1.0", BLUR, ":", "cuVS + blur 1M")):
        p, o = st[n]["pur"], st[n]["o1"]
        idx = np.digitize(p, bins)
        sel = [i for i in range(1, len(bins)) if (idx == i).sum() > 30]
        a.plot([p[idx == i].mean() for i in sel], [o[idx == i].mean() for i in sel], color=col, lw=2, ls=ls,
               marker="o", ms=4, label=f"{lab}: corr {np.corrcoef(p, o)[0, 1]:.2f}, в своём кластере {st[n]['own'].mean():.0%} 100-NN")
    a.plot([0.3, 1], [0.3, 1], color=MUTED, lw=0.8, ls="--")
    a.set_xlabel("доля кластера генератора запроса в главном шарде")
    a.set_ylabel("оракул k = 1 (доля 100-NN в лучшем шарде)")
    a.set_title("оракул идёт за тем, как шарды режут\nкластер генератора запроса")
    a.legend(fontsize=7.5, loc="upper left")
    a.grid(True, color=GRID, lw=0.6)
    fig.tight_layout()
    fig.savefig(RESULTS / "blur_oracle.png", dpi=150)
    plt.close(fig)


PPC1M = None


def main() -> None:
    global PPC1M
    import faiss
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style(plt)
    set_threads()

    h1 = hidden("s1m-fp-blur1.0")
    PPC1M = h1["ppc"]
    print(f"s1m-fp-blur1.0: gen_cluster против базы, max |Δ| {check_bitwise('s1m-fp-blur1.0', h1):.1e}")

    # хабы: в stats.py N₁₀ берётся по первой половине отсортированной выборки, а база синтетики идёт кластер
    # за кластером — выборка покрывает половину кластеров. Здесь выборка равномерная
    hubs = {}
    for name in ("laion1m", "s1m-fp", "s1m-fp-blur1.0"):
        B = open_base(name)
        rows = random_rows(B.shape[0], N_HUB, 123)
        hubs[name] = {"rows": rows, "N": n10(gather(B, rows))}
        pre = random_rows(B.shape[0], 100_000, 0)[:N_HUB]
        print(f"{name:16s} скошенность N₁₀: равномерно {skew(hubs[name]['N']):.2f}, как в stats.py {skew(n10(gather(B, pre))):.2f}")
    fig_hubs(plt, hubs, h1)

    fp = h1["fp"]
    tv = fp.cluster_variances.sum(1)
    ok = np.array([fp.pca_components_list[k] is not None for k in range(fp.nclusters)])
    typical = int(np.where(ok)[0][np.argmin(np.abs(tv[ok] - np.median(tv)) + 1e9 * (np.abs(PPC1M[ok] - 1000) > 200))])
    top = np.argsort(-hubs["s1m-fp-blur1.0"]["N"])[0]
    tight = int(h1["b_cl"][hubs["s1m-fp-blur1.0"]["rows"][top]])
    LB = open_base("laion1m")
    C = np.ascontiguousarray(fp.cluster_centers, np.float32)
    lab = np.concatenate([faiss.knn(np.ascontiguousarray(LB[s:s + 200_000]), C, 1)[1][:, 0]
                          for s in range(0, LB.shape[0], 200_000)])
    print(f"кластеры для картинки: типичный {typical}, тугой {tight} (у него главный хаб выборки)")
    fig_geometry(plt, fp, typical, tight, lab, LB)

    st = {n: oracle_stats(n, None) for n in ("laion1m", "laion10m")}
    for name in ("s1m-fp", "s1m-fp-blur1.0", "s10m-fp", "s10m-fp-blur1.0"):
        st[name] = oracle_stats(name, hidden(name))
    for n, v in st.items():
        extra = f", доля кластера в главном шарде {v['pur'].mean():.3f}, 100-NN из своего кластера {v['own'].mean():.2f}" if "pur" in v else ""
        print(f"{n:16s} оракул k = 1 {v['o1'].mean():.4f}{extra}")
    fig_oracle(plt, st)
    print("→", RESULTS / "blur_geometry.png", RESULTS / "blur_hubs.png", RESULTS / "blur_oracle.png")


if __name__ == "__main__":
    sys.exit(main())
