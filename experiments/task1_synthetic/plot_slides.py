"""Графики для слайдов: кривые ρ(q), recall@100 по битам RaBitQ и диагностические бары.

    python experiments/task1_synthetic/plot_slides.py

Пишет SVG прямо в slides/public/ (палитра слайдов: фон #121212, амбер #F2B84B, зелёный #6BC075,
графит #6E675F, текст #fffefb). Данные — результаты пилота results/task1/three_generators/
(1M, три генератора, общий набор из 1000 запросов LAION-heldout) и эталон LAION 1M из таблиц
checkpoints/02-table.md (единственный сид k-means bkm16_s0).

    slides/public/13-rho-curves-1m.svg    ρ̄(k), роутер по центроидам и оракул, плюс Δρ̄ к LAION
    slides/public/14-rabitq-bits-1m.svg   recall@100 по кодам от числа бит
    slides/public/15-stats-bars-1m.svg    LID₁₀₀, loc_d90, хабы N₁₀, C2ST
"""

from __future__ import annotations

import csv

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "task1" / "three_generators"
CHECKPOINT = ROOT / "experiments" / "task1_synthetic" / "checkpoints" / "02-table.md"
OUT = ROOT / "slides" / "public"

BG, TEXT, MUTED = "#121212", "#fffefb", "#6E675F"
AMBER, GREEN = "#F2B84B", "#6BC075"
FONT = "Inter, system-ui, -apple-system, sans-serif"

# ряды: цвет, тип линии, маркер
SERIES = {
    "LAION": (AMBER, "dash", "circle"),
    "perturbation": (MUTED, "solid", "square"),
    "GMM": (TEXT, "solid", "triangle"),
    "cuVS": (GREEN, "solid", "diamond"),
}
DASH = {"solid": None, "dash": "9 6"}


# ---------------------------------------------------------------------------
# данные
# ---------------------------------------------------------------------------

DATASETS = {
    "perturbation": "tg1m-perturb-heldout-v2",
    "GMM": "tg1m-gmm-heldout-v2",
    "cuVS": "tg1m-cuvs-heldout-v2",
}


def read_md_table(path: Path) -> dict[str, dict[str, str]]:
    """Таблица из markdown-файла чекпоинта: {метрика: {колонка: значение}}"""
    out: dict[str, dict[str, str]] = {}
    header: list[str] | None = None
    for line in path.read_text().splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells[0] == "метрика":
            header = cells
            continue
        if header is None or set(cells[0]) <= set("-: "):
            continue
        out[cells[0]] = dict(zip(header[1:], cells[1:]))
    return out


def md_float(raw: str) -> float:
    return float(raw.split("/")[0].strip().replace(",", "."))


def laion_reference() -> dict:
    """Эталон LAION 1M из 02-table.md: ρ по обоим роутерам и RaBitQ 1–8 бит"""
    tab = read_md_table(CHECKPOINT)
    rho = {}
    for router in ("centroid", "oracle"):
        for k in (1, 2, 3, 4, 8):
            rho[(router, k)] = md_float(tab[f"{router} k={k}"]["laion1m"])
    quant = {}
    for b in range(1, 9):
        key = next(k for k in tab if k.startswith(f"rabitq-b{b} ("))
        quant[b] = md_float(tab[key]["laion1m"])
    stats = {
        "lid100": md_float(tab["lid100"]["laion1m"]),
        "loc_d90": md_float(tab["loc_d90"]["laion1m"]),
        "hub_skew10": md_float(tab["hub_skew10"]["laion1m"]),
    }
    return {"rho": rho, "quant": quant, "stats": stats}


def generator_rho() -> dict[str, dict[str, dict[int, float]]]:
    out = {name: {"centroid": {}, "oracle": {}} for name in DATASETS}
    with open(RESULTS / "rho.csv", newline="") as f:
        for row in csv.DictReader(f):
            for name, ds in DATASETS.items():
                if row["dataset"] == ds:
                    out[name][row["router"]][int(row["k"])] = float(row["rho_mean"])
    return out


def generator_quant() -> dict[str, dict[int, float]]:
    out = {name: {} for name in DATASETS}
    with open(RESULTS / "quant_summary.csv", newline="") as f:
        for row in csv.DictReader(f):
            if not row["method"].startswith("rabitq-b"):
                continue
            for name, ds in DATASETS.items():
                if row["dataset"] == ds:
                    out[name][int(float(row["bits_per_dim"]))] = float(row["recall100"])
    return out


def generator_stats() -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {name: {} for name in DATASETS}
    # в stats_summary.csv колонки названы человекочитаемо
    columns = {"perturbation": "Perturbation baseline", "GMM": "Diagonal GMM", "cuVS": "cuVS"}
    with open(RESULTS / "stats_summary.csv", newline="") as f:
        for row in csv.DictReader(f):
            for name, col in columns.items():
                out[name][row["metric"]] = float(row[col])
    with open(RESULTS / "c2st.csv", newline="") as f:
        for row in csv.DictReader(f):
            for name, ds in DATASETS.items():
                if row["dataset"] == ds:
                    out[name]["c2st_1nn"] = float(row["c2st_1nn"])
                    out[name]["fd_rel"] = float(row["fd_rel"])
    return out


# ---------------------------------------------------------------------------
# мини-движок SVG: панель с осями, линии с маркерами, подписи
# ---------------------------------------------------------------------------

class Panel:
    def __init__(self, x, y, w, h, xlim, ylim, xticks, yticks, *, title="", yfmt="{:.2f}",
                 grid=True):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.xlim, self.ylim = xlim, ylim
        self.xticks, self.yticks = xticks, yticks
        self.title, self.yfmt, self.grid = title, yfmt, grid
        self.parts: list[str] = []
        if grid:
            for t in yticks:
                self.parts.append(
                    f'<line x1="{self.x:.1f}" y1="{self.sy(t):.1f}" x2="{self.x + self.w:.1f}" '
                    f'y2="{self.sy(t):.1f}" stroke="{TEXT}" stroke-opacity="0.07" stroke-width="1"/>')
        if title:
            self.parts.append(self.text(x + 2, y - 9, title, size=15, colour=TEXT, opacity=0.72))
        for t in yticks:
            self.parts.append(self.text(x - 10, self.sy(t) + 4.5, yfmt.format(t), size=14,
                                        colour=TEXT, opacity=0.55, anchor="end"))

    # координаты
    def sx(self, v: float) -> float:
        lo, hi = self.xlim
        return self.x + (v - lo) / (hi - lo) * self.w

    def sy(self, v: float) -> float:
        lo, hi = self.ylim
        return self.y + self.h - (v - lo) / (hi - lo) * self.h

    # элементы
    def text(self, x, y, s, *, size=14, colour=TEXT, opacity=1.0, anchor="start", weight="400"):
        self.parts.append(f'<text x="{x:.1f}" y="{y:.1f}" fill="{colour}" fill-opacity="{opacity}" '
                          f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}">{s}</text>')
        return ""

    def band(self, lo, hi, colour=AMBER, opacity=0.14):
        y1, y2 = self.sy(hi), self.sy(lo)
        self.parts.append(f'<rect x="{self.x:.1f}" y="{y1:.1f}" width="{self.w:.1f}" '
                          f'height="{y2 - y1:.1f}" fill="{colour}" fill-opacity="{opacity}"/>')

    def hline(self, v, *, colour=AMBER, opacity=0.85, width=1.6, dash="7 5"):
        self.parts.append(
            f'<line x1="{self.x:.1f}" y1="{self.sy(v):.1f}" x2="{self.x + self.w:.1f}" '
            f'y2="{self.sy(v):.1f}" stroke="{colour}" stroke-opacity="{opacity}" '
            f'stroke-width="{width}" stroke-dasharray="{dash}"/>')

    def frame(self):
        self.parts.append(
            f'<line x1="{self.x:.1f}" y1="{self.y + self.h:.1f}" x2="{self.x + self.w:.1f}" '
            f'y2="{self.y + self.h:.1f}" stroke="{MUTED}" stroke-opacity="0.75" stroke-width="1.2"/>'
            f'<line x1="{self.x:.1f}" y1="{self.y:.1f}" x2="{self.x:.1f}" '
            f'y2="{self.y + self.h:.1f}" stroke="{MUTED}" stroke-opacity="0.75" stroke-width="1.2"/>')

    def xtick_labels(self, dy=20, every=1, label=None, label_offset=22):
        for t in self.xticks:
            if t % every:
                continue
            self.parts.append(self.text(self.sx(t), self.y + self.h + dy, str(t), size=14,
                                        colour=TEXT, opacity=0.55, anchor="middle"))
        if label:
            self.parts.append(self.text(self.x + self.w, self.y + self.h + dy + label_offset,
                                        label, size=14, colour=TEXT, opacity=0.55, anchor="end"))

    def line(self, series: str, points: list[tuple[float, float]], width=2.4):
        colour, kind, _ = SERIES[series]
        opacity = 0.8 if series == "GMM" else 1.0
        d = " ".join(f"{'M' if i == 0 else 'L'} {self.sx(px):.1f} {self.sy(py):.1f}"
                     for i, (px, py) in enumerate(points))
        dash = DASH[kind]
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<path d="{d}" fill="none" stroke="{colour}" stroke-opacity="{opacity}" '
                          f'stroke-width="{width}" stroke-linecap="round"{dash_attr}/>')

    def markers(self, series: str, points: list[tuple[float, float]], size=5.0):
        colour, _, shape = SERIES[series]
        opacity = 0.8 if series == "GMM" else 1.0
        for px, py in points:
            cx, cy = self.sx(px), self.sy(py)
            if shape == "circle":
                self.parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{size}" fill="{colour}" '
                                  f'fill-opacity="{opacity}" stroke="{BG}" stroke-width="1.6"/>')
            elif shape == "square":
                self.parts.append(f'<rect x="{cx - size:.1f}" y="{cy - size:.1f}" width="{2 * size:.1f}" '
                                  f'height="{2 * size:.1f}" rx="1.5" fill="{colour}" '
                                  f'stroke="{BG}" stroke-width="1.6"/>')
            elif shape == "diamond":
                self.parts.append(
                    f'<path d="M {cx:.1f} {cy - size * 1.35:.1f} L {cx + size * 1.35:.1f} {cy:.1f} '
                    f'L {cx:.1f} {cy + size * 1.35:.1f} L {cx - size * 1.35:.1f} {cy:.1f} Z" '
                    f'fill="{colour}" stroke="{BG}" stroke-width="1.6"/>')
            else:  # triangle
                self.parts.append(
                    f'<path d="M {cx:.1f} {cy - size * 1.3:.1f} L {cx + size * 1.2:.1f} {cy + size:.1f} '
                    f'L {cx - size * 1.2:.1f} {cy + size:.1f} Z" fill="{colour}" fill-opacity="{opacity}" '
                    f'stroke="{BG}" stroke-width="1.6"/>')

    def bar(self, cx: float, value: float, width: float, colour: str, *, opacity=1.0):
        y = self.sy(value)
        self.parts.append(f'<rect x="{cx - width / 2:.1f}" y="{y:.1f}" width="{width:.1f}" '
                          f'height="{self.y + self.h - y:.1f}" rx="3" fill="{colour}" '
                          f'fill-opacity="{opacity}"/>')

    def render(self) -> str:
        return "\n  ".join(self.parts)


def legend(items: list[tuple[str, str]], y: float, width: float) -> str:
    """Ряд легенды по центру: [линия с маркером + подпись] для каждого ряда"""
    widths = []
    for name, label in items:
        widths.append(26 + 8.4 * len(label))
    total = sum(widths) + 34 * (len(items) - 1)
    x = (width - total) / 2
    parts = []
    for (name, label), w in zip(items, widths):
        colour, kind, shape = SERIES[name]
        opacity = 0.8 if name == "GMM" else 1.0
        dash = DASH[kind]
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x + 20:.1f}" y2="{y:.1f}" '
                     f'stroke="{colour}" stroke-opacity="{opacity}" stroke-width="2.4" '
                     f'stroke-linecap="round"{dash_attr}/>')
        cx = x + 10
        if shape == "circle":
            parts.append(f'<circle cx="{cx:.1f}" cy="{y:.1f}" r="4.2" fill="{colour}" '
                         f'stroke="{BG}" stroke-width="1.4"/>')
        elif shape == "square":
            parts.append(f'<rect x="{cx - 4:.1f}" y="{y - 4:.1f}" width="8" height="8" rx="1.5" '
                         f'fill="{colour}" stroke="{BG}" stroke-width="1.4"/>')
        elif shape == "diamond":
            parts.append(f'<path d="M {cx:.1f} {y - 5.6:.1f} L {cx + 5.2:.1f} {y:.1f} '
                         f'L {cx:.1f} {y + 5.6:.1f} L {cx - 5.2:.1f} {y:.1f} Z" fill="{colour}" '
                         f'stroke="{BG}" stroke-width="1.4"/>')
        else:
            parts.append(f'<path d="M {cx:.1f} {y - 5:.1f} L {cx + 4.8:.1f} {y + 4:.1f} '
                         f'L {cx - 4.8:.1f} {y + 4:.1f} Z" fill="{colour}" fill-opacity="{opacity}" '
                         f'stroke="{BG}" stroke-width="1.4"/>')
        parts.append(f'<text x="{x + 26:.1f}" y="{y + 5:.1f}" fill="{TEXT}" fill-opacity="0.8" '
                     f'font-size="15">{label}</text>')
        x += w + 34
    return "\n  ".join(parts)


def svg(width: int, height: int, title: str, desc: str, body: str) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" '
            f'height="100%" role="img" aria-labelledby="title desc" font-family="{FONT}">\n'
            f'  <title id="title">{title}</title>\n  <desc id="desc">{desc}</desc>\n'
            f'  <rect width="{width}" height="{height}" rx="16" fill="{BG}"/>\n  {body}\n</svg>\n')


# ---------------------------------------------------------------------------
# 13: ρ̄(k) — роутер по центроидам и оракул, плюс Δρ̄ к LAION
# ---------------------------------------------------------------------------

def fig_rho() -> Path:
    ref = laion_reference()["rho"]
    rho = generator_rho()
    ks = list(range(1, 9))
    ref_ks = [1, 2, 3, 4, 8]

    W, H = 980, 512
    panels = {
        ("centroid", "rho"): Panel(62, 56, 418, 150, (0.8, 8.2), (0.66, 1.01), ks,
                                   [0.7, 0.8, 0.9, 1.0], title="роутер по центроидам"),
        ("oracle", "rho"): Panel(540, 56, 418, 150, (0.8, 8.2), (0.73, 1.01), ks,
                                 [0.75, 0.85, 0.95, 1.0], title="оракул"),
        ("centroid", "d"): Panel(62, 268, 418, 148, (0.8, 8.2), (-0.03, 0.14), ks,
                                 [0.0, 0.05, 0.10], title="Δρ̄ к LAION, центроиды"),
        ("oracle", "d"): Panel(540, 268, 418, 148, (0.8, 8.2), (-0.03, 0.14), ks,
                               [0.0, 0.05, 0.10], title="Δρ̄ к LAION, оракул"),
    }
    for (router, kind), p in panels.items():
        p.frame()
        if kind == "d":
            p.band(-0.02, 0.02)
            p.hline(0.02, width=1.4)
            p.hline(-0.02, width=1.4)
            p.text(p.x + p.w - 4, p.sy(0.02) - 7, "порог ±0.02", size=13, colour=AMBER,
                   opacity=0.85, anchor="end")
        lin = [(k, ref[(router, k)]) for k in ref_ks]
        p.line("LAION", lin)
        p.markers("LAION", lin)
        for name in ("perturbation", "GMM", "cuVS"):
            if kind == "rho":
                pts = [(k, rho[name][router][k]) for k in ks]
            else:
                pts = [(k, rho[name][router][k] - ref[(router, k)]) for k in ref_ks]
            p.line(name, pts)
            p.markers(name, pts)
        p.xtick_labels(dy=19)
    body = "\n  ".join(p.render() for p in panels.values())
    body += "\n  " + legend([("LAION", "LAION 1M"), ("perturbation", "perturbation"),
                             ("GMM", "GMM"), ("cuVS", "cuVS")], 484, W)
    body += ('\n  <text x="530" y="458" fill="' + TEXT + '" fill-opacity="0.55" font-size="14" '
             'text-anchor="middle">k — число шардов (из 16)</text>')
    title = "ρ̄(k): три генератора против LAION 1M при роутере по центроидам и оракуле"
    desc = ("Слева и справа сверху — средняя доля 100 ближайших соседей запроса, попавших в k выбранных "
            "шардов, для LAION 1M и трёх генераторов. Внизу — отклонение от LAION: при k = 1 центроиды "
            "завышены на 0.021–0.030, оракул на 0.030 у cuVS и на 0.115–0.121 у perturbation и GMM; "
            "при k ≥ 3 все кривые сходятся в пределах 0.01")
    path = OUT / "13-rho-curves-1m.svg"
    path.write_text(svg(W, H, title, desc, body))
    return path


# ---------------------------------------------------------------------------
# 14: recall@100 от числа бит RaBitQ
# ---------------------------------------------------------------------------

def fig_rabitq() -> Path:
    ref = laion_reference()["quant"]
    quant = generator_quant()
    bits = list(range(1, 9))
    W, H = 980, 440
    p = Panel(70, 40, 850, 268, (0.8, 8.2), (0.35, 1.0), bits,
              [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], title="")
    p.frame()
    for thr, label in ((0.95, "recall 0.95"), (0.99, "recall 0.99")):
        p.hline(thr, opacity=0.7, width=1.4)
        p.text(p.x + p.w - 6, p.sy(thr) - 8, label, size=13, colour=AMBER, opacity=0.85,
               anchor="end")
    p.line("LAION", [(b, ref[b]) for b in bits])
    p.markers("LAION", [(b, ref[b]) for b in bits])
    for name in ("perturbation", "GMM", "cuVS"):
        pts = [(b, quant[name][b]) for b in bits]
        p.line(name, pts)
        p.markers(name, pts)
    p.xtick_labels(dy=22, label="бит на координату")
    body = p.render()
    body += "\n  " + legend([("LAION", "LAION 1M"), ("perturbation", "perturbation"),
                             ("GMM", "GMM"), ("cuVS", "cuVS")], 400, W)
    title = "recall@100 по кодам RaBitQ от числа бит: три генератора против LAION 1M"
    desc = ("Средний recall@100 по 1000 общим запросам LAION-heldout. При 1 бите LAION даёт 0.739, "
            "cuVS 0.671, perturbation 0.382 и GMM 0.397; минимум бит для recall 0.99 — 7 у LAION, "
            "8 у cuVS и больше 8 у perturbation и GMM")
    path = OUT / "14-rabitq-bits-1m.svg"
    path.write_text(svg(W, H, title, desc, body))
    return path


# ---------------------------------------------------------------------------
# 15: диагностические бары
# ---------------------------------------------------------------------------

def fig_stats() -> Path:
    ref = laion_reference()["stats"]
    st = generator_stats()
    W, H = 980, 232
    metrics = [
        ("lid100", "LID₁₀₀", ref["lid100"], "{:.1f}", None),
        ("loc_d90", "loc_d90", ref["loc_d90"], "{:.0f}", None),
        ("hub_skew10", "хабы N₁₀", ref["hub_skew10"], "{:.1f}", None),
        ("c2st_1nn", "C2ST", 0.5, "{:.2f}", "0.5 — неотличимы"),
    ]
    names = ["perturbation", "GMM", "cuVS"]
    labels = ["perturb", "GMM", "cuVS"]
    parts = []
    pw, gap, x0, top, ph = 214, 22, 30, 52, 132
    for i, (key, title, refv, fmt, ref_label) in enumerate(metrics):
        x = x0 + i * (pw + gap)
        peak = max(st[n][key] for n in names)
        p = Panel(x, top, pw, ph, (0, 1), (0, peak * 1.28), [], [], grid=False)
        parts.append(p.text(x, top - 20, title, size=16, colour=TEXT, opacity=0.9))
        parts.append(p.text(x, top - 4, ref_label or f"LAION {fmt.format(refv)}", size=13,
                            colour=AMBER, opacity=0.85))
        bw, bgap = 44, 14
        start = x + (pw - (3 * bw + 2 * bgap)) / 2 + bw / 2
        for j, (name, label) in enumerate(zip(names, labels)):
            cx = start + j * (bw + bgap)
            v = st[name][key]
            colour = MUTED if name == "perturbation" else (GREEN if name == "cuVS" else TEXT)
            p.bar(cx, v, bw, colour, opacity=0.85 if name == "GMM" else 1.0)
            parts.append(p.text(cx, p.sy(v) - 7, fmt.format(v), size=13, colour=TEXT,
                                opacity=0.85, anchor="middle"))
            parts.append(p.text(cx, top + ph + 17, label, size=13, colour=TEXT, opacity=0.6,
                               anchor="middle"))
        if key == "c2st_1nn" or True:
            y = p.sy(refv)
            parts.append(f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x + pw:.1f}" y2="{y:.1f}" '
                         f'stroke="{AMBER}" stroke-opacity="0.9" stroke-width="1.8" '
                         f'stroke-dasharray="7 5"/>')
        p.parts.append(f'<line x1="{x:.1f}" y1="{top + ph:.1f}" x2="{x + pw:.1f}" '
                       f'y2="{top + ph:.1f}" stroke="{MUTED}" stroke-opacity="0.75" stroke-width="1.2"/>')
        parts.append(p.render())
    title = "Диагностика: локальная размерность, ширина соседства, хабы, C2ST"
    desc = ("LID₁₀₀ у perturbation 99.4 и у GMM 96.7 против 24.2 у LAION 1M; loc_d90 80 и 80 против 53; "
            "асимметрия хабов N₁₀ 5.1 и 15.4 против 3.8; cuVS по всем трём ближе к LAION, но легко "
            "отличим по C2ST (0.885 против 0.5). Пунктир — значение LAION 1M")
    path = OUT / "15-stats-bars-1m.svg"
    path.write_text(svg(W, H, title, desc, "\n  ".join(parts)))
    return path


def main() -> None:
    for path in (fig_rho(), fig_rabitq(), fig_stats()):
        print(f"→ {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
