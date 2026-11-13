"""Figures and tables for the results section of the report, from out/summary.csv.

  python scripts/plot_results.py      # first: collects out/ into out/summary.csv
  python scripts/report_figures.py

Needs out/summary.csv and figures/pattern_stats.csv (scripts/pattern_stats.py).
Writes figures/report/:
  stability.tex                     stable patterns per dataset and rule (LaTeX table)
  hebb_stability_vs_overlap.png     Hebb: fraction of stable patterns vs mean |m|
  recall_flip_<rule>.png            recall vs flipped pixels, one line per given dataset
  nearest_flip_<rule>.png           the same, but counting the pattern nearest to the input as success
  overlap_flip_hebb.png             Hebb: overlap of the final state with the source pattern
  mix_<rule>.png                    pixel mixtures: ending in the source / in another pattern vs share q
  cycles_sync.png                   share of synchronous retrievals ending in a 2-cycle
  large_25x25_vs_25x50.png          recall vs flipped pixels for both large sets and both rules
  similar_vs_diverse.png            pseudoinverse: similar vs diverse letters (flip, mix)
  random_outcomes.png               where random inputs end up, per dataset and rule
  stable_5x5.png                    recall of the custom 5x5 sets vs number of flipped pixels
  shapes.png                        MPEG-7 sets: recall vs flipped pixels and pixel mixtures
  capacity_stability.png            random patterns: stable fraction vs load N/n, with the theory for Hebb
  capacity_recall.png               random patterns: recall from 10% / 20% flipped pixels vs load N/n
Async retrieval unless stated otherwise; values are means over the 5 seeds.
"""

import math
import os

import numpy as np
import pandas as pd

from common import FIG_DIR, OUT_DIR, ROOT, is_capacity

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = FIG_DIR / "report"
SERIES = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
TEXT = "#3d3d3a"
GRID = "#d9d8d2"

GIVEN = {
    "small-7x7": "small",
    "letters-abc-8x12": "letters-abc",
    "animals-14x9": "animals",
    "letters-14x20": "letters",
    "OCRA-12x30-cut": "OCRA",
    "large-25x25": "large",
    "large-25x25.plus": "large plus",
    "large-25x50": "large 25×50",
}
SHAPES = {
    "shapes5-200x300": "shapes5",
    "shapes10-200x300": "shapes10",
    "shapes20-200x300": "shapes20",
}
CUSTOM = {
    "letters-similar-14x20": "letters-similar",
    "letters-diverse-14x20": "letters-diverse",
    "stable-hebb-5x5": "stable-hebb 5×5",
    "random12-5x5": "random12 5×5",
    "random24-5x5": "random24 5×5",
    "oscillation-4x4": "oscillation 4×4",
}
NAMES = {**GIVEN, **SHAPES, **CUSTOM}
RULES = {"hebb": "Hebb", "pseudoinverse": "pseudoodwrotność"}
RULE_STYLE = {"hebb": "--", "pseudoinverse": "-"}
KIND_AXIS = {
    "flip": "odwrócone piksele [%]",
    "mix": "piksele z wzorca źródłowego [%]",
}
CATEGORIES = {
    "cat_other_pattern": "wzorzec",
    "cat_other_inverse": "negatyw wzorca",
    "cat_spurious": "stan fałszywy",
}

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "text.color": TEXT,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 2,
        "legend.frameon": False,
    }
)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    if fig.get_layout_engine() is None:
        fig.tight_layout()
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)
    print((OUT / name).relative_to(ROOT))


def axis_x(kind, levels):
    return np.asarray(levels) if kind == "shift" else np.asarray(levels) * 100


def style(ax, xlabel, ylabel="dokładnie odtworzone", ylim=(-0.02, 1.02)):
    ax.set(xlabel=xlabel, ylabel=ylabel, ylim=ylim)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def curve(ax, df, dataset, rule, kind, metric="recall_rate", mode="async", **kw):
    g = df[
        (df.dataset == dataset)
        & (df.rule == rule)
        & (df.kind == kind)
        & (df["mode"] == mode)
    ]
    if g.empty:
        return
    m = g.groupby("level")[metric].mean()
    ax.plot(axis_x(kind, m.index), m.values, markersize=4, **kw)


def per_dataset_lines(
    df,
    datasets,
    rule,
    kind,
    path,
    metric="recall_rate",
    ylabel="dokładnie odtworzone",
    ylim=(-0.02, 1.02),
    reference=None,
):
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for k, (ds, label) in enumerate(datasets.items()):
        curve(
            ax,
            df,
            ds,
            rule,
            kind,
            metric,
            color=SERIES[k],
            marker=MARKERS[k],
            label=label,
        )
    if reference is not None:
        x = np.linspace(0, 50, 50)
        ax.plot(
            x,
            reference(x / 100),
            color=TEXT,
            linewidth=1,
            linestyle=":",
            label="wejście",
        )
    style(ax, KIND_AXIS[kind], ylabel, ylim)
    if kind == "shift":
        ax.set_xticks([1, 2, 3])
    ax.legend(fontsize=7, ncol=2)
    save(fig, path)


def stability_table(df):
    sub = (
        df[df.kind == "patterns"]
        .groupby(["dataset", "rule"])[["stable_patterns", "n_patterns"]]
        .first()
    )
    rows = []
    for ds, label in NAMES.items():
        if (ds, "hebb") not in sub.index:
            continue
        n = int(sub.loc[(ds, "hebb"), "n_patterns"])
        cells = [
            f"{int(sub.loc[(ds, r), 'stable_patterns'])}/{n}"
            if (ds, r) in sub.index
            else "--"
            for r in RULES
        ]
        rows.append(
            f"\\texttt{{{label.replace('×', 'x')}}} & {n} & {' & '.join(cells)} \\\\"
        )
        if ds in ("large-25x50", "shapes20-200x300"):
            rows.append("\\midrule")
    while rows and rows[-1] == "\\midrule":
        rows.pop()
    tex = "\n".join(
        [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\small",
            "\\begin{tabular}{lrrr}",
            "\\toprule",
            "zbiór & $N$ & Hebb & pseudoodwrotność \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Liczba wzorców zapamiętanych, które są stanami stabilnymi sieci.}",
            "\\label{tab:stabilnosc}",
            "\\end{table}",
        ]
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stability.tex").write_text(tex + "\n")
    print((OUT / "stability.tex").relative_to(ROOT))


def stability_vs_overlap(df):
    path = FIG_DIR / "pattern_stats.csv"
    if not path.exists():
        print(f"skip hebb_stability_vs_overlap.png: no {path.relative_to(ROOT)}")
        return
    stats = pd.read_csv(path).set_index("dataset")
    sub = (
        df[(df.kind == "patterns") & (df.rule == "hebb")]
        .groupby("dataset")[["stable_patterns", "n_patterns"]]
        .first()
    )
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for ds, label in {**GIVEN, **SHAPES}.items():
        if ds not in stats.index or ds not in sub.index:
            continue
        x = stats.loc[ds, "mean_abs_overlap"]
        y = sub.loc[ds, "stable_patterns"] / sub.loc[ds, "n_patterns"]
        ax.scatter(
            x, y, s=40, color=SERIES[0], edgecolor="white", linewidth=1.5, zorder=3
        )
        on_axis = y == 0
        ax.annotate(
            f"{label} ($N$={int(sub.loc[ds, 'n_patterns'])})",
            (x, y),
            textcoords="offset points",
            xytext=(3, -6) if on_axis else (5, 3),
            rotation=-45 if on_axis else 0,
            va="top" if on_axis else "baseline",
            fontsize=7,
        )
    style(
        ax,
        "średnie podobieństwo $|m_{pq}|$",
        "stabilne wzorce (Hebb)",
        ylim=(-0.45, 1.05),
    )
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.spines["bottom"].set_visible(False)
    ax.set_xlim(0, 1)
    save(fig, "hebb_stability_vs_overlap.png")


def cycles(df):
    sync = df[(df["mode"] == "sync") & df.dataset.isin(GIVEN)].fillna(
        {"status_cycle": 0}
    )
    share = (
        sync.groupby(["dataset", "rule"])[["status_cycle", "n_queries"]]
        .sum()
        .assign(share=lambda t: t.status_cycle / t.n_queries)
        .share.unstack()
    )
    names = [d for d in GIVEN if d in share.index]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    h = 0.38
    for k, rule in enumerate(RULES):
        if rule in share:
            ax.barh(
                y + (k - 0.5) * h,
                share.loc[names, rule] * 100,
                height=h,
                color=SERIES[k],
                edgecolor="white",
                linewidth=1,
                label=RULES[rule],
            )
    ax.set_yticks(y, [GIVEN[d] for d in names])
    ax.invert_yaxis()
    ax.set_xlabel("odtworzenia zakończone cyklem [%]")
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7)
    save(fig, "cycles_sync.png")


def pair_plot(
    df, datasets, kinds, rules, path, colors, xlabel_override=None, x_scale=None
):
    fig, axes = plt.subplots(
        1,
        len(kinds),
        figsize=(max(5.2, 3.6 * len(kinds)), 3.6),
        squeeze=False,
        layout="constrained",
    )
    for ax, kind in zip(axes[0], kinds):
        for k, (ds, label) in enumerate(datasets.items()):
            for rule in rules:
                g = df[
                    (df.dataset == ds)
                    & (df.rule == rule)
                    & (df.kind == kind)
                    & (df["mode"] == "async")
                ]
                if g.empty:
                    continue
                m = g.groupby("level").recall_rate.mean()
                x = x_scale(ds, m.index) if x_scale else axis_x(kind, m.index)
                name = label if len(rules) == 1 else f"{label}, {RULES[rule]}"
                ax.plot(
                    x,
                    m.values,
                    color=colors[k],
                    linestyle=RULE_STYLE[rule],
                    marker=MARKERS[k],
                    markersize=4,
                    label=name,
                )
        style(ax, xlabel_override or KIND_AXIS[kind])
        if kind == "shift":
            ax.set_xticks([1, 2, 3])
        if x_scale:
            ax.xaxis.get_major_locator().set_params(integer=True)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if not labels:
        plt.close(fig)
        print(f"skip {path}: no runs for {', '.join(datasets)}")
        return
    ncol = min(3 if len(kinds) > 1 else 2, len(labels))
    fig.legend(handles, labels, loc="outside lower center", ncol=ncol, fontsize=7)
    save(fig, path)


def random_outcomes(df):
    sub = df[
        (df.kind == "random")
        & (df["mode"] == "async")
        & df.dataset.isin({**GIVEN, **SHAPES})
    ]
    names = [d for d in {**GIVEN, **SHAPES} if d in set(sub.dataset)]
    fig, axes = plt.subplots(
        1, 2, figsize=(7.2, 3.6), sharey=True, layout="constrained"
    )
    for ax, rule in zip(axes, RULES):
        t = (
            sub[sub.rule == rule]
            .groupby("dataset")[list(CATEGORIES)]
            .sum()
            .reindex(names)
        )
        frac = t.div(t.sum(axis=1), axis=0) * 100
        left = np.zeros(len(names))
        for k, (col, label) in enumerate(CATEGORIES.items()):
            ax.barh(
                range(len(names)),
                frac[col],
                left=left,
                color=SERIES[k],
                edgecolor="white",
                linewidth=1,
                height=0.7,
                label=label,
            )
            left += frac[col].to_numpy()
        ax.set_title(RULES[rule], fontsize=9)
        ax.set_xlabel("wejścia losowe [%]")
        ax.set_xlim(0, 100)
    axes[0].set_yticks(range(len(names)), [NAMES[d] for d in names])
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=7)
    save(fig, "random_outcomes.png")


def mix_plot(df, rule):
    sub = df[(df.kind == "mix") & (df.rule == rule) & (df["mode"] == "async")]
    if sub.empty:
        return
    fig, axes = plt.subplots(
        1, 2, figsize=(7.2, 3.6), sharey=True, layout="constrained"
    )
    panels = [
        ("recall_rate", "koniec we wzorcu źródłowym"),
        ("other_rate", "koniec w innym wzorcu"),
    ]
    sub = sub.assign(other_rate=sub.cat_other_pattern / sub.n_queries)
    for ax, (metric, title) in zip(axes, panels):
        for k, (ds, label) in enumerate(GIVEN.items()):
            g = sub[sub.dataset == ds]
            if g.empty:
                continue
            m = g.groupby("level")[metric].mean()
            ax.plot(
                m.index * 100,
                m.values,
                color=SERIES[k],
                marker=MARKERS[k],
                markersize=4,
                label=label,
            )
        style(ax, KIND_AXIS["mix"], "udział wejść" if metric == "recall_rate" else "")
        ax.set_title(title, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=7)
    save(fig, f"mix_{rule}.png")


def hebb_stable_theory(load, n):
    """Expected fraction of stable patterns. Without the self-coupling the field of neuron i of a
    stored pattern is x_i (n - 1) / n plus the crosstalk of the N - 1 other patterns over the
    remaining n - 1 neurons, approximately normal with variance (N - 1)(n - 1) / n^2.
    Neurons are treated as independent, so the pattern is stable with probability (1 - p)^n."""
    n_patterns = load * n
    if n_patterns <= 1:
        return 1.0
    z = ((n - 1) / n) / math.sqrt((n_patterns - 1) * (n - 1) / n**2)
    p_wrong = 0.5 * math.erfc(z / math.sqrt(2))
    return (1 - p_wrong) ** n


def capacity(df):
    sub = df[df.dataset.map(is_capacity)]
    if sub.empty:
        print("skip capacity_*.png: no capacity-* runs")
        return
    n = (
        int(sub.n_neurons.iloc[0])
        if "n_neurons" in sub
        else int(sub.width.iloc[0] * sub.height.iloc[0])
    )
    stable = (
        sub[sub.kind == "patterns"]
        .groupby(["dataset", "rule"])[["stable_patterns", "n_patterns"]]
        .first()
        .assign(
            frac=lambda t: t.stable_patterns / t.n_patterns,
            load=lambda t: t.n_patterns / n,
        )
    )

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    loads = np.linspace(0.005, 0.95, 300)
    ax.plot(
        loads,
        [hebb_stable_theory(a, n) for a in loads],
        color=TEXT,
        linewidth=1,
        linestyle=":",
        label="Hebb, teoria",
    )
    for k, rule in enumerate(RULES):
        t = stable.xs(rule, level="rule").sort_values("load")
        ax.plot(
            t.load,
            t.frac,
            color=SERIES[k],
            linestyle=RULE_STYLE[rule],
            marker=MARKERS[k],
            markersize=4,
            label=RULES[rule],
        )
    ax.axvline(0.138, color=GRID, linewidth=1.5)
    ax.annotate(
        "0,138", (0.138, 1.0), textcoords="offset points", xytext=(3, -10), fontsize=7
    )
    style(ax, "obciążenie $N/n$", "stabilne wzorce")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=7)
    save(fig, "capacity_stability.png")

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    flips = sub[
        (sub.kind == "flip") & (sub["mode"] == "async") & sub.level.isin([0.10, 0.20])
    ]
    for j, level in enumerate([0.10, 0.20]):
        for rule in RULES:
            g = flips[(flips.level == level) & (flips.rule == rule)]
            if g.empty:
                continue
            m = g.groupby("dataset").agg(
                load=("n_patterns", "first"), rate=("recall_rate", "mean")
            )
            m = m.assign(load=m.load / n).sort_values("load")
            ax.plot(
                m.load,
                m.rate,
                color=SERIES[j + 2],
                linestyle=RULE_STYLE[rule],
                marker=MARKERS[j + 2],
                markersize=4,
                label=f"{RULES[rule]}, {level:.0%} odwróconych".replace(".", ","),
            )
    ax.axvline(0.138, color=GRID, linewidth=1.5)
    style(ax, "obciążenie $N/n$")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=7)
    save(fig, "capacity_recall.png")


def main():
    path = OUT_DIR / "summary.csv"
    if not path.exists():
        raise SystemExit(
            f"no {path.relative_to(ROOT)} - run scripts/plot_results.py first"
        )
    df = pd.read_csv(path)

    stability_table(df)
    stability_vs_overlap(df)
    for rule in RULES:
        per_dataset_lines(df, GIVEN, rule, "flip", f"recall_flip_{rule}.png")
    for rule in RULES:
        per_dataset_lines(
            df,
            GIVEN,
            rule,
            "flip",
            f"nearest_flip_{rule}.png",
            metric="nearest_recall_rate",
            ylabel="odtworzony wzorzec najbliższy wejściu",
        )
        mix_plot(df, rule)
    per_dataset_lines(
        df,
        GIVEN,
        "hebb",
        "flip",
        "overlap_flip_hebb.png",
        metric="mean_target_overlap",
        ylabel="podobieństwo stanu końcowego do wzorca",
        ylim=(-0.05, 1.02),
        reference=lambda p: 1 - 2 * p,
    )
    cycles(df)
    pair_plot(
        df,
        {"large-25x25": "large", "large-25x50": "large 25×50"},
        ["flip"],
        list(RULES),
        "large_25x25_vs_25x50.png",
        [SERIES[5], SERIES[7]],
    )
    pair_plot(
        df,
        {"letters-similar-14x20": "podobne", "letters-diverse-14x20": "różnorodne"},
        ["flip", "mix"],
        ["pseudoinverse"],
        "similar_vs_diverse.png",
        [SERIES[6], SERIES[2]],
    )
    random_outcomes(df)
    pair_plot(
        df,
        {
            "stable-hebb-5x5": "stable-hebb",
            "random12-5x5": "random12",
            "random24-5x5": "random24",
        },
        ["flip"],
        list(RULES),
        "stable_5x5.png",
        [SERIES[3], SERIES[4], SERIES[6]],
        xlabel_override="odwrócone piksele (z 25)",
        x_scale=lambda ds, lv: np.round(np.asarray(lv) * 25),
    )
    pair_plot(
        df,
        SHAPES,
        ["flip", "mix"],
        list(RULES),
        "shapes.png",
        [SERIES[0], SERIES[1], SERIES[2]],
    )
    capacity(df)


if __name__ == "__main__":
    main()
