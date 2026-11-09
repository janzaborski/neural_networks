import json
import os

import numpy as np
import pandas as pd

from common import FIG_DIR, OUT_DIR, ROOT

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CATEGORIES = ["target", "target_inverse", "other_pattern", "other_inverse", "spurious"]
CATEGORY_LABELS = {
    "target": "wzorzec źródłowy",
    "target_inverse": "negatyw źródłowego",
    "other_pattern": "inny wzorzec",
    "other_inverse": "negatyw innego",
    "spurious": "stan fałszywy",
}
COLORS = dict(zip(CATEGORIES, ["#2a9d8f", "#8ab17d", "#e9c46a", "#f4a261", "#e76f51"]))
DAMAGE_KINDS = ["flip", "block", "erase", "shift", "mix"]
KIND_LABELS = {
    "flip": "odwrócenie pikseli",
    "block": "zaszumiony prostokąt",
    "erase": "wymazany pas",
    "shift": "przesunięcie",
    "mix": "mieszanka z innym",
}
LEVEL_LABELS = {
    "flip": "odwrócone piksele [%]",
    "block": "zaszumiony obszar [%]",
    "erase": "wymazany obszar [%]",
    "shift": "przesunięcie [px]",
    "mix": "kolumny z wzorca źródłowego [%]",
}


def level_values(kind: str, levels):
    return np.asarray(levels) if kind == "shift" else np.asarray(levels) * 100


def collect() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, results = [], []
    for summary_path in OUT_DIR.rglob("summary.json"):
        run = summary_path.parent
        cfg_path = run / "config.json"
        if not cfg_path.exists():
            continue
        cfg = json.loads(cfg_path.read_text())
        summary = json.loads(summary_path.read_text())
        row = {
            k: v for k, v in cfg.items() if k not in ("patterns", "queries", "targets")
        }
        row.update(
            {
                k: summary.get(k)
                for k in (
                    "n_neurons",
                    "n_patterns",
                    "stable_patterns",
                    "n_queries",
                    "recalled",
                    "recall_rate",
                    "mean_initial_target_overlap",
                    "mean_target_overlap",
                    "mean_steps",
                    "seconds",
                )
            }
        )
        for cat in CATEGORIES:
            row[f"cat_{cat}"] = summary["categories"].get(cat, 0)
        for status, count in summary["statuses"].items():
            row[f"status_{status}"] = count
        rows.append(row)

        if cfg["kind"] in DAMAGE_KINDS and (run / "results.csv").exists():
            r = pd.read_csv(
                run / "results.csv", usecols=["category", "hamming_initial"]
            )
            r = r.assign(
                dataset=cfg["dataset"],
                rule=cfg["rule"],
                mode=cfg["mode"],
                kind=cfg["kind"],
                distance=r["hamming_initial"] / summary["n_neurons"],
            )
            results.append(
                r[["dataset", "rule", "mode", "kind", "distance", "category"]]
            )
    if not rows:
        raise SystemExit(
            f"no finished runs in {OUT_DIR} - run scripts/experiments.py first"
        )
    df = pd.DataFrame(rows).sort_values(
        ["dataset", "rule", "mode", "kind", "level", "seed"]
    )
    df.to_csv(OUT_DIR / "summary.csv", index=False)
    per_query = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    return df, per_query


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(path.relative_to(ROOT))


def line_plot(df: pd.DataFrame, kind: str, metric: str, ylabel: str, path, ylim=None):
    sub = df[df.kind == kind]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for (rule, mode), g in sub.groupby(["rule", "mode"]):
        stats = g.groupby("level")[metric].agg(["median", "min", "max"])
        x = level_values(kind, stats.index)
        (line,) = ax.plot(x, stats["median"], marker="o", ms=3, label=f"{rule}, {mode}")
        ax.fill_between(
            x, stats["min"], stats["max"], color=line.get_color(), alpha=0.2
        )
    ax.set(xlabel=LEVEL_LABELS[kind], ylabel=ylabel)
    if kind == "shift":
        ax.set_xticks(level_values(kind, sorted(sub.level.unique())))
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    save(fig, path)


def distance_plot(per_query: pd.DataFrame, path, bins=np.linspace(0, 0.5, 11)):
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    centers = (bins[:-1] + bins[1:]) / 2
    for kind in DAMAGE_KINDS:
        g = per_query[per_query.kind == kind]
        if g.empty:
            continue
        idx = np.digitize(g.distance, bins) - 1
        ok = (idx >= 0) & (idx < len(centers))
        stats = (
            g[ok]
            .assign(bin=idx[ok], hit=lambda x: x.category == "target")
            .groupby("bin")
            .agg(rate=("hit", "mean"), count=("hit", "size"))
        )
        stats = stats[stats["count"] >= 5]
        ax.plot(
            centers[stats.index] * 100,
            stats["rate"],
            marker="o",
            ms=3,
            label=KIND_LABELS[kind],
        )
    ax.set(
        xlabel="odległość Hamminga wejścia od wzorca źródłowego [% pikseli]",
        ylabel="dokładnie odtworzone",
        ylim=(0, 1.02),
        xlim=(0, 50),
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    save(fig, path)


def outcome_bars(
    sub: pd.DataFrame, x_col: str, xlabel: str, path, kind: str | None = None
):
    totals = sub.groupby(x_col)[[f"cat_{c}" for c in CATEGORIES]].sum()
    frac = totals.div(totals.sum(axis=1), axis=0)
    if kind in LEVEL_LABELS:
        labels = [f"{v:g}" for v in level_values(kind, frac.index)]
    else:
        labels = [str(v) for v in frac.index]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    bottom = np.zeros(len(frac))
    for cat in CATEGORIES:
        values = frac[f"cat_{cat}"].to_numpy()
        ax.bar(
            labels, values, bottom=bottom, color=COLORS[cat], label=CATEGORY_LABELS[cat]
        )
        bottom = bottom + values
    ax.set(xlabel=xlabel, ylabel="udział wejść", ylim=(0, 1))
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
    save(fig, path)


def stability(df: pd.DataFrame):
    sub = df[(df.kind == "patterns") & (df.seed == 0)]
    if sub.empty:
        return
    table = (
        sub.groupby(["dataset", "rule"])[["stable_patterns", "n_patterns", "n_neurons"]]
        .first()
        .reset_index()
        .sort_values(["n_neurons", "dataset"])
    )
    table.to_csv(FIG_DIR / "stability.csv", index=False)
    print((FIG_DIR / "stability.csv").relative_to(ROOT))

    names = list(dict.fromkeys(table.dataset))
    counts = table.groupby("dataset").n_patterns.first().reindex(names)
    rules = sorted(table.rule.unique())
    width = 0.8 / len(rules)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for k, rule in enumerate(rules):
        t = table[table.rule == rule].set_index("dataset").reindex(names)
        x = [i + (k - (len(rules) - 1) / 2) * width for i in range(len(names))]
        ax.bar(x, t.stable_patterns / t.n_patterns, width=width, label=rule)
    ax.set_xticks(
        range(len(names)),
        [f"{n}\n({int(p)} wz.)" for n, p in zip(names, counts)],
        rotation=45,
        ha="right",
        fontsize=7,
    )
    ax.set(ylabel="stabilne wzorce uczące", ylim=(0, 1))
    ax.legend(fontsize=8)
    save(fig, FIG_DIR / "stability.png")


def main():
    df, per_query = collect()
    print(f"{len(df)} runs -> {(OUT_DIR / 'summary.csv').relative_to(ROOT)}")
    stability(df)
    for name, d in df.groupby("dataset"):
        out = FIG_DIR / name
        for kind in DAMAGE_KINDS:
            line_plot(
                d,
                kind,
                "recall_rate",
                "dokładnie odtworzone",
                out / f"recall_{kind}.png",
                (0, 1.02),
            )
            for (rule, mode), g in d[d.kind == kind].groupby(["rule", "mode"]):
                outcome_bars(
                    g,
                    "level",
                    LEVEL_LABELS[kind],
                    out / f"outcomes_{kind}_{rule}_{mode}.png",
                    kind,
                )
        line_plot(
            d,
            "flip",
            "mean_target_overlap",
            "średni overlap z wzorcem",
            out / "overlap_flip.png",
            (-1, 1.02),
        )
        line_plot(d, "flip", "mean_steps", "kroki / przebiegi", out / "steps_flip.png")
        for special in ("invert", "random"):
            sub = d[d.kind == special].assign(
                variant=lambda x: x["rule"] + ", " + x["mode"]
            )
            if not sub.empty:
                outcome_bars(sub, "variant", "", out / f"outcomes_{special}.png")
        if not per_query.empty:
            for (rule, mode), g in per_query[per_query.dataset == name].groupby(
                ["rule", "mode"]
            ):
                distance_plot(g, out / f"recall_vs_distance_{rule}_{mode}.png")


if __name__ == "__main__":
    main()
