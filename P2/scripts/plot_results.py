import json
import os

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


def collect() -> pd.DataFrame:
    rows = []
    for summary_path in OUT_DIR.rglob("summary.json"):
        cfg_path = summary_path.with_name("config.json")
        if not cfg_path.exists():
            continue
        cfg = json.loads(cfg_path.read_text())
        summary = json.loads(summary_path.read_text())
        row = {
            k: v for k, v in cfg.items() if k not in ("patterns", "queries", "targets")
        }
        row.update(
            {
                k: summary[k]
                for k in (
                    "n_neurons",
                    "n_patterns",
                    "stable_patterns",
                    "n_queries",
                    "recalled",
                    "recall_rate",
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
    if not rows:
        raise SystemExit(
            f"no finished runs in {OUT_DIR} - run scripts/experiments.py first"
        )
    df = pd.DataFrame(rows).sort_values(
        ["dataset", "rule", "mode", "kind", "level", "seed"]
    )
    df.to_csv(OUT_DIR / "summary.csv", index=False)
    return df


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
        x = stats.index * 100
        (line,) = ax.plot(x, stats["median"], marker="o", ms=3, label=f"{rule}, {mode}")
        ax.fill_between(
            x, stats["min"], stats["max"], color=line.get_color(), alpha=0.2
        )
    xlabel = "odwrócone piksele [%]" if kind == "flip" else "zaszumiony obszar [%]"
    ax.set(xlabel=xlabel, ylabel=ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    save(fig, path)


def outcome_bars(sub: pd.DataFrame, x_col: str, xlabel: str, path):
    totals = sub.groupby(x_col)[[f"cat_{c}" for c in CATEGORIES]].sum()
    frac = totals.div(totals.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    bottom = None
    labels = [f"{v * 100:.0f}" if isinstance(v, float) else str(v) for v in frac.index]
    for cat in CATEGORIES:
        values = frac[f"cat_{cat}"].to_numpy()
        ax.bar(
            labels, values, bottom=bottom, color=COLORS[cat], label=CATEGORY_LABELS[cat]
        )
        bottom = values if bottom is None else bottom + values
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

    fig, ax = plt.subplots(figsize=(7, 3.8))
    rules = sorted(table.rule.unique())
    names = list(dict.fromkeys(table.dataset))
    width = 0.8 / len(rules)
    for k, rule in enumerate(rules):
        t = table[table.rule == rule].set_index("dataset").reindex(names)
        x = [i + (k - (len(rules) - 1) / 2) * width for i in range(len(names))]
        ax.bar(x, t.stable_patterns / t.n_patterns, width=width, label=rule)
    ax.set_xticks(
        range(len(names)),
        [
            f"{n}\n({int(p)} wz.)"
            for n, p in zip(
                names, table.groupby("dataset").n_patterns.first().reindex(names)
            )
        ],
        rotation=45,
        ha="right",
        fontsize=7,
    )
    ax.set(ylabel="stabilne wzorce uczące", ylim=(0, 1))
    ax.legend(fontsize=8)
    save(fig, FIG_DIR / "stability.png")


def main():
    df = collect()
    print(f"{len(df)} runs -> {(OUT_DIR / 'summary.csv').relative_to(ROOT)}")
    stability(df)
    for name, d in df.groupby("dataset"):
        out = FIG_DIR / name
        line_plot(
            d,
            "flip",
            "recall_rate",
            "dokładnie odtworzone",
            out / "recall_flip.png",
            (0, 1.02),
        )
        line_plot(
            d,
            "block",
            "recall_rate",
            "dokładnie odtworzone",
            out / "recall_block.png",
            (0, 1.02),
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
        for (rule, mode), g in d[d.kind == "flip"].groupby(["rule", "mode"]):
            outcome_bars(
                g,
                "level",
                "odwrócone piksele [%]",
                out / f"outcomes_flip_{rule}_{mode}.png",
            )
        rnd = d[d.kind == "random"].assign(
            variant=lambda x: x["rule"] + ", " + x["mode"]
        )
        if not rnd.empty:
            outcome_bars(rnd, "variant", "", out / "outcomes_random.png")


if __name__ == "__main__":
    main()
