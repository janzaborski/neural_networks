import argparse
import os

import numpy as np
import pandas as pd

from common import FIG_DIR, ROOT, Dataset, datasets

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def hebb_stable(patterns: np.ndarray) -> np.ndarray:
    x = patterns.astype(np.int64)
    n_patterns, n = x.shape
    h = (x @ x.T) @ x - n_patterns * x
    new = np.where(h > 0, 1, np.where(h < 0, -1, patterns))
    return (new == patterns).all(axis=1)


def overlaps(patterns: np.ndarray) -> np.ndarray:
    x = patterns.astype(np.float64)
    return x @ x.T / x.shape[1]


def stats(ds: Dataset) -> dict:
    patterns = ds.load()
    n_patterns = len(patterns)
    m = overlaps(patterns)
    off = np.abs(m[~np.eye(n_patterns, dtype=bool)])
    return {
        "dataset": ds.name,
        "N": n_patterns,
        "n": ds.n,
        "load": n_patterns / ds.n,
        "mean_state": float(patterns.mean()),
        "mean_abs_overlap": float(off.mean()) if off.size else 0.0,
        "max_abs_overlap": float(off.max()) if off.size else 0.0,
        "hebb_stable": int(hebb_stable(patterns).sum()),
    }


def overlap_plot(ds: Dataset):
    m = overlaps(ds.load())
    size = min(8.0, 2.0 + 0.3 * len(m))
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    im = ax.imshow(m, cmap="RdBu_r", vmin=-1, vmax=1)
    if len(m) <= 30:
        ticks = range(len(m))
        ax.set_xticks(ticks, ticks, fontsize=7)
        ax.set_yticks(ticks, ticks, fontsize=7)
    ax.set(xlabel="wzorzec", ylabel="wzorzec")
    fig.colorbar(im, ax=ax, label="overlap $m$")
    fig.tight_layout()
    path = FIG_DIR / ds.name / "overlaps.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def num(value: float, spec: str) -> str:
    return "$" + format(value, spec).replace(".", "{,}") + "$"


def latex(table: pd.DataFrame) -> str:
    rows = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        "zbiór & $N$ & $n$ & $N/n$ & śr.\\ stan & śr.\\ $|m|$ & maks.\\ $|m|$ & stabilne \\\\",
        "\\midrule",
    ]
    for r in table.itertuples():
        rows.append(
            f"\\texttt{{{r.dataset}}} & {r.N} & {r.n} & {num(r.load, '.2g')} & {num(r.mean_state, '+.2f')} & "
            f"{num(r.mean_abs_overlap, '.2f')} & {num(r.max_abs_overlap, '.2f')} & {r.hebb_stable}/{r.N} \\\\"
        )
    rows += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Zbiory wzorców zapamiętywanych przez sieć, uporządkowane według średniego podobieństwa.",
        " $N$ -- liczba wzorców, $n$ -- liczba neuronów, $N/n$ -- obciążenie sieci,",
        " śr.\\ stan -- przewaga czerni, śr.\\ $|m|$ -- średnie podobieństwo,",
        " maks.\\ $|m|$ -- największe podobieństwo, stabilne -- stabilne wzorce (reguła Hebba).}",
        "\\label{tab:zbiory}",
        "\\end{table}",
    ]
    return "\n".join(rows) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="*")
    args = p.parse_args()

    found = datasets(args.datasets)
    table = pd.DataFrame([stats(ds) for ds in found]).sort_values(
        ["mean_abs_overlap", "dataset"]
    )
    FIG_DIR.mkdir(exist_ok=True)
    table.to_csv(FIG_DIR / "pattern_stats.csv", index=False)
    (FIG_DIR / "pattern_stats.tex").write_text(latex(table))
    for ds in found:
        overlap_plot(ds)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(
        f"-> {FIG_DIR / 'pattern_stats.csv'}, {FIG_DIR / 'pattern_stats.tex'}, figures/<dataset>/overlaps.png"
    )


if __name__ == "__main__":
    main()
