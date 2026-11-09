import argparse
import math
import os

import numpy as np

from common import FIG_DIR, ROOT, Dataset, datasets

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

import make_noise  # noqa: E402


def to_pixels(ds: Dataset, state: np.ndarray, scale: int) -> Image.Image:
    img = np.where(ds.image(state) > 0, 0, 255).astype(np.uint8)
    return Image.fromarray(img, mode="L").resize(
        (ds.width * scale, ds.height * scale), Image.NEAREST
    )


def show(ax, ds: Dataset, state: np.ndarray, title: str):
    ax.imshow(ds.image(state), cmap="gray_r", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def sheet(
    ds: Dataset, states: list[np.ndarray], titles: list[str], path, max_cols: int = 8
):
    cols = min(max_cols, len(states))
    rows = math.ceil(len(states) / cols)
    aspect = ds.height / ds.width
    fig, axes = plt.subplots(
        rows, cols, figsize=(1.6 * cols, 1.6 * aspect * rows + 0.4), squeeze=False
    )
    for ax in axes.flat:
        ax.axis("off")
    for ax, state, title in zip(axes.flat, states, titles):
        ax.axis("on")
        show(ax, ds, state, title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


DAMAGE_KINDS = ["flip", "block", "erase", "shift", "mix"]
KIND_LABELS = {
    "flip": "odwrócenie pikseli",
    "block": "zaszumiony prostokąt",
    "erase": "wymazany pas",
    "shift": "przesunięcie",
    "mix": "mieszanka z innym",
    "invert": "negatyw",
}


def level_label(kind: str, level: float) -> str:
    if kind == "shift":
        return f"{int(level)} px"
    return f"{level:.0%}"


def sheet_rows(ds: Dataset, rows, path):
    cols = max(len(states) for _, states, _ in rows)
    aspect = ds.height / ds.width
    fig, axes = plt.subplots(
        len(rows),
        cols,
        figsize=(1.3 * cols + 1.2, 1.3 * aspect * len(rows) + 0.6),
        squeeze=False,
    )
    for ax in axes.flat:
        ax.axis("off")
    for r, (label, states, titles) in enumerate(rows):
        for c, (state, title) in enumerate(zip(states, titles)):
            ax = axes[r, c]
            ax.axis("on")
            show(ax, ds, state, title)
        axes[r, 0].set_ylabel(label, fontsize=8, rotation=0, ha="right", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render(ds: Dataset, scale: int):
    out = FIG_DIR / ds.name
    (out / "patterns").mkdir(parents=True, exist_ok=True)
    patterns = ds.load()

    for mu, p in enumerate(patterns):
        to_pixels(ds, p, scale).save(out / "patterns" / f"pattern_{mu:02d}.png")
    sheet(
        ds,
        list(patterns),
        [str(mu) for mu in range(len(patterns))],
        out / "patterns.png",
    )

    rng = np.random.default_rng(0)
    rows = []
    for kind in DAMAGE_KINDS:
        levels = make_noise.KINDS[kind]
        states = [patterns[0]] + [
            make_noise.damage(kind, patterns, 0, ds, lv, rng) for lv in levels
        ]
        titles = ["oryginał"] + [level_label(kind, lv) for lv in levels]
        sheet(ds, states, titles, out / f"noise_{kind}.png", max_cols=len(states))
        rows.append((KIND_LABELS[kind], states, titles))
    rows.append(
        (KIND_LABELS["invert"], [patterns[0], -patterns[0]], ["oryginał", "negatyw"])
    )
    sheet_rows(ds, rows, out / "noise_examples.png")
    print(f"{ds.name}: {len(patterns)} patterns {ds.width}x{ds.height} -> {out}")
    return len(patterns)


def figure(path: str, width: str, caption: str, label: str) -> str:
    return (
        "\\begin{figure}[htbp]\n"
        "  \\centering\n"
        f"  \\includegraphics[width={width}]{{{path}}}\n"
        f"  \\caption{{{caption}}}\n"
        f"  \\label{{{label}}}\n"
        "\\end{figure}\n"
    )


def latex_blocks(ds: Dataset, n_patterns: int) -> str:
    width = f"{min(1.0, 0.12 * min(8, n_patterns)):.2f}\\linewidth"
    name = f"\\texttt{{{ds.name}}}"
    patterns = figure(
        f"figures/{ds.name}/patterns.png",
        width,
        f"Zbiór {name}: {n_patterns} wzorców o rozmiarze ${ds.width}\\times{ds.height}$ ($n = {ds.n}$). "
        "Czarny piksel to stan $1$, biały $-1$.",
        f"fig:{ds.name}:patterns",
    )
    noise = figure(
        f"figures/{ds.name}/noise_examples.png",
        "\\linewidth",
        f"Przykłady uszkodzonych wejść utworzonych z wzorca 0 zbioru {name}: "
        "wszystkie rodzaje i poziomy uszkodzenia.",
        f"fig:{ds.name}:noise",
    )
    return patterns + "\n" + noise


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--datasets", nargs="*")
    p.add_argument(
        "--scale",
        type=int,
        default=12,
        help="pixels per cell in patterns/pattern_XX.png",
    )
    args = p.parse_args()
    blocks = []
    for ds in datasets(args.datasets):
        blocks.append(latex_blocks(ds, render(ds, args.scale)))
    tex = FIG_DIR / "patterns.tex"
    tex.write_text("\n".join(blocks))
    print(f"{len(blocks)} figure blocks -> {tex}")


if __name__ == "__main__":
    main()
