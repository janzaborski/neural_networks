import argparse
import math
import os

import numpy as np

from common import FIG_DIR, ROOT, Dataset, datasets

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import make_noise


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
    levels = [0.0] + make_noise.FLIP_LEVELS
    flipped = [make_noise.flip(patterns[0], lv, rng) for lv in levels]
    sheet(
        ds,
        flipped,
        [f"{lv:.0%}" for lv in levels],
        out / "noise_flip.png",
        max_cols=len(levels),
    )

    levels = [0.0] + make_noise.BLOCK_LEVELS
    blocked = [
        make_noise.block(patterns[0], ds, lv, rng) if lv > 0 else patterns[0]
        for lv in levels
    ]
    sheet(
        ds,
        blocked,
        [f"{lv:.0%}" for lv in levels],
        out / "noise_block.png",
        max_cols=len(levels),
    )
    print(f"{ds.name}: {len(patterns)} patterns {ds.width}x{ds.height} -> {out}")


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
    for ds in datasets(args.datasets):
        render(ds, args.scale)


if __name__ == "__main__":
    main()
