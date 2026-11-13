"""Converts MPEG-7 CE-Shape-1 Part B silhouettes into +-1 bitmaps, one image per class.

  python scripts/convert_mpeg7.py ~/Downloads/MPEG7dataset          # extracted archive (searched recursively)
  python scripts/convert_mpeg7.py ~/Downloads/MPEG7dataset --sample 3 --width 200 --height 300

Files are named <class>-<k>.gif (k = 1..20, sometimes -01); --sample picks k for every class.
Each image is processed as follows:
  1. binarised; the background is the value that dominates the image border
     (MPEG-7 has white shapes on black, but this does not rely on it),
  2. cropped to the bounding box of the shape,
  3. rotated by 90 degrees if the shape is wider than tall and the target is taller than wide
     (or vice versa), so that it fills more of the frame (--no-rotate disables it),
  4. padded with background to the target aspect ratio, with a --margin around the shape,
  5. resized (bilinear on 0/1, then threshold 0.5) to --width x --height.
The shape is +1 (black in the figures), the background -1.

Writes to data/mpeg7/ (not a dataset directory, so other scripts ignore it):
  all-<W>x<H>.csv     one row per class, same format as the training sets
  classes.csv         class, source file, foreground fraction, mean state, mean and max |m| to the other classes
  overlaps.csv        |classes| x |classes| matrix of overlaps m = x . y / n
and figures/mpeg7/catalog.png with all converted classes and their foreground fraction.
Next step: scripts/select_shapes.py picks the training sets from this catalogue.
"""

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from common import FIG_DIR, ROOT, save_states

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MPEG7_DIR = ROOT / "data" / "mpeg7"
NAME = re.compile(r"^(?P<cls>.+)-(?P<k>\d+)\.gif$", re.IGNORECASE)


def find_files(src: Path, sample: int) -> dict[str, Path]:
    files = {}
    for path in sorted(src.rglob("*")):
        match = NAME.match(path.name)
        if match is None or path.name.startswith("._") or int(match["k"]) != sample:
            continue
        cls = match["cls"].lower()
        if cls in files:
            raise SystemExit(f"two files for class {cls}: {files[cls]} and {path}")
        files[cls] = path
    if not files:
        raise SystemExit(f"no <class>-{sample}.gif files under {src}")
    return files


def binarise(path: Path) -> np.ndarray:
    """True = shape."""
    img = np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 127
    border = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])
    background = border.mean() > 0.5
    return img != background


def fit(
    shape: np.ndarray, width: int, height: int, margin: float, rotate: bool
) -> np.ndarray:
    rows = np.flatnonzero(shape.any(axis=1))
    cols = np.flatnonzero(shape.any(axis=0))
    if rows.size == 0:
        raise ValueError("empty image")
    crop = shape[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
    if rotate and (crop.shape[1] > crop.shape[0]) != (width > height):
        crop = np.rot90(crop)
    h, w = crop.shape
    h_pad, w_pad = h * (1 + 2 * margin), w * (1 + 2 * margin)
    if w_pad / h_pad < width / height:
        w_pad = h_pad * width / height
    else:
        h_pad = w_pad * height / width
    canvas = np.zeros((round(h_pad), round(w_pad)), dtype=np.float32)
    r0 = (canvas.shape[0] - h) // 2
    c0 = (canvas.shape[1] - w) // 2
    canvas[r0 : r0 + h, c0 : c0 + w] = crop
    resized = Image.fromarray(canvas).resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(resized) >= 0.5


def catalog_figure(
    names: list[str], states: np.ndarray, width: int, height: int, path: Path
):
    cols = 10
    rows = -(-len(names) // cols)
    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 1.2, rows * 1.2 * height / width + 0.3 * rows)
    )
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, name, state in zip(np.ravel(axes), names, states):
        ax.imshow(
            state.reshape(height, width),
            cmap="gray_r",
            vmin=-1,
            vmax=1,
            interpolation="nearest",
        )
        ax.set_title(f"{name}\n{(state > 0).mean():.2f}", fontsize=6)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("source", type=Path, help="directory with the extracted MPEG-7 GIFs")
    p.add_argument(
        "--sample",
        type=int,
        default=1,
        help="which of the 20 images of every class (1..20)",
    )
    p.add_argument("--width", type=int, default=200)
    p.add_argument("--height", type=int, default=300)
    p.add_argument(
        "--margin",
        type=float,
        default=0.05,
        help="background margin, fraction of the shape size",
    )
    p.add_argument("--no-rotate", dest="rotate", action="store_false")
    args = p.parse_args()

    files = find_files(args.source, args.sample)
    names = sorted(files)
    states = np.stack(
        [
            np.where(
                fit(
                    binarise(files[c]),
                    args.width,
                    args.height,
                    args.margin,
                    args.rotate,
                ),
                1,
                -1,
            )
            .astype(np.int8)
            .reshape(-1)
            for c in names
        ]
    )

    x = states.astype(np.float64)
    m = x @ x.T / x.shape[1]
    off = np.abs(m)
    np.fill_diagonal(off, np.nan)

    MPEG7_DIR.mkdir(parents=True, exist_ok=True)
    shape = f"{args.width}x{args.height}"
    save_states(MPEG7_DIR / f"all-{shape}.csv", states)
    pd.DataFrame(
        {
            "class": names,
            "file": [str(files[c].relative_to(args.source)) for c in names],
            "foreground": (states > 0).mean(axis=1),
            "mean_state": states.mean(axis=1),
            "mean_abs_overlap": np.nanmean(off, axis=1),
            "max_abs_overlap": np.nanmax(off, axis=1),
        }
    ).to_csv(MPEG7_DIR / "classes.csv", index=False, float_format="%.4f")
    pd.DataFrame(m, index=names, columns=names).to_csv(
        MPEG7_DIR / "overlaps.csv", float_format="%.4f"
    )
    catalog_figure(
        names, states, args.width, args.height, FIG_DIR / "mpeg7" / "catalog.png"
    )

    fg = (states > 0).mean(axis=1)
    print(
        f"{len(names)} classes, {shape}, foreground {fg.min():.2f}..{fg.max():.2f} (mean {fg.mean():.2f}), "
        f"mean |m| between classes {np.nanmean(off):.3f}"
    )
    print(
        f"-> {MPEG7_DIR}/all-{shape}.csv, classes.csv, overlaps.csv; {FIG_DIR / 'mpeg7' / 'catalog.png'}"
    )


if __name__ == "__main__":
    main()
