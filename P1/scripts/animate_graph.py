"""Weight evolution drawn as the network graph, saved as a GIF.

  python scripts/animate_graph.py out/study2/<config>/seed_0
  python scripts/animate_graph.py out/study2/<config>/seed_0 --frames 80 --fps 10 --scale layer

Needs weights_layer*.csv (output.weights_every > 0; the showcase configs log every epoch).

Every edge is one weight: red = positive, blue = negative; width and opacity grow with |w|.
Every neuron's fill shows its bias the same way (white = 0, no bias = gray outline).
The right panel shows the training/test cost with a marker at the current epoch.
Frames are spaced logarithmically in epochs by default, because weights change most
at the beginning of training.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

POS = np.array([0.80, 0.10, 0.10])  # red
NEG = np.array([0.10, 0.30, 0.85])  # blue


def load_layers(run):
    paths = sorted(
        run.glob("weights_layer*.csv"),
        key=lambda p: int(p.stem.removeprefix("weights_layer")),
    )
    if not paths:
        raise SystemExit(
            f"no weights_layer*.csv in {run} (needs output.weights_every > 0)"
        )
    layers = []
    for p in paths:
        header = p.open().readline().strip().split(",")
        rows, cols = (int(v) + 1 for v in header[-1][1:].split("_"))
        data = np.loadtxt(p, delimiter=",", skiprows=1, ndmin=2)
        layers.append((data[:, 0].astype(int), data[:, 1:].reshape(-1, rows, cols)))
    return layers


def pick_frames(epochs, n, spacing):
    if len(epochs) <= n:
        return np.arange(len(epochs))
    if spacing == "log":
        targets = np.unique(np.round(np.geomspace(1, epochs[-1] + 1, n) - 1))
    else:
        targets = np.linspace(epochs[0], epochs[-1], n)
    idx = np.searchsorted(epochs, targets)
    return np.unique(np.clip(idx, 0, len(epochs) - 1))


def layout(sizes, height=1.0):
    """x = layer index, y = neurons spread evenly and centred; big layers are squeezed."""
    biggest = max(sizes)
    pos = []
    for layer, n in enumerate(sizes):
        span = height * (n - 1) / max(biggest - 1, 1)
        ys = np.linspace(span / 2, -span / 2, n) if n > 1 else np.zeros(1)
        pos.append(np.column_stack([np.full(n, float(layer)), ys]))
    return pos


def rgba(values, vmax, min_alpha=0.04):
    t = np.clip(np.abs(values) / vmax, 0, 1)
    colors = np.where(values[..., None] >= 0, POS, NEG)
    return np.column_stack([colors, min_alpha + (1 - min_alpha) * t])


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("run", type=Path, help="seed folder with weights_layer*.csv")
    p.add_argument("--frames", type=int, default=80)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--spacing", choices=["log", "linear"], default="log")
    p.add_argument(
        "--scale",
        choices=["global", "layer"],
        default="layer",
        help="normalise |w| per layer (default; layers have different scales) or over the whole net",
    )
    p.add_argument(
        "--max-edges",
        type=int,
        default=3000,
        help="per layer; keeps the edges with the largest |w| over training (for wide nets)",
    )
    p.add_argument(
        "--max-width", type=float, default=3.0, help="line width of the largest |w|"
    )
    p.add_argument(
        "--hide-below",
        type=float,
        default=0.0,
        help="hide edges with |w| below this fraction of the layer maximum (declutters wide nets)",
    )
    p.add_argument("--out", type=Path)
    args = p.parse_args()

    layers = load_layers(args.run)
    epochs = layers[0][0]
    config_path = args.run.parent / "config.json"
    bias = (
        json.loads(config_path.read_text())["net"]["bias"]
        if config_path.exists()
        else True
    )
    has_bias = [bias] * len(layers)
    sizes = [layers[0][1].shape[1] - int(bias)] + [w.shape[2] for _, w in layers]
    pos = layout(sizes)
    frames = pick_frames(epochs, args.frames, args.spacing)

    edges, segments = [], []
    for l, (_, w) in enumerate(layers):
        weights = w[:, : sizes[l], :]  # [epoch, in, out]
        flat = weights.reshape(len(epochs), -1)
        keep = np.argsort(-np.abs(flat).max(axis=0))[: args.max_edges]
        i, j = np.unravel_index(keep, (sizes[l], sizes[l + 1]))
        segments.append(np.stack([pos[l][i], pos[l + 1][j]], axis=1))
        biases = w[:, sizes[l], :] if has_bias[l] else None  # [epoch, out]
        edges.append((flat[:, keep], biases))
        if len(keep) < flat.shape[1]:
            print(
                f"layer {l}: drawing {len(keep)} of {flat.shape[1]} edges (largest |w|)"
            )

    def vmax_of(values):
        return max(float(np.abs(values[frames]).max()), 1e-12)

    if args.scale == "global":
        g = max(vmax_of(e) for e, _ in edges)
        w_max = [g] * len(edges)
    else:
        w_max = [vmax_of(e) for e, _ in edges]
    b_all = [b for _, b in edges if b is not None]
    b_max = max((vmax_of(b) for b in b_all), default=1.0)

    history_path = args.run / "history.csv"
    history = pd.read_csv(history_path) if history_path.exists() else None

    fig = plt.figure(figsize=(12, 5.5))
    ax = fig.add_axes([0.02, 0.05, 0.62, 0.85])
    ax.set_axis_off()
    ax.set_xlim(-0.3, len(sizes) - 0.7)
    ax.set_ylim(-0.6, 0.6)

    collections = []
    for seg in segments:
        lc = LineCollection(seg, linewidths=0.5)
        ax.add_collection(lc)
        collections.append(lc)

    node_size = np.clip(3000 / max(sizes), 6, 120)
    scatters = []
    for l, xy in enumerate(pos):
        s = ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=node_size,
            c="white",
            edgecolors="0.3",
            linewidths=0.6,
            zorder=3,
        )
        scatters.append(s)
        label = (
            "wejście" if l == 0 else ("wyjście" if l == len(pos) - 1 else f"ukryta {l}")
        )
        ax.text(l, 0.56, f"{label}\n({sizes[l]})", ha="center", va="bottom", fontsize=9)
    title = ax.set_title("", fontsize=12)

    curve_ax = fig.add_axes([0.70, 0.25, 0.28, 0.58])
    marker = None
    if history is not None:
        test_col = "test_in_cost" if "test_in_cost" in history else "test_cost"
        for col, label in (("train_cost", "trening"), (test_col, "test")):
            if col in history:
                curve_ax.plot(
                    history.epoch.clip(lower=1), history[col], label=label, lw=1
                )
        curve_ax.set(
            xscale="log" if args.spacing == "log" else "linear",
            yscale="log",
            xlabel="epoka",
            ylabel="koszt",
            title="przebieg uczenia",
        )
        curve_ax.legend(fontsize=8)
        marker = curve_ax.axvline(1, color="black", lw=1)
    fig.text(
        0.70,
        0.04,
        "krawędź = waga: czerwona > 0, niebieska < 0,\ngrubość i nasycenie ~ |w|\n"
        "wypełnienie neuronu = bias (te same kolory)",
        fontsize=8,
        color="0.3",
    )

    def update(k):
        t = frames[k]
        for lc, (values, _), vm in zip(collections, edges, w_max):
            v = values[t]
            c = rgba(v, vm)
            c[np.abs(v) < args.hide_below * vm, 3] = 0
            lc.set_color(c)
            lc.set_linewidths(0.2 + args.max_width * np.clip(np.abs(v) / vm, 0, 1))
        for l, s in enumerate(scatters):
            if l == 0:
                continue
            b = edges[l - 1][1]
            if b is None:
                s.set_facecolor("0.85")
            else:
                c = rgba(b[t], b_max, min_alpha=0.0)
                c[:, :3] = c[:, :3] * c[:, 3:] + (1 - c[:, 3:])  # blend with white
                c[:, 3] = 1
                s.set_facecolor(c)
        title.set_text(f"{args.run.parent.name}\nepoka {epochs[t]}")
        if marker is not None:
            marker.set_xdata([max(epochs[t], 1)] * 2)
        return collections + scatters

    anim = FuncAnimation(fig, update, frames=len(frames), blit=False)
    out = args.out or args.run / "weights_graph.gif"
    anim.save(out, writer=PillowWriter(fps=args.fps))
    print(
        f"{len(frames)} frames (epochs {epochs[frames[0]]}..{epochs[frames[-1]]}) -> {out}"
    )


if __name__ == "__main__":
    main()
