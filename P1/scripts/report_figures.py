"""Extra report figures that plot_results.py does not make.

  python scripts/report_figures.py

Reads out/study2 (same runs as plot_results.py) and writes into figures/<dataset>/:
  cost_curve.png      training and test cost per epoch of the baseline network (median, IQR over seeds)
  graph_frames.png    frames of seed_0/weights_graph.gif of the 2x8 "logged" network
                      (run scripts/animate_graph.py on it first)
  mnist/arch_curves.png    train and test accuracy per epoch for a few MNIST architectures
  mnist/weights_tiles.png  first-layer weights of the logged 1x128 MNIST network as 28x28 images
                           (needs weights_layer0.csv in its seed_0)
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

OUT = Path("out/study2")
FIGURES = Path("figures")
SEEDS = range(5)
SMALL = {
    "simple": ("softmax_cross_entropy", 260, "test_cost"),
    "three_gauss": ("softmax_cross_entropy", 340, "test_cost"),
    "cube": ("identity_mse", 1000, "test_in_cost"),
    "activation": ("identity_mse", 1000, "test_in_cost"),
}
MNIST_BASE = "mnist__{arch}_relu_softmax_cross_entropy_lr0.1_bs64_bias_ep20"
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})


def histories(name):
    out = []
    for s in SEEDS:
        p = OUT / name / f"seed_{s}" / "history.csv"
        if p.exists():
            h = pd.read_csv(p)
            if np.isfinite(h.select_dtypes("number").to_numpy()).all():
                out.append(h.set_index("epoch"))
    return out


def band(ax, hs, col, label, **kw):
    table = pd.concat([h[col] for h in hs], axis=1)
    med = table.median(axis=1)
    (line,) = ax.plot(med.index, med, label=label, lw=1.2, **kw)
    ax.fill_between(
        med.index,
        table.quantile(0.25, axis=1),
        table.quantile(0.75, axis=1),
        color=line.get_color(),
        alpha=0.2,
    )


def save(fig, dataset, name):
    path = FIGURES / dataset / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(path)


def cost_curves():
    for ds, (head, epochs, test_col) in SMALL.items():
        hs = histories(f"{ds}__32x32_tanh_{head}_lr0.1_bs32_bias_ep{epochs}")
        if not hs:
            continue
        fig, ax = plt.subplots(figsize=(5.5, 4))
        band(ax, hs, "train_cost", "zbiór uczący")
        band(
            ax,
            hs,
            test_col,
            "zbiór testowy" + (" (w zakresie uczenia)" if "in" in test_col else ""),
        )
        ax.set(yscale="log", xlabel="epoka", ylabel="wartość funkcji kosztu")
        ax.legend()
        save(fig, ds, "cost_curve")
    hs = histories(MNIST_BASE.format(arch="128"))
    if hs:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        band(ax, hs, "train_cost", "zbiór uczący")
        band(ax, hs, "test_cost", "zbiór testowy")
        ax.set(
            yscale="log",
            xlabel="epoka",
            ylabel="entropia krzyżowa",
            xlim=(1, None),
            xticks=range(0, 21, 5),
        )
        ax.legend()
        save(fig, "mnist", "cost_curve")


def mnist_arch_curves():
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for k, (arch, label) in enumerate(
        [("linear", "liniowa"), ("128", "1×128"), ("512x512", "2×512")]
    ):
        hs = [h.loc[1:] for h in histories(MNIST_BASE.format(arch=arch))]
        if not hs:
            continue
        color = f"C{k}"
        band(ax, hs, "train_accuracy", f"{label}, uczący", color=color, ls="--")
        band(ax, hs, "test_accuracy", f"{label}, testowy", color=color)
    ax.set(xlabel="epoka", ylabel="accuracy", xticks=range(0, 21, 5))
    ax.legend(fontsize=8, ncol=2)
    save(fig, "mnist", "arch_curves")


def epochs_of_gif_frames(history_epochs, n=80):
    """Same frame choice as animate_graph.py (default: 80 frames, log spacing)."""
    epochs = np.asarray(history_epochs)
    if len(epochs) <= n:
        return epochs
    targets = np.unique(np.round(np.geomspace(1, epochs[-1] + 1, n) - 1))
    idx = np.unique(np.clip(np.searchsorted(epochs, targets), 0, len(epochs) - 1))
    return epochs[idx]


def graph_frames(want=(0, 10, 100, None)):
    for ds, (head, epochs, _) in SMALL.items():
        run = (
            OUT / f"{ds}__8x8_tanh_{head}_lr0.1_bs32_bias_ep{epochs}_logged" / "seed_0"
        )
        gif = run / "weights_graph.gif"
        if not gif.exists():
            continue
        frame_epochs = epochs_of_gif_frames(pd.read_csv(run / "history.csv").epoch)
        g = Image.open(gif)
        if g.n_frames != len(frame_epochs):
            print(
                f"{gif}: {g.n_frames} frames, expected {len(frame_epochs)} - made with non-default options?"
            )
            continue
        picks = [
            len(frame_epochs) - 1
            if e is None
            else int(np.argmin(np.abs(frame_epochs - e)))
            for e in want
        ]
        crops = []
        for k in picks:
            g.seek(k)
            im = g.convert("RGB")
            w, h = im.size
            im = im.crop((0, 0, int(0.62 * w), h))
            im.paste(
                (255, 255, 255), (0, 0, im.size[0], int(0.06 * h))
            )  # config name line
            crops.append(im)
        w, h = crops[0].size
        sheet = Image.new("RGB", (2 * w, 2 * h), "white")
        for i, im in enumerate(crops):
            sheet.paste(im, ((i % 2) * w, (i // 2) * h))
        path = FIGURES / ds / "graph_frames.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(path)
        print(path, "epochs", [int(frame_epochs[k]) for k in picks])


def mnist_weight_tiles(neurons=12):
    """Row 1: weights of the first neurons at epoch 0; rows 2-3: change after 1 and after all epochs."""
    path = (
        OUT
        / (MNIST_BASE.format(arch="128") + "_logged")
        / "seed_0"
        / "weights_layer0.csv"
    )
    if not path.exists():
        print(f"skip weights_tiles: no {path}")
        return
    data = pd.read_csv(path)
    last = int(data.epoch.max())

    def at(e):
        return (
            data[data.epoch == e]
            .drop(columns="epoch")
            .to_numpy()[0]
            .reshape(-1, 128)[:784]
        )  # no bias row

    w0 = at(0)
    rows = [
        ("epoka 0", w0),
        ("zmiana po 1", at(1) - w0),
        (f"zmiana po {last}", at(last) - w0),
    ]
    fig, axes = plt.subplots(
        len(rows), neurons, figsize=(neurons * 0.9, len(rows) * 1.05)
    )
    for r, (name, w) in enumerate(rows):
        for c in range(neurons):
            img = w[:, c].reshape(28, 28)
            v = np.abs(img).max() or 1.0
            ax = axes[r, c]
            ax.imshow(img, cmap="RdBu_r", vmin=-v, vmax=v)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            if c == 0:
                ax.set_ylabel(name, fontsize=8)
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    save(fig, "mnist", "weights_tiles")


if __name__ == "__main__":
    cost_curves()
    mnist_arch_curves()
    graph_frames()
    mnist_weight_tiles()
