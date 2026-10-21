"""Read the saved CSV/JSON files, average the seeds, and draw the figures."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save(figure, filename):
    figure.tight_layout()
    figure.savefig(Path("figures") / filename, dpi=160)
    plt.close(figure)
    figure_files.append(filename)


def results_dir(config, seed=None):
    directory = Path(config["out_dir"]) / config["name"]
    if seed is not None:
        directory = directory / f"seed_{seed}"
    return directory


def errorbars(axis, configs, labels, metric, title):
    means = [summary.loc[c["name"], metric + "_mean"] for c in configs]
    deviations = [summary.loc[c["name"], metric + "_std"] for c in configs]
    axis.errorbar(range(len(configs)), means, yerr=np.nan_to_num(deviations), fmt="o", capsize=4)
    axis.set_xticks(range(len(configs)), labels)
    axis.set(title=title, ylabel=metric)


if __name__ == "__main__":
    configs = json.loads(Path("configs/experiments.json").read_text())
    Path("figures").mkdir(exist_ok=True)
    figure_files = []

    # One row per seed; mean and sample standard deviation per configuration.
    rows, failures = [], []
    for config in configs:
        for seed in config["seeds"]:
            metrics = json.loads((results_dir(config, seed) / "metrics.json").read_text())
            if "error" in metrics:
                failures.append({"name": config["name"], "seed": seed, "error": metrics["error"]})
                continue
            row = {"name": config["name"], "seed": seed}
            for split in ("train", "val", "test"):
                for key in ("accuracy", "mse", "mae", "macro_f1"):
                    if key in metrics.get(split, {}):
                        row[f"{split}_{key}"] = metrics[split][key]
            rows.append(row)
    runs = pd.DataFrame(rows)
    runs.to_csv("figures/runs.csv", index=False)
    summary = runs.drop(columns="seed").groupby("name").agg(["mean", "std", "count"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary.to_csv("figures/summary.csv")
    if failures:
        pd.DataFrame(failures).to_csv("figures/failures.csv", index=False)
        raise SystemExit("Some runs failed; see figures/failures.csv before plotting comparisons.")
    Path("figures/failures.csv").unlink(missing_ok=True)

    for dataset in ("simple", "three_gauss", "cube", "activation"):
        experiments = [c for c in configs if c["name"].startswith(dataset + "_")]
        classification = experiments[0]["data"]["task"] == "classification"
        head = "softmax_cross_entropy" if classification else "identity_mse"
        metric = "test_accuracy" if classification else "test_mse"
        baseline = next(c for c in experiments if c["net"] == {
            "hidden": [32, 32], "activation": "tanh", "head": head, "bias": True})
        activations = [next(c for c in experiments if c["net"]["hidden"] == [32, 32]
                           and c["net"]["activation"] == activation and c["net"]["head"] == head)
                       for activation in ("sigmoid", "tanh", "relu")]
        points = pd.read_csv(results_dir(baseline) / "points_train.csv")

        # Predictions: decision regions in 2D, or the learned function in 1D.
        if classification:
            fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
            for axis, config in zip(axes, activations):
                grid = pd.read_csv(results_dir(config, 0) / "grid.csv")
                steps = int(np.sqrt(len(grid)))
                classes = int(points.label.max())
                colors = plt.get_cmap("tab10", classes)
                axis.pcolormesh(grid.x1.to_numpy().reshape(steps, steps), grid.x2.to_numpy().reshape(steps, steps),
                                grid["class"].to_numpy().reshape(steps, steps), cmap=colors, vmin=1, vmax=classes, alpha=0.25)
                axis.scatter(points.x1, points.x2, c=points.label, cmap=colors, vmin=1, vmax=classes, s=5)
                axis.set(title=config["net"]["activation"], xlabel="x₁", ylabel="x₂")
        else:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            for axis in axes:
                truth = pd.read_csv(results_dir(baseline, 0) / "pred.csv").sort_values("x")
                axis.plot(truth.x, truth.y_true, color="black", label="cel", linewidth=2)
                axis.scatter(points.x1, points.label, color="black", s=4, alpha=0.3, label="dane uczące")
                for config in activations:
                    pred = pd.read_csv(results_dir(config, 0) / "pred.csv").sort_values("x")
                    axis.plot(pred.x, pred.y_pred, label=config["net"]["activation"])
                axis.axvspan(points.x1.min(), points.x1.max(), alpha=0.08, color="green")
                axis.set(xlabel="x", ylabel="y")
                axis.legend(fontsize=8)
            axes[0].set_title("Cały zakres testu")
            axes[1].set(title="Powiększenie zakresu uczenia", xlim=(points.x1.min(), points.x1.max()),
                        ylim=(points.label.min() - 5, points.label.max() + 5))
        save(fig, f"{dataset}_predictions.png")

        # Training/validation loss, averaged over the three seeds.
        fig, axis = plt.subplots(figsize=(8, 4))
        for config in activations:
            histories = [pd.read_csv(results_dir(config, seed) / "history.csv") for seed in config["seeds"]]
            history = pd.concat(histories).groupby("epoch").mean()
            line, = axis.plot(history.index, history.train_cost, label=config["net"]["activation"] + " train")
            axis.plot(history.index, history.val_cost, "--", color=line.get_color(), label=config["net"]["activation"] + " val")
        axis.set(xlabel="Epoka", ylabel="Koszt (skala log.)", yscale="log", title=dataset)
        axis.legend(ncol=2)
        save(fig, f"{dataset}_learning.png")

        # Depth: 0–4 hidden layers, width 32. Width: 8 versus 32 in one layer.
        architectures = [c for c in experiments if c["net"]["activation"] == "tanh" and c["net"]["head"] == head]
        depths = [c for c in architectures if c["net"]["hidden"] != [8]]
        widths = [c for c in architectures if len(c["net"]["hidden"]) == 1]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        errorbars(axes[0], depths, [str(len(c["net"]["hidden"])) for c in depths], metric, "Liczba warstw ukrytych")
        errorbars(axes[1], widths, [str(c["net"]["hidden"][0]) for c in widths], metric, "Neurony w jednej warstwie")
        save(fig, f"{dataset}_architecture.png")

        losses = [c for c in experiments if c["net"]["hidden"] == [32, 32] and c["net"]["activation"] == "tanh"]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        errorbars(axes[0], activations, [c["net"]["activation"] for c in activations], metric, "Aktywacja — [32, 32]")
        errorbars(axes[1], losses, [c["net"]["head"].replace("softmax_", "").replace("identity_", "") for c in losses], metric, "Koszt — [32, 32], tanh")
        save(fig, f"{dataset}_activation_loss.png")

        # Eight individual connections per layer, baseline model, seed 0.
        fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
        for layer, inputs in enumerate([2 if classification else 1, 32, 32]):
            weights = pd.read_csv(results_dir(baseline, 0) / f"weights_layer{layer}.csv")
            columns = [c for c in weights if c != "epoch" and not c.startswith(f"w{inputs}_")]
            for column in columns[:8]:
                axes[layer].plot(weights.epoch, weights[column], label=column)
            axes[layer].set(title=f"Warstwa {layer}", ylabel="Wartość wagi")
            axes[layer].legend(ncol=4, fontsize=7)
        axes[-1].set_xlim(left=0)
        axes[-1].set_xlabel("Epoka")
        save(fig, f"{dataset}_weights.png")

    # MNIST: test confusion matrix of seed 0 and the two training curves.
    config = next(c for c in configs if c["name"] == "mnist")
    metrics = json.loads((results_dir(config, 0) / "metrics.json").read_text())
    confusion = np.array(metrics["test"]["confusion"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    image = axes[0].imshow(confusion, cmap="Blues")
    fig.colorbar(image, ax=axes[0])
    axes[0].set(title="MNIST: pomyłki, seed 0", xlabel="Przewidziana cyfra", ylabel="Prawdziwa cyfra")
    axes[0].set_xticks(range(10))
    axes[0].set_yticks(range(10))
    for seed in config["seeds"]:
        history = pd.read_csv(results_dir(config, seed) / "history.csv")
        axes[1].plot(history.epoch, history.train_cost, label=f"seed {seed}")
    axes[1].set(title="MNIST: uczenie", xlabel="Epoka", ylabel="Koszt train", yscale="log")
    axes[1].legend()
    save(fig, "mnist.png")

    # A browsable list of images, without writing a report or conclusions.
    gallery = "# Figury do przejrzenia\n\nWyniki: [średnie i odchylenia](summary.csv), [poszczególne ziarna](runs.csv).\n\n"
    gallery += "Słupki błędu: odchylenie standardowe między ziarnami. Predykcje i wagi: seed 0.\n\n"
    gallery += "Wagi są zapisywane co epokę i przedstawione na liniowej osi x, począwszy od epoki 0.\n\n"
    for filename in figure_files:
        gallery += f"## {filename}\n\n![{filename}]({filename})\n\n"
    Path("figures/README.md").write_text(gallery)
    print(f"Saved {len(figure_files)} figures. Open figures/README.md to review them.")
