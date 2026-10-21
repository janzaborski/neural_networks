"""Visualize one saved 1D regression or 2D classification experiment."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def visualize(experiment, seed, output):
    config = json.loads((experiment / "config.json").read_text())
    run = experiment / f"seed_{seed}"
    metrics = json.loads((run / "metrics.json").read_text())
    if "error" in metrics:
        raise ValueError(f"Training failed: {metrics['error']}")
    if not (experiment / "points_train.csv").exists():
        raise ValueError("This script supports the assignment's 1D regression and 2D classification datasets.")
    weight_files = sorted(run.glob("weights_layer*.csv"))
    if len(weight_files) != len(config["net"]["hidden"]) + 1:
        raise ValueError("Missing weight logs. Set train.log_every=1 and output.weights_every=1 in the config, "
                         "then rerun this experiment in Rust before visualizing it.")
    delta_files = [run / f"deltas_layer{layer}.csv" for layer in range(len(weight_files))]
    if not all(path.exists() for path in delta_files):
        raise ValueError("Missing backpropagation logs. Set output.log_deltas=true and train.log_every=1, "
                         "then rerun this experiment in Rust before visualizing it.")
    points = pd.read_csv(experiment / "points_train.csv")
    history = pd.read_csv(run / "history.csv")
    classification = config["data"]["task"] == "classification"
    if points.shape[1] != (3 if classification else 2):
        raise ValueError("Expected two input features for classification or one for regression.")
    output.mkdir(parents=True, exist_ok=True)

    def save(figure, filename):
        figure.suptitle(f"{config['name']} — seed {seed}", fontsize=11)
        figure.tight_layout()
        figure.savefig(output / filename, dpi=160)
        plt.close(figure)

    # 1. Training data and the network's final predictions.
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].set_title("Zbiór uczący")
    axes[1].set_title("Wynik klasyfikacji" if classification else "Wynik regresji")
    if classification:
        grid = pd.read_csv(run / "grid.csv")
        steps = int(np.sqrt(len(grid)))
        classes = int(points.label.max())
        colors = plt.get_cmap("tab10", classes)
        axes[1].pcolormesh(grid.x1.to_numpy().reshape(steps, steps), grid.x2.to_numpy().reshape(steps, steps),
                           grid["class"].to_numpy().reshape(steps, steps), cmap=colors,
                           vmin=1, vmax=classes, alpha=0.25, shading="auto")
        for axis in axes:
            dots = axis.scatter(points.x1, points.x2, c=points.label, cmap=colors,
                                vmin=1, vmax=classes, s=7)
            handles, labels = dots.legend_elements()
            axis.legend(handles, labels, title="Klasa", fontsize=8)
            axis.set(xlabel="x₁", ylabel="x₂")
    else:
        predictions = pd.read_csv(run / "pred.csv").sort_values("x")
        axes[0].scatter(points.x1, points.label, s=8)
        axes[1].plot(predictions.x, predictions.y_true, color="black", label="Wartość rzeczywista")
        axes[1].plot(predictions.x, predictions.y_pred, label="Predykcja sieci")
        axes[1].scatter(points.x1, points.label, s=4, alpha=0.3, color="black", label="Dane uczące")
        axes[1].axvspan(points.x1.min(), points.x1.max(), color="green", alpha=0.08, label="Zakres uczenia")
        axes[1].legend(fontsize=8)
        for axis in axes:
            axis.set(xlabel="x", ylabel="y")
    save(figure, "data_and_predictions.png")

    # 2. Actual backpropagation signals for one fixed training example.
    figure, axes = plt.subplots(len(delta_files), 2, figsize=(12, 3 * len(delta_files)), squeeze=False)
    # Display the output layer first: this is the direction of backpropagation.
    for row, layer in enumerate(reversed(range(len(delta_files)))):
        deltas = pd.read_csv(delta_files[layer])
        signals = deltas.drop(columns="epoch")
        label = "Wyjście" if layer == len(delta_files) - 1 else f"Warstwa ukryta {layer + 1}"
        for column in signals.columns[:8]:
            axes[row, 0].plot(deltas.epoch, deltas[column], label=column.replace("delta_", "δ"))
        axes[row, 0].set(title=f"{label}: do 8 sygnałów δ", xlabel="Epoka", ylabel="δ = ∂ℓ/∂z")
        axes[row, 0].axhline(0, color="black", linewidth=0.5)
        axes[row, 0].legend(ncol=4, fontsize=7)
        limit = max(float(np.abs(signals.to_numpy()).max()), 1e-12)
        # Explicit cell edges also give a single output neuron a visible row.
        epochs = deltas.epoch.to_numpy()
        edges = np.r_[epochs[0] - 0.5, (epochs[:-1] + epochs[1:]) / 2, epochs[-1] + 0.5]
        image = axes[row, 1].pcolormesh(edges, np.arange(len(signals.columns) + 1) - 0.5, signals.to_numpy().T,
                                       shading="flat", cmap="RdBu_r", vmin=-limit, vmax=limit, rasterized=True)
        axes[row, 1].set(title=f"{label}: wszystkie neurony", xlabel="Epoka", ylabel="Indeks neuronu")
        if len(signals.columns) <= 8:
            axes[row, 1].set_yticks(range(len(signals.columns)))
        figure.colorbar(image, ax=axes[row, 1], label="δ = ∂ℓ/∂z")
        for axis in axes[row]:
            axis.set_xlim(deltas.epoch.iloc[0], deltas.epoch.iloc[-1])
    save(figure, "backpropagation.png")

    # 3. Loss is a separate quantity from the propagated signals above.
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(history.epoch, history.train_cost, label="Uczenie")
    if "val_cost" in history:
        axis.plot(history.epoch, history.val_cost, label="Walidacja")
    axis.set(xlabel="Epoka", ylabel="Wartość funkcji kosztu", title=config["net"]["head"])
    axis.legend()
    save(figure, "loss.png")

    # 4. Weight trajectories and a heatmap containing every recorded parameter.
    figure, axes = plt.subplots(len(weight_files), 2, figsize=(12, 3 * len(weight_files)), squeeze=False)
    inputs = [points.shape[1] - 1, *config["net"]["hidden"]]
    for layer, path in enumerate(weight_files):
        weights = pd.read_csv(path)
        parameters = weights.drop(columns="epoch")
        connections = [column for column in parameters if not column.startswith(f"w{inputs[layer]}_")]
        for column in connections[:8]:
            axes[layer, 0].plot(weights.epoch, weights[column], label=column)
        axes[layer, 0].set(title=f"Warstwa {layer}: do 8 pierwszych wag", xlabel="Epoka", ylabel="Wartość wagi")
        axes[layer, 0].legend(ncol=4, fontsize=7)
        limit = max(float(np.abs(parameters.to_numpy()).max()), 1e-12)
        epochs = weights.epoch.to_numpy()
        edges = np.r_[epochs[0] - 0.5, (epochs[:-1] + epochs[1:]) / 2, epochs[-1] + 0.5]
        image = axes[layer, 1].pcolormesh(edges, np.arange(len(parameters.columns) + 1) - 0.5, parameters.to_numpy().T,
                                         shading="flat", cmap="RdBu_r", vmin=-limit, vmax=limit, rasterized=True)
        axes[layer, 1].set(title=f"Warstwa {layer}: wszystkie wagi i biasy", xlabel="Epoka", ylabel="Indeks parametru")
        figure.colorbar(image, ax=axes[layer, 1], label="Wartość parametru")
        for axis in axes[layer]:
            axis.set_xlim(weights.epoch.iloc[0], weights.epoch.iloc[-1])
    save(figure, "weights.png")

    names = ["data_and_predictions.png", "backpropagation.png", "loss.png", "weights.png"]
    index = f"# {config['name']} — seed {seed}\n\n"
    index += "Dane i predykcje, propagowane sygnały δ, osobno koszt oraz wagi. Osie epok są liniowe.\n\n"
    index += "δ = ∂ℓ/∂z: pochodna kosztu jednej próbki względem wejścia neuronu przed aktywacją. To nie wartość funkcji kosztu.\n\n"
    index += "W każdej zapisanej epoce Rust ponownie wykonuje forward/backward na tej samej pierwszej próbce ze zbioru uczącego (pierwszy wiersz points_train.csv). Nie aktualizuje wtedy wag i nie używa testu.\n\n"
    index += "Przepływ wstecz pokazano od góry: wyjście → kolejne warstwy ukryte. Mapy zawierają wszystkie neurony; każda warstwa ma własną skalę kolorów.\n\n"
    index += "Koszt regresji dotyczy celu po normalizacji. Wagi są migawkami po epokach, nie po każdej partii SGD.\n\n"
    index += "Na mapie wag każdy wiersz odpowiada jednej kolumnie pliku weights_layer*.csv, bez kolumny epoch.\n\n"
    for name in names:
        index += f"## {name}\n\n![{name}]({name})\n\n"
    (output / "README.md").write_text(index)
    print(f"Saved visualizations: {output / 'README.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path, help="Results folder containing config.json and seed_* directories")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    output = ROOT / "figures/single" / experiment.name / f"seed_{args.seed}"
    try:
        visualize(experiment, args.seed, output)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Error: {error}\n")
