"""Aggregate all finished runs listed in configs/manifest.json and draw the study figures.

  python scripts/plot_results.py

Methodology:
  - no hyperparameter selection: every model is trained with the parameters in its config;
  - the "final" value of a metric is its mean over the last K epochs of history.csv
    (K = avg_last from the manifest), because with constant-lr SGD the last epoch alone
    is a random point of the oscillations;
  - across seeds we show the median and the interquartile range (25–75%);
  - diverged runs are left out and marked with a red ✗ and the number of such seeds;
  - regression: test_in = test points inside the training x-range (fit),
    test_out = outside it (extrapolation).
"""

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIGURES = Path("figures")
SMALL = ["simple", "three_gauss", "cube", "activation"]
BASE_LR = {
    "simple": 0.1,
    "three_gauss": 0.1,
    "cube": 0.1,
    "activation": 0.1,
    "mnist": 0.1,
}

LABELS = {
    "test_accuracy": "accuracy (test)",
    "test_macro_f1": "macro-F1 (test)",
    "train_accuracy": "accuracy (trening)",
    "test_in_mse": "MSE (test w zakresie uczenia)",
    "test_in_mae": "MAE (test w zakresie uczenia)",
    "test_out_mse": "MSE (test poza zakresem, ekstrapolacja)",
    "train_mse": "MSE (trening)",
    "test_cost": "koszt (test)",
    "test_in_cost": "koszt (test w zakresie uczenia)",
}


def main_metrics(task):
    if task == "classification":
        return ["test_accuracy", "test_macro_f1"]
    return ["test_in_mse", "test_in_mae", "test_out_mse"]


def run_dir(name):
    return Path(manifest[name].get("out_dir", "out/study2")) / name


def load():
    rows = []
    for name, entry in manifest.items():
        d = run_dir(name)
        if not (d / "config.json").exists():
            continue
        config = json.loads((d / "config.json").read_text())
        k = entry["params"].get("avg_last", 1)
        for seed in config["seeds"]:
            m_path = d / f"seed_{seed}" / "metrics.json"
            if not m_path.exists():
                continue
            metrics = json.loads(m_path.read_text())
            row = {
                "name": name,
                "dataset": entry["dataset"],
                "task": entry["task"],
                "studies": ",".join(entry["studies"]),
                "seed": seed,
                **{p: v for p, v in entry["params"].items() if p != "hidden"},
                "failed": "error" in metrics,
                "error": metrics.get("error", ""),
                "seconds": metrics.get("seconds", np.nan),
            }
            if not row["failed"]:
                h = pd.read_csv(d / f"seed_{seed}" / "history.csv")
                tail = h[h.epoch > h.epoch.max() - k]
                for col in h.columns:
                    if col != "epoch":
                        row[col] = tail[col].mean()
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(runs):
    ok = runs[~runs.failed]
    cols = [c for c in ok.columns if c.startswith(("train_", "test")) or c == "seconds"]
    g = ok.groupby("name")[cols]
    parts = {
        "median": g.median(),
        "q25": g.quantile(0.25),
        "q75": g.quantile(0.75),
        "mean": g.mean(),
        "std": g.std(),
    }
    stats = pd.concat({k: v for k, v in parts.items()}, axis=1)
    stats.columns = [f"{c}_{s}" for s, c in stats.columns]
    keys = [
        "dataset",
        "task",
        "studies",
        "L",
        "B",
        "activation",
        "head",
        "lr",
        "batch",
        "bias",
        "epochs",
        "avg_last",
    ]
    info = runs.groupby("name")[keys].first()
    counts = runs.groupby("name").failed.agg(
        n_ok=lambda f: int((~f).sum()), n_failed=lambda f: int(f.sum())
    )
    return info.join(counts).join(stats)


def in_study(summary, dataset, study):
    mask = (summary.dataset == dataset) & summary.studies.str.split(",").apply(
        lambda s: study in s
    )
    return summary[mask].copy()


# ---------------------------------------------------------------------------------------------
# Every figure has exactly one plot, so each can be referenced on its own in the report.
# Files go to figures/<dataset>/<name>.png; figures/figures.tex holds a ready \begin{figure}
# block with \label{fig:<dataset>:<name>} for every file, and figures/README.md a gallery.
# ---------------------------------------------------------------------------------------------

FIGSIZE = (5.2, 3.8)
NICE = {
    "simple": "simple",
    "three_gauss": "three\\_gauss",
    "cube": "cube",
    "activation": "activation",
    "mnist": "MNIST",
}


def new_figure(size=FIGSIZE):
    fig, ax = plt.subplots(figsize=size)
    return fig, ax


def save(fig, dataset, name, caption):
    (FIGURES / dataset).mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / dataset / f"{name}.png", dpi=170)
    plt.close(fig)
    gallery.append((dataset, name, caption))


def band(ax, x, rows, metric, label, **kw):
    """Median line with an interquartile band; ✗n above points where n seeds diverged."""
    (line,) = ax.plot(x, rows[f"{metric}_median"], marker="o", label=label, **kw)
    ax.fill_between(
        x,
        rows[f"{metric}_q25"],
        rows[f"{metric}_q75"],
        color=line.get_color(),
        alpha=0.2,
    )
    for xi, (_, r) in zip(x, rows.iterrows()):
        if r.n_failed:
            ax.annotate(
                f"✗{r.n_failed}",
                (xi, 1.0),
                xycoords=("data", "axes fraction"),
                ha="center",
                va="top",
                color=line.get_color(),
                fontsize=8,
                fontweight="bold",
            )
    return line


def log_scale(metric):
    return "mse" in metric or "mae" in metric


def label(metric):
    return LABELS.get(metric, metric)


def curves(ax, rows, label_of, metric):
    """Median over non-diverged seeds of `metric` against epoch, one line per config row."""
    for name, r in rows.iterrows():
        hs = []
        for seed_dir in sorted(run_dir(name).glob("seed_*")):
            m_path = seed_dir / "metrics.json"
            if not m_path.exists() or "error" in json.loads(m_path.read_text()):
                continue
            hs.append(pd.read_csv(seed_dir / "history.csv")[["epoch", metric]])
        if not hs:
            continue
        g = pd.concat(hs).groupby("epoch")[metric]
        med = g.median()
        (line,) = ax.plot(med.index, med, label=label_of(r), lw=1.2)
        ax.fill_between(
            med.index,
            g.quantile(0.25),
            g.quantile(0.75),
            color=line.get_color(),
            alpha=0.15,
        )
    ax.set(xlabel="epoka", ylabel=label(metric))
    if log_scale(metric):
        ax.set_yscale("log")
    ax.legend(fontsize=8)


# --- architecture ----------------------------------------------------------------------------


def arch_figures(dataset, rows, task):
    metrics = (
        ["test_accuracy"]
        if task == "classification"
        else ["test_in_mse", "test_out_mse"]
    )
    depths = sorted(rows.L.unique())
    widths = sorted(b for b in rows.B.unique() if b > 0)
    for m in metrics:
        grid = np.full((len(depths), len(widths)), np.nan)
        for _, r in rows.iterrows():
            for j in range(len(widths)) if r.L == 0 else [widths.index(r.B)]:
                grid[depths.index(r.L), j] = r[f"{m}_median"]
        shown = np.log10(grid) if log_scale(m) else grid
        fig, ax = new_figure()
        im = ax.imshow(
            shown, cmap="viridis_r" if log_scale(m) else "viridis", aspect="auto"
        )
        for i in range(len(depths)):
            for j in range(len(widths)):
                if not np.isnan(grid[i, j]):
                    rgb = im.cmap(im.norm(shown[i, j]))[:3]
                    color = (
                        "black"
                        if 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2] > 0.5
                        else "white"
                    )
                    ax.text(
                        j,
                        i,
                        f"{grid[i, j]:.3g}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=color,
                    )
        ax.set_xticks(range(len(widths)), widths)
        ax.set_yticks(
            range(len(depths)), [f"{d} (liniowa)" if d == 0 else str(d) for d in depths]
        )
        ax.set(xlabel="neurony w warstwie (B)", ylabel="warstwy ukryte (L)")
        fig.colorbar(im, ax=ax, label=label(m) + (" [log₁₀]" if log_scale(m) else ""))
        save(
            fig,
            dataset,
            f"arch_heatmap_{m}",
            f"{NICE[dataset]}: {label(m)} (mediana) w zależności od liczby warstw ukrytych $L$ "
            f"i neuronów w warstwie $B$.",
        )

        fig, ax = new_figure()
        for L, sub in rows.groupby("L"):
            sub = sub.sort_values("B")
            if L == 0:
                ax.axhline(
                    sub[f"{m}_median"].iloc[0],
                    color="gray",
                    ls="--",
                    label="L=0 (liniowa)",
                )
            else:
                band(ax, sub.B, sub, m, f"L={L}")
        ax.set_xscale("log", base=2)
        if log_scale(m):
            ax.set_yscale("log")
        ax.set(xlabel="neurony w warstwie (B)", ylabel=label(m))
        ax.legend(fontsize=8)
        save(
            fig,
            dataset,
            f"arch_lines_{m}",
            f"{NICE[dataset]}: {label(m)} w zależności od $B$ dla każdej liczby warstw $L$ "
            f"(mediana i kwartyle po ziarnach).",
        )


# --- activation / loss x learning rate -------------------------------------------------------


def sweep_figures(dataset, rows, task, study, group, what):
    if task == "classification":
        metrics = ["test_accuracy", "test_macro_f1"]
    elif study == "activation":
        metrics = ["test_in_mse", "test_out_mse"]
    else:
        metrics = ["test_in_mse", "test_in_mae"]
    for m in metrics:
        fig, ax = new_figure()
        for g, sub in rows.groupby(group, sort=False):
            sub = sub.sort_values("lr")
            band(ax, sub.lr, sub, m, str(g))
        ax.set_xscale("log")
        if log_scale(m):
            ax.set_yscale("log")
        ax.set(xlabel="współczynnik uczenia", ylabel=label(m))
        ax.legend(fontsize=8)
        save(
            fig,
            dataset,
            f"{study}_lr_{m}",
            f"{NICE[dataset]}: {label(m)} dla różnych {what} w zależności od współczynnika uczenia "
            f"(mediana i kwartyle; liczba przy krzyżyku nad wykresem to liczba rozbieżnych ziaren).",
        )

    lr = BASE_LR[dataset]
    curve_metrics = (
        ["test_accuracy", "test_cost"] if task == "classification" else ["test_in_mse"]
    )
    if study == "loss" and task == "classification":
        curve_metrics = [
            "test_accuracy"
        ]  # costs of different losses are not comparable
    for m in curve_metrics:
        fig, ax = new_figure()
        curves(ax, rows[rows.lr == lr], lambda r: str(r[group]), m)
        save(
            fig,
            dataset,
            f"{study}_curve_{m}",
            f"{NICE[dataset]}: przebieg {label(m)} w trakcie uczenia dla różnych {what} "
            f"(współczynnik uczenia {lr:g}).",
        )


# --- bias --------------------------------------------------------------------------------------


def bias_figure(dataset, rows, task):
    m = "test_accuracy" if task == "classification" else "test_in_mse"
    rows = rows.sort_values(["L", "B"])
    archs = list(dict.fromkeys(f"{L}×{B}" for L, B in zip(rows.L, rows.B)))
    fig, ax = new_figure()
    for k, bias in enumerate([True, False]):
        sub = rows[rows.bias == bias]
        x = np.arange(len(sub)) + (k - 0.5) * 0.35
        med = sub[f"{m}_median"].to_numpy()
        err = np.vstack([med - sub[f"{m}_q25"], sub[f"{m}_q75"] - med])
        ax.bar(
            x,
            med,
            0.35,
            yerr=np.nan_to_num(err),
            capsize=3,
            label="z biasami" if bias else "bez biasów",
        )
    ax.set_xticks(range(len(archs)), [f"L×B = {a}" for a in archs])
    if log_scale(m):
        ax.set_yscale("log")
    elif task == "classification":
        lo = np.nanmin(rows[f"{m}_q25"])
        ax.set_ylim(max(0, lo - 0.05), 1)
    ax.set(ylabel=label(m))
    ax.legend(fontsize=8)
    save(
        fig,
        dataset,
        "bias",
        f"{NICE[dataset]}: {label(m)} sieci z biasami i bez biasów.",
    )


# --- predictions and confusion matrices (seed 0 / sum over seeds, baseline lr) -----------------


def prediction_figures(dataset, rows, task):
    rows = rows[rows.lr == BASE_LR[dataset]]
    for name, r in rows.iterrows():
        d, act = run_dir(name), r.activation
        if not (d / "points_train.csv").exists():
            continue
        points = pd.read_csv(d / "points_train.csv")
        fig, ax = new_figure((4.8, 4.2) if task == "classification" else FIGSIZE)
        if task == "classification":
            grid_path, pred_path = d / "seed_0" / "grid.csv", d / "seed_0" / "pred.csv"
            if not grid_path.exists():
                plt.close(fig)
                continue
            grid = pd.read_csv(grid_path)
            n = int(round(np.sqrt(len(grid))))
            k = int(points.label.max())
            cmap = plt.get_cmap("tab10", k)
            ax.pcolormesh(
                grid.x1.to_numpy().reshape(n, n),
                grid.x2.to_numpy().reshape(n, n),
                grid["class"].to_numpy().reshape(n, n),
                cmap=cmap,
                vmin=0.5,
                vmax=k + 0.5,
                alpha=0.25,
                shading="auto",
            )
            extra = ""
            if pred_path.exists():
                test = pd.read_csv(pred_path)
                wrong = test["true"] != test["pred"]
                ax.scatter(
                    test.x1,
                    test.x2,
                    c=test["true"],
                    cmap=cmap,
                    vmin=0.5,
                    vmax=k + 0.5,
                    s=4,
                )
                ax.scatter(
                    test.x1[wrong],
                    test.x2[wrong],
                    marker="x",
                    color="black",
                    s=14,
                    lw=0.8,
                    label=f"pomyłki: {wrong.sum()}/{len(test)}",
                )
                ax.legend(fontsize=8, loc="lower right")
                extra = (
                    f" Punkty testowe w kolorze prawdziwej klasy, × oznacza pomyłkę "
                    f"(accuracy {1 - wrong.mean():.3f})."
                )
            else:
                ax.scatter(
                    points.x1,
                    points.x2,
                    c=points.label,
                    cmap=cmap,
                    vmin=0.5,
                    vmax=k + 0.5,
                    s=4,
                )
            ax.set(xlabel="$x_1$", ylabel="$x_2$")
            caption = f"{NICE[dataset]}: obszary decyzyjne sieci z aktywacją {act} (ziarno 0).{extra}"
        else:
            pred_path = d / "seed_0" / "pred.csv"
            if not pred_path.exists():
                plt.close(fig)
                continue
            pred = pd.read_csv(pred_path).sort_values("x")
            ax.plot(pred.x, pred.y_true, color="black", lw=2, label="funkcja (test)")
            ax.plot(pred.x, pred.y_pred, color="tab:red", label="sieć")
            ax.scatter(
                points.x1,
                points.label,
                s=3,
                alpha=0.3,
                color="gray",
                label="dane uczące",
            )
            ax.axvspan(
                points.x1.min(),
                points.x1.max(),
                color="green",
                alpha=0.07,
                label="zakres uczenia",
            )
            lo, hi = pred.y_true.min(), pred.y_true.max()
            ax.set(
                xlabel="$x$",
                ylabel="$y$",
                ylim=(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo)),
            )
            ax.legend(fontsize=7)
            caption = (
                f"{NICE[dataset]}: funkcja nauczona przez sieć z aktywacją {act} (ziarno 0) na całym "
                f"zakresie testu; zielone tło to zakres danych uczących."
            )
        save(fig, dataset, f"predictions_{act}", caption)


def confusion_figures(dataset, rows):
    rows = rows[rows.lr == BASE_LR[dataset]]
    for name, r in rows.iterrows():
        total, n_seeds = None, 0
        for m_path in sorted(run_dir(name).glob("seed_*/metrics.json")):
            metrics = json.loads(m_path.read_text())
            if "error" in metrics or "test" not in metrics:
                continue
            c = np.array(metrics["test"]["confusion"])
            total = c if total is None else total + c
            n_seeds += 1
        if total is None:
            continue
        share = total / total.sum(axis=1, keepdims=True).clip(min=1)
        k = len(total)
        fig, ax = new_figure((4.6, 4.0) if k <= 3 else (6.5, 5.5))
        im = ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for i in range(k):
            for j in range(k):
                text = (
                    f"{total[i, j]}\n{share[i, j]:.1%}" if k <= 3 else f"{total[i, j]}"
                )
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=8 if k <= 3 else 6,
                    color="white" if share[i, j] > 0.5 else "black",
                )
        first = 0 if dataset == "mnist" else 1
        ax.set_xticks(range(k), [str(t + first) for t in range(k)])
        ax.set_yticks(range(k), [str(t + first) for t in range(k)])
        ax.set(xlabel="przewidziana klasa", ylabel="prawdziwa klasa")
        fig.colorbar(im, ax=ax, label="udział w prawdziwej klasie")
        arch = "x".join(map(str, [r.B] * int(r.L))) or "linear"
        save(
            fig,
            dataset,
            f"confusion_{r.activation}_{arch}",
            f"{NICE[dataset]}: macierz pomyłek na zbiorze testowym, aktywacja {r.activation}, "
            f"architektura {arch}, suma po {n_seeds} ziarnach (accuracy {np.trace(total) / total.sum():.3f}).",
        )


# --- MNIST extras -----------------------------------------------------------------------------


def mnist_time_figure(arch):
    fig, ax = new_figure()
    for L, sub in arch[arch.L > 0].groupby("L"):
        sub = sub.sort_values("B")
        ax.plot(sub.B, sub.seconds_median / sub.epochs, marker="o", label=f"L={L}")
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="neurony w warstwie (B)",
        ylabel="sekundy na epokę",
    )
    ax.legend(fontsize=8)
    save(
        fig,
        "mnist",
        "time_per_epoch",
        "MNIST: czas jednej epoki (uczenie i ewaluacja, jeden wątek) w zależności od architektury.",
    )


# --- output ------------------------------------------------------------------------------------


def write_index(runs):
    failed = int(runs.failed.sum())
    md = [
        "# Wyniki",
        "",
        "Wartość końcowa = średnia z ostatnich K epok; linia = mediana po ziarnach, pas = kwartyle; "
        f"✗n = rozbieżne ziarna (lista w failures.csv, razem {failed}). Tabele: summary.csv, runs.csv.",
        "",
    ]
    tex = [
        "% Generated by scripts/plot_results.py. \\input{figures/figures.tex} or copy single blocks.",
        "",
    ]
    for dataset, name, caption in gallery:
        path = f"{dataset}/{name}.png"
        md += [
            f"### `fig:{dataset}:{name}`",
            "",
            caption.replace("\\_", "_"),
            "",
            f"![{name}]({path})",
            "",
        ]
        cap = re.sub(r"(?<!\\)_", r"\\_", caption).replace("%", "\\%")
        tex += [
            "\\begin{figure}[htbp]",
            "  \\centering",
            f"  \\includegraphics[width=0.7\\linewidth]{{figures/{path}}}",
            f"  \\caption{{{cap}}}",
            f"  \\label{{fig:{dataset}:{name}}}",
            "\\end{figure}",
            "",
        ]
    (FIGURES / "README.md").write_text("\n".join(md))
    (FIGURES / "figures.tex").write_text("\n".join(tex))


if __name__ == "__main__":
    manifest = json.loads(Path("configs/manifest.json").read_text())
    FIGURES.mkdir(exist_ok=True)
    runs = load()
    if runs.empty:
        raise SystemExit("No finished runs found.")
    runs.to_csv(FIGURES / "runs.csv", index=False)
    summary = summarize(runs)
    summary.to_csv(FIGURES / "summary.csv")
    runs[runs.failed][["name", "seed", "error"]].to_csv(
        FIGURES / "failures.csv", index=False
    )

    gallery = []
    for dataset in [*SMALL, "mnist"]:
        task = "regression" if dataset in ("cube", "activation") else "classification"
        rows = in_study(summary, dataset, "arch")
        if not rows.empty:
            arch_figures(dataset, rows, task)
            if dataset == "mnist":
                mnist_time_figure(rows)
        rows = in_study(summary, dataset, "activation")
        if not rows.empty:
            sweep_figures(
                dataset, rows, task, "activation", "activation", "funkcji aktywacji"
            )
            prediction_figures(dataset, rows, task)
            if task == "classification":
                confusion_figures(dataset, rows)
        rows = in_study(summary, dataset, "loss")
        if not rows.empty:
            sweep_figures(dataset, rows, task, "loss", "head", "funkcji kosztu")
        rows = in_study(summary, dataset, "bias")
        if not rows.empty:
            bias_figure(dataset, rows, task)

    write_index(runs)
    print(
        f"{len(runs)} runs ({int(runs.failed.sum())} failed), {len(gallery)} figures -> "
        f"figures/README.md, figures/figures.tex"
    )
