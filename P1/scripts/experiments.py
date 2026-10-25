"""Generate the study configs, build the Rust binary once and run what is not done yet.

  python scripts/experiments.py                       # everything that has no results yet
  python scripts/experiments.py --studies arch,loss   # only some studies
  python scripts/experiments.py --datasets cube,mnist # only some datasets
  python scripts/experiments.py --dry-run             # just list what would run
  python scripts/experiments.py --force               # rerun even finished configs

Methodology: no hyperparameter selection. Every model is trained on the whole training
file and evaluated on the test set every epoch (test never influences training).
Each study changes one thing relative to the baseline; activation and loss are
additionally shown for every learning rate, so the plots show the whole picture.

Regression is evaluated on the full test file ("test") and additionally on its two
parts: points inside the x-range of the training data ("test_in", how well the
function was learned) and outside it ("test_out", extrapolation). The parts are
written to data/derived/ and passed to the binary as extra evaluation sets, so every
run reports all three in history.csv and metrics.json.

A config name is built from its parameters only, so a configuration shared by
several studies (e.g. the baseline) is trained once. configs/manifest.json maps
each name to its dataset, parameters and studies; plot_results.py reads only the
manifest and the output folders.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source"
BINARY = SOURCE / "target" / "release" / "neural_net"
OUT_DIR = "out/study2"
DERIVED = Path("data/derived")
EVAL_SETS = {}  # dataset -> {"test_in": path, "test_out": path}, filled in main()

SEEDS = [0, 1, 2, 3, 4]
MNIST_SEEDS = [0, 1, 2, 3, 4]

# Fixed per dataset: number of epochs, baseline learning rate, and the window
# (last K epochs) over which plot_results.py averages the final metrics.
DATASETS = {
    "simple": {"task": "classification", "epochs": 260, "lr": 0.1, "avg_last": 50},
    "three_gauss": {"task": "classification", "epochs": 340, "lr": 0.1, "avg_last": 50},
    "cube": {"task": "regression", "epochs": 1000, "lr": 0.1, "avg_last": 50},
    "activation": {"task": "regression", "epochs": 1000, "lr": 0.1, "avg_last": 50},
}
HEADS = {
    "classification": ["softmax_cross_entropy", "softmax_mse"],
    "regression": ["identity_mse", "identity_mae"],
}
BASELINE = {"L": 2, "B": 32, "activation": "tanh", "bias": True, "batch": 32}

DEPTHS = [0, 1, 2, 3, 4]
WIDTHS = [4, 8, 16, 32, 64]
LRS = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
ACTIVATIONS = ["sigmoid", "tanh", "relu"]
BIAS_ARCHS = [(1, 8), (2, 32)]

MNIST = {
    "train": {"x": "data/train-images.idx3-ubyte", "y": "data/train-labels.idx1-ubyte"},
    "test": {"x": "data/t10k-images.idx3-ubyte", "y": "data/t10k-labels.idx1-ubyte"},
    "epochs": 20,
    "batch": 64,
    "lr": 0.1,
    "avg_last": 5,
}
MNIST_DEPTHS = [0, 1, 2, 3]
MNIST_WIDTHS = [32, 128, 512]
MNIST_LRS = [0.01, 0.03, 0.1, 0.3]
MNIST_BIAS_ARCHS = [(1, 128), (2, 128)]


def tag(value):
    return f"{value:g}" if isinstance(value, float) else str(value)


def data_files(dataset):
    task = DATASETS[dataset]["task"]
    train = f"data/projekt1/{task}/data.{dataset}.train.1000.csv"
    test = f"data/projekt1/{task}/data.{dataset}.test.1000.csv"
    return train, test


def split_test_by_train_range(dataset):
    """Write test_in / test_out: test points with x inside / outside the training x-range.

    Returns {name: path} of the non-empty parts, plus counts and the range for printing.
    """
    train, test = (ROOT / p for p in data_files(dataset))
    with open(train) as f:
        xs = [float(row[0]) for row in csv.reader(f) if row and _is_number(row[0])]
    lo, hi = min(xs), max(xs)
    with open(test) as f:
        rows = [r for r in csv.reader(f) if r]
    header = [] if _is_number(rows[0][0]) else [rows.pop(0)]
    parts = {
        "test_in": [r for r in rows if lo <= float(r[0]) <= hi],
        "test_out": [r for r in rows if not lo <= float(r[0]) <= hi],
    }
    paths = {}
    for name, part in parts.items():
        path = DERIVED / f"{dataset}.{name}.csv"
        (ROOT / path).parent.mkdir(parents=True, exist_ok=True)
        with open(ROOT / path, "w", newline="") as f:
            csv.writer(f).writerows(header + part)
        if part:
            paths[name] = str(path)
    counts = {name: len(part) for name, part in parts.items()}
    return paths, counts, lo, hi


def _is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def make_config(dataset, p):
    """p: L, B, activation, head, lr, batch, bias, epochs, [weights_every, log_deltas]"""
    hidden = [p["B"]] * p["L"]
    arch = "x".join(map(str, hidden)) or "linear"
    name = (
        f"{dataset}__{arch}_{p['activation']}_{p['head']}_lr{tag(p['lr'])}_bs{p['batch']}"
        f"_{'bias' if p['bias'] else 'nobias'}_ep{p['epochs']}"
    )
    if p.get("weights_every"):
        name += "_logged"

    if dataset == "mnist":
        data = {
            "task": "classification",
            "train": MNIST["train"],
            "test": MNIST["test"],
            "normalize": "min_max",
        }
        dtype, seeds, avg_last = "f32", MNIST_SEEDS, MNIST["avg_last"]
    else:
        train, test = data_files(dataset)
        data = {
            "task": DATASETS[dataset]["task"],
            "train": train,
            "test": test,
            "normalize": "per_column",
        }
        if EVAL_SETS.get(dataset):
            data["eval"] = EVAL_SETS[dataset]
        dtype, seeds, avg_last = "f64", SEEDS, DATASETS[dataset]["avg_last"]

    config = {
        "name": name,
        "out_dir": OUT_DIR,
        "dtype": dtype,
        "seeds": seeds,
        "data": data,
        "net": {
            "hidden": hidden,
            "activation": p["activation"],
            "head": p["head"],
            "bias": p["bias"],
        },
        "train": {
            "epochs": p["epochs"],
            "batch": p["batch"],
            "lr": p["lr"],
            "log_every": 1,
        },
        "output": {
            "weights_every": p.get("weights_every", 0),
            "grid_steps": 100,
            "log_deltas": p.get("log_deltas", False),
        },
    }
    params = {
        k: p[k]
        for k in ("L", "B", "activation", "head", "lr", "batch", "bias", "epochs")
    }
    params["hidden"] = hidden
    params["avg_last"] = avg_last
    return config, params


def small_dataset_studies(dataset):
    info = DATASETS[dataset]
    heads = HEADS[info["task"]]
    base = {**BASELINE, "head": heads[0], "lr": info["lr"], "epochs": info["epochs"]}

    def v(**changes):
        return {**base, **changes}

    studies = {
        "arch": [v(L=0, B=0)] + [v(L=L, B=B) for L in DEPTHS[1:] for B in WIDTHS],
        "activation": [v(activation=a, lr=lr) for a in ACTIVATIONS for lr in LRS],
        "loss": [v(head=h, lr=lr) for h in heads for lr in LRS],
        "bias": [v(L=L, B=B, bias=b) for L, B in BIAS_ARCHS for b in (True, False)],
        # Baseline with full logging (weights, deltas) and a narrow 2x8 net whose graph GIF stays readable.
        "showcase": [
            v(weights_every=1, log_deltas=True),
            v(B=8, weights_every=1, log_deltas=True),
        ],
    }
    return studies


def mnist_studies():
    base = {
        "L": 1,
        "B": 128,
        "activation": "relu",
        "head": "softmax_cross_entropy",
        "bias": True,
        "lr": MNIST["lr"],
        "batch": MNIST["batch"],
        "epochs": MNIST["epochs"],
    }

    def v(**changes):
        return {**base, **changes}

    return {
        "arch": [v(L=0, B=0)]
        + [v(L=L, B=B) for L in MNIST_DEPTHS[1:] for B in MNIST_WIDTHS],
        "activation": [v(activation=a, lr=lr) for a in ACTIVATIONS for lr in MNIST_LRS],
        "loss": [v(head=h, lr=lr) for h in HEADS["classification"] for lr in MNIST_LRS],
        "bias": [
            v(L=L, B=B, bias=b) for L, B in MNIST_BIAS_ARCHS for b in (True, False)
        ],
        "showcase": [v(weights_every=1)],
    }


def generate(datasets, studies):
    configs, manifest = {}, {}
    for dataset in datasets:
        all_studies = (
            mnist_studies() if dataset == "mnist" else small_dataset_studies(dataset)
        )
        for study, variants in all_studies.items():
            if study not in studies:
                continue
            for p in variants:
                config, params = make_config(dataset, deepcopy(p))
                name = config["name"]
                if name not in configs:
                    if dataset == "mnist" and study == "showcase":
                        config["seeds"] = [0]
                    configs[name] = config
                    manifest[name] = {
                        "dataset": dataset,
                        "task": config["data"]["task"],
                        "out_dir": config["out_dir"],
                        "params": params,
                        "studies": [],
                    }
                if study not in manifest[name]["studies"]:
                    manifest[name]["studies"].append(study)
    return configs, manifest


def finished(config):
    run_dir = ROOT / config["out_dir"] / config["name"]
    return all(
        (run_dir / f"seed_{s}" / "metrics.json").exists() for s in config["seeds"]
    )


def main():
    all_datasets = [*DATASETS, "mnist"]
    all_studies = ["arch", "activation", "loss", "bias", "showcase"]
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--datasets", default=",".join(all_datasets))
    parser.add_argument("--studies", default=",".join(all_studies))
    parser.add_argument(
        "--force", action="store_true", help="rerun configs that already have results"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--threads", type=int, help="worker threads (default: all cores)"
    )
    args = parser.parse_args()

    datasets = [d for d in args.datasets.split(",") if d]
    studies = [s for s in args.studies.split(",") if s]
    unknown = sorted(set(datasets) - set(all_datasets)) + sorted(
        set(studies) - set(all_studies)
    )
    if unknown:
        parser.error(f"unknown dataset/study: {unknown}")

    os.chdir(ROOT)
    for dataset in datasets:
        if dataset != "mnist" and DATASETS[dataset]["task"] == "regression":
            EVAL_SETS[dataset], counts, lo, hi = split_test_by_train_range(dataset)
            print(
                f"{dataset}: train x-range [{lo:.3g}, {hi:.3g}], test points inside "
                f"{counts['test_in']}, outside {counts['test_out']}"
            )

    configs, manifest = generate(datasets, studies)

    manifest_path = ROOT / "configs" / "manifest.json"
    manifest_path.parent.mkdir(exist_ok=True)
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    old = {k: v for k, v in old.items() if v.get("out_dir") == OUT_DIR}
    for name, entry in manifest.items():
        if name in old:
            entry["studies"] = sorted(set(old[name]["studies"]) | set(entry["studies"]))
    manifest_path.write_text(json.dumps({**old, **manifest}, indent=2) + "\n")

    todo = [c for c in configs.values() if args.force or not finished(c)]
    runs = sum(len(c["seeds"]) for c in todo)
    print(
        f"{len(configs)} config(s) selected, {len(configs) - len(todo)} already done, "
        f"{len(todo)} to run ({runs} training runs)"
    )
    if args.dry_run or not todo:
        return 0

    todo_path = ROOT / "configs" / "todo.json"
    todo_path.write_text(json.dumps(todo, indent=2) + "\n")

    # Build inside source/ so that source/.cargo/config.toml (target-cpu=native) applies.
    build = subprocess.run(["cargo", "build", "--release", "--locked"], cwd=SOURCE)
    if build.returncode != 0:
        return build.returncode

    env = dict(os.environ)
    if args.threads:
        env["RAYON_NUM_THREADS"] = str(args.threads)
    # Unfinished configs may have partial output folders; --overwrite replaces only those.
    result = subprocess.run([str(BINARY), "--overwrite", str(todo_path)], env=env)
    if result.returncode == 1:
        print(
            "Some runs diverged; they are recorded in metrics.json and shown in the plots."
        )
        return 0
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
