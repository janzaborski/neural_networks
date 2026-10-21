"""Create a list of JSON configs and run the Rust program, as in README.md."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]

# Fixed training settings from the earlier validation runs. No automatic tuning.
DATASETS = [
    ("simple", "classification", 260),
    ("three_gauss", "classification", 340),
    ("cube", "regression", 1000),
    ("activation", "regression", 1000),
]
SEEDS = [0, 1, 2]
ARCHITECTURES = [[], [8], [32], [32, 32], [32, 32, 32], [32, 32, 32, 32]]


def make_configs():
    configs = []
    for dataset, task, epochs in DATASETS:
        if task == "classification":
            head, other_head = "softmax_cross_entropy", "softmax_mse"
        else:
            head, other_head = "identity_mse", "identity_mae"

        base = {
            "seeds": SEEDS,
            "out_dir": "out/study",
            "data": {
                "task": task,
                "train": f"data/projekt1/{task}/data.{dataset}.train.1000.csv",
                "test": f"data/projekt1/{task}/data.{dataset}.test.1000.csv",
                "normalize": "per_column", "validation": 0.2, "split_seed": 42,
            },
            "net": {"hidden": [32, 32], "activation": "tanh", "head": head, "bias": True},
            "train": {"epochs": epochs, "batch": 32, "lr": 0.1, "log_every": 10},
            "output": {"weights_every": 0, "grid_steps": 100},
        }

        # Change depth and width, keeping tanh and the default loss.
        variants = []
        for hidden in ARCHITECTURES:
            variants.append((hidden, "tanh", head))
        # Change activation or loss, keeping the baseline architecture [32, 32].
        variants.extend([
            ([32, 32], "sigmoid", head),
            ([32, 32], "relu", head),
            ([32, 32], "tanh", other_head),
        ])

        for hidden, activation, loss in variants:
            config = deepcopy(base)
            architecture = "-".join(map(str, hidden)) if hidden else "linear"
            config["name"] = f"{dataset}_{architecture}_{activation}_{loss}"
            config["net"] = {"hidden": hidden, "activation": activation, "head": loss, "bias": True}
            if hidden == [32, 32] and activation == "tanh" and loss == head:
                # Rust records weights during logging, so both intervals must be 1.
                config["train"]["log_every"] = 1
                config["output"]["weights_every"] = 1
                config["output"]["log_deltas"] = True
            configs.append(config)

    # One MNIST model, with parameters chosen in the earlier validation runs.
    configs.append({
        "name": "mnist", "out_dir": "out/study", "seeds": [0, 1], "dtype": "f32",
        "data": {
            "task": "classification",
            "train": {"x": "data/train-images.idx3-ubyte", "y": "data/train-labels.idx1-ubyte"},
            "test": {"x": "data/t10k-images.idx3-ubyte", "y": "data/t10k-labels.idx1-ubyte"},
            "normalize": "min_max", "validation": 0, "split_seed": 42,
        },
        "net": {"hidden": [128, 64], "activation": "relu", "head": "softmax_cross_entropy", "bias": True},
        "train": {"epochs": 26, "batch": 128, "lr": 0.1, "log_every": 1},
    })
    return configs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="Replace the selected experiment outputs")
    args = parser.parse_args()
    os.chdir(ROOT)
    configs = make_configs()
    Path("configs").mkdir(exist_ok=True)
    Path("configs/experiments.json").write_text(json.dumps(configs, indent=2) + "\n")
    print(f"{len(configs)} configurations, {sum(len(c['seeds']) for c in configs)} training runs", flush=True)
    command = ["cargo", "run", "--release", "--locked", "--manifest-path", "source/Cargo.toml", "--"]
    if args.overwrite:
        command.append("--overwrite")
    result = subprocess.run(command + ["configs/experiments.json"],
                            env={**os.environ, "RAYON_NUM_THREADS": "4"})
    raise SystemExit(result.returncode)
