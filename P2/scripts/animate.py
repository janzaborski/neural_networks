"""Convergence of the network step by step, from trajectory.csv of a run.

  python scripts/animate.py out/small-7x7/hebb/async/flip_0.30/seed0
  python scripts/animate.py out/large-25x25/hebb/sync/patterns_0.00/seed0 --queries 0 3 --fps 2

For each selected query writes into figures/<dataset>/trajectories/:
  <rule>_<mode>_<set>_seed<s>_q<k>.gif   animation, one frame per step (sync) / sweep (async)
  <rule>_<mode>_<set>_seed<s>_q<k>.png   the same frames side by side, with step, energy and changed pixels
Convention: +1 = black, -1 = white. Needs a run made with --trajectory (see experiments.py).
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from common import FIG_DIR, ROOT, decode_state

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402


def frame(
    state: np.ndarray, height: int, width: int, scale: int, caption: str
) -> Image.Image:
    img = np.where(state.reshape(height, width) > 0, 0, 255).astype(np.uint8)
    body = Image.fromarray(img, mode="L").resize(
        (width * scale, height * scale), Image.NEAREST
    )
    bar = 14
    canvas = Image.new("L", (max(body.width, 120), body.height + bar), 255)
    canvas.paste(body, (0, bar))
    ImageDraw.Draw(canvas).text((2, 1), caption, fill=0)
    return canvas


def strip(states, captions, height, width, path):
    cols = min(len(states), 10)
    rows = math.ceil(len(states) / cols)
    aspect = height / width
    fig, axes = plt.subplots(
        rows, cols, figsize=(1.5 * cols, 1.5 * aspect * rows + 0.5), squeeze=False
    )
    for ax in axes.flat:
        ax.axis("off")
    for ax, state, caption in zip(axes.flat, states, captions):
        ax.imshow(
            state.reshape(height, width),
            cmap="gray_r",
            vmin=-1,
            vmax=1,
            interpolation="nearest",
        )
        ax.set_title(caption, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "runs", nargs="+", type=Path, help="run directories with trajectory.csv"
    )
    p.add_argument(
        "--queries", nargs="*", type=int, help="query indices (default: first 3)"
    )
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument(
        "--scale",
        type=int,
        default=0,
        help="pixels per cell in the GIF (default: auto)",
    )
    args = p.parse_args()

    for run in args.runs:
        run = run if run.is_absolute() else ROOT / run
        cfg = json.loads((run / "config.json").read_text())
        traj_path = run / "trajectory.csv"
        if not traj_path.exists():
            print(f"{run}: no trajectory.csv (run with --trajectory)")
            continue
        traj = pd.read_csv(traj_path, dtype={"state": str})
        h, w = cfg["height"], cfg["width"]
        scale = args.scale or max(2, 240 // max(h, w))
        queries = (
            args.queries
            if args.queries is not None
            else sorted(traj["query"].unique())[:3]
        )
        out = FIG_DIR / cfg["dataset"] / "trajectories"
        out.mkdir(parents=True, exist_ok=True)
        stem = f"{cfg['rule']}_{cfg['mode']}_{cfg['kind']}_{cfg['level']:.2f}_seed{cfg['seed']}"

        for q in queries:
            rows = traj[traj["query"] == q].sort_values("step")
            if rows.empty:
                continue
            states = [decode_state(s) for s in rows["state"]]
            captions = [
                f"t={r.step} E={r.energy:.1f} zm.={r.changed}"
                for r in rows.itertuples()
            ]
            captions[-1] += f" [{rows['status'].iloc[-1]}]"
            frames = [frame(s, h, w, scale, c) for s, c in zip(states, captions)]
            frames += [frames[-1]] * 2
            gif = out / f"{stem}_q{q}.gif"
            frames[0].save(
                gif,
                save_all=True,
                append_images=frames[1:],
                duration=int(1000 / args.fps),
                loop=0,
            )
            strip(states, captions, h, w, out / f"{stem}_q{q}.png")
            print(f"{gif.relative_to(ROOT)} ({len(states)} frames)")


if __name__ == "__main__":
    main()
