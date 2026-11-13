"""Full experiment grid: every dataset x rule x mode x input set x seed.

  cmake --preset release && cmake --build --preset release
  python scripts/experiments.py --dry-run          # list what would run
  python scripts/experiments.py                    # run everything missing (parallel)
  python scripts/experiments.py --datasets small-7x7 --modes async --kinds flip --force

Input sets ("kinds"):
  patterns  the training patterns themselves (stability, no damage)
  flip, mix, random
            damaged / random inputs from make_noise.py (see its --help for the meaning of levels)
Rules:
  hebb           W = (X X^T - N I) / n
  pseudoinverse  W = projection onto the span of the patterns, X (X^T X)^-1 X^T, zero diagonal
Seeds 0..4 everywhere randomness is involved: the damage and the async update order.
Synchronous retrieval of undamaged patterns is deterministic, so it runs with seed 0 only.
Capacity sets (random patterns, capacity-*) run in async mode only.

Every run writes out/<dataset>/<rule>/<mode>/<kind>_<level>/seed<seed>/:
  config.json   the parameters below (written by this script)
  summary.json, results.csv, stability.csv   (written by hopfield_app)
  trajectory.csv  for seed 0 of selected sets (see TRAJECTORY_SETS)
  weights.csv     for the undamaged sync run (one per dataset and rule, only if n <= MAX_WEIGHTS_N)
All configs are also collected in configs/grid.json. Finished runs (summary.json present) are skipped.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from common import DEFAULT_BINARY, OUT_DIR, ROOT, datasets, is_capacity, level_tag
import make_noise

RULES = ["hebb", "pseudoinverse"]
MODES = ["sync", "async"]
KINDS = ["patterns", *make_noise.KINDS]
DETERMINISTIC = {"patterns"}
SEEDS = make_noise.SEEDS
MAX_STEPS = 100
TRAJECTORY_SETS = {
    ("patterns", 0.0),
    ("flip", 0.10),
    ("flip", 0.30),
    ("mix", 0.50),
    ("random", 0.0),
}
MAX_WEIGHTS_N = 2000


def levels(kind: str) -> list[float]:
    return [0.0] if kind == "patterns" else make_noise.KINDS[kind]


def build_grid(args) -> list[dict]:
    grid = []
    for ds in datasets(args.datasets):
        n_patterns = len(ds.load())
        for rule in args.rules:
            for mode in args.modes:
                if is_capacity(ds.name) and mode != "async":
                    continue
                for kind in args.kinds:
                    if kind != "patterns" and not make_noise.applicable(
                        kind, n_patterns, ds.name
                    ):
                        continue
                    for level in levels(kind):
                        deterministic = kind in DETERMINISTIC and mode == "sync"
                        seeds = [0] if deterministic else SEEDS[: args.seeds]
                        for seed in seeds:
                            set_name = f"{kind}_{level_tag(level)}"
                            out = (
                                OUT_DIR
                                / ds.name
                                / rule
                                / mode
                                / set_name
                                / f"seed{seed}"
                            )
                            cfg = {
                                "id": f"{ds.name}/{rule}/{mode}/{set_name}/seed{seed}",
                                "dataset": ds.name,
                                "patterns": str(ds.path.relative_to(ROOT)),
                                "width": ds.width,
                                "height": ds.height,
                                "rule": rule,
                                "mode": mode,
                                "kind": kind,
                                "level": level,
                                "seed": seed,
                                "max_steps": args.max_steps,
                                "trajectory": seed == 0
                                and (kind, level) in TRAJECTORY_SETS,
                                "weights": (
                                    kind == "patterns"
                                    and mode == "sync"
                                    and seed == 0
                                    and ds.n <= MAX_WEIGHTS_N
                                ),
                                "out": str(out.relative_to(ROOT)),
                            }
                            if kind != "patterns":
                                q, t = make_noise.paths(ds, kind, level, seed)
                                cfg["queries"] = str(q.relative_to(ROOT))
                                cfg["targets"] = str(t.relative_to(ROOT))
                            grid.append(cfg)
    return grid


def command(binary: list[str], cfg: dict) -> list[str]:
    cmd = binary + [
        "--patterns",
        cfg["patterns"],
        "--width",
        str(cfg["width"]),
        "--height",
        str(cfg["height"]),
        "--rule",
        cfg["rule"],
        "--mode",
        cfg["mode"],
        "--max-steps",
        str(cfg["max_steps"]),
        "--seed",
        str(cfg["seed"]),
        "--out",
        cfg["out"],
    ]
    if "queries" in cfg:
        cmd += ["--queries", cfg["queries"], "--targets", cfg["targets"]]
    if cfg["trajectory"]:
        cmd.append("--trajectory")
    if cfg["weights"]:
        cmd.append("--weights")
    return cmd


def run_one(binary: list[str], cfg: dict) -> tuple[dict, str | None]:
    out = ROOT / cfg["out"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    (out / "summary.json").unlink(missing_ok=True)
    proc = subprocess.run(
        command(binary, cfg), cwd=ROOT, capture_output=True, text=True
    )
    if proc.returncode != 0:
        return cfg, (proc.stderr or proc.stdout).strip()
    return cfg, None


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--datasets", nargs="*")
    p.add_argument("--rules", nargs="*", choices=RULES, default=RULES)
    p.add_argument("--modes", nargs="*", choices=MODES, default=MODES)
    p.add_argument("--kinds", nargs="*", choices=KINDS, default=KINDS)
    p.add_argument("--seeds", type=int, default=len(SEEDS))
    p.add_argument("--max-steps", type=int, default=MAX_STEPS)
    p.add_argument(
        "--binary", default=str(DEFAULT_BINARY), help="path or command of hopfield_app"
    )
    p.add_argument("--jobs", type=int, default=os.cpu_count())
    p.add_argument("--force", action="store_true", help="rerun finished runs")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    grid = build_grid(args)
    (ROOT / "configs").mkdir(exist_ok=True)
    (ROOT / "configs" / "grid.json").write_text(json.dumps(grid, indent=1))

    todo = [
        c for c in grid if args.force or not (ROOT / c["out"] / "summary.json").exists()
    ]
    print(
        f"{len(grid)} runs in the grid, {len(grid) - len(todo)} done, {len(todo)} to run"
    )
    if args.dry_run:
        for c in todo[:20]:
            print("  " + shlex.join(command(["hopfield_app"], c)))
        if len(todo) > 20:
            print(f"  ... and {len(todo) - 20} more")
        return

    binary = shlex.split(args.binary)
    if len(binary) == 1 and not os.path.exists(binary[0]):
        sys.exit(
            f"{binary[0]} not found - build it first: cmake --preset release && cmake --build --preset release"
        )

    for c in todo:
        if "queries" in c:
            ds = next(d for d in datasets([c["dataset"]]))
            make_noise.ensure(ds, c["kind"], c["level"], c["seed"])

    start = time.time()
    failures = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_one, binary, c) for c in todo]
        for k, f in enumerate(as_completed(futures), 1):
            cfg, error = f.result()
            if error:
                failures.append((cfg["id"], error))
            if k % 50 == 0 or k == len(todo):
                print(
                    f"  {k}/{len(todo)} ({time.time() - start:.0f}s, {len(failures)} failed)"
                )

    if failures:
        log = OUT_DIR / "failures.txt"
        log.write_text("\n".join(f"{i}: {e}" for i, e in failures) + "\n")
        print(f"{len(failures)} runs failed, see {log}")


if __name__ == "__main__":
    main()
