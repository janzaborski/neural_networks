import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from common import DEFAULT_BINARY, OUT_DIR, ROOT, datasets, level_tag
import make_noise

RULES = ["hebb", "oja"]
MODES = ["sync", "async"]
KINDS = ["patterns", "flip", "block", "random"]
SEEDS = make_noise.SEEDS
MAX_STEPS = 100
TRAJECTORY_SETS = {
    ("patterns", 0.0),
    ("flip", 0.10),
    ("flip", 0.30),
    ("block", 0.30),
    ("random", 0.0),
}


def levels(kind: str) -> list[float]:
    return {
        "patterns": [0.0],
        "flip": make_noise.FLIP_LEVELS,
        "block": make_noise.BLOCK_LEVELS,
        "random": [0.0],
    }[kind]


def build_grid(args) -> list[dict]:
    grid = []
    for ds in datasets(args.datasets):
        for rule in args.rules:
            for mode in args.modes:
                for kind in args.kinds:
                    for level in levels(kind):
                        seeds = (
                            [0]
                            if (kind == "patterns" and mode == "sync")
                            else SEEDS[: args.seeds]
                        )
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
                                "weights": kind == "patterns"
                                and mode == "sync"
                                and seed == 0,
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
    p.add_argument(
        "--rules",
        nargs="*",
        choices=RULES,
        default=["hebb"],
        help="default: hebb only (add oja once it is implemented)",
    )
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
