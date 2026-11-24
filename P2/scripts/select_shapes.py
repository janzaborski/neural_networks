import argparse
import json
import re

import numpy as np
import pandas as pd

from common import CUSTOM_DIR, load_states, save_states
from convert_mpeg7 import MPEG7_DIR


def hebb_stable(x: np.ndarray) -> int:
    xf = x.astype(np.float64)
    n_patterns, n = xf.shape
    h = ((xf @ xf.T) @ xf - n_patterns * xf) / n
    return int(((h * xf > 0) | (h == 0)).all(axis=1).sum())


def set_stats(x: np.ndarray) -> dict:
    xf = x.astype(np.float64)
    gram = xf @ xf.T / xf.shape[1]
    off = np.abs(gram[~np.eye(len(x), dtype=bool)])
    eig = np.linalg.eigvalsh(gram)
    return {
        "N": len(x),
        "foreground": round(float((x > 0).mean()), 3),
        "mean_abs_overlap": round(float(off.mean()), 3),
        "max_abs_overlap": round(float(off.max()), 3),
        "hebb_stable": hebb_stable(x),
        "gram_condition": round(float(eig[-1] / eig[0]), 1) if eig[0] > 1e-12 else None,
    }


def greedy(m: np.ndarray, eligible: list[int], size: int, criterion: str) -> list[int]:
    a = np.abs(m)
    sub = a[np.ix_(eligible, eligible)].copy()
    np.fill_diagonal(sub, np.inf)
    i, j = np.unravel_index(np.argmin(sub), sub.shape)
    chosen = [eligible[i], eligible[j]]
    score = (
        (lambda k: a[k, chosen].max())
        if criterion == "max"
        else (lambda k: a[k, chosen].mean())
    )
    while len(chosen) < size:
        rest = [k for k in eligible if k not in chosen]
        chosen.append(min(rest, key=score))
    return chosen


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--shape",
        default=None,
        help="WxH of the catalogue (default: the only all-WxH.csv)",
    )
    p.add_argument("--sizes", nargs="*", type=int, default=[5, 10, 20])
    p.add_argument(
        "--classes", nargs="*", help="explicit classes instead of the automatic choice"
    )
    p.add_argument(
        "--exclude", nargs="*", default=[], help="classes never chosen automatically"
    )
    p.add_argument("--min-foreground", type=float, default=0.25)
    p.add_argument("--max-foreground", type=float, default=0.75)
    p.add_argument("--criterion", choices=["max", "mean"], default="max")
    p.add_argument("--name", default="shapes")
    args = p.parse_args()

    catalogues = sorted(MPEG7_DIR.glob("all-*x*.csv"))
    if args.shape:
        catalogues = [MPEG7_DIR / f"all-{args.shape}.csv"]
    if len(catalogues) != 1 or not catalogues[0].exists():
        raise SystemExit(
            f"run convert_mpeg7.py first or pass --shape (found: {[c.name for c in catalogues]})"
        )
    shape = re.search(r"\d+x\d+", catalogues[0].name).group(0)
    states = load_states(catalogues[0])
    info = pd.read_csv(MPEG7_DIR / "classes.csv")
    names = list(info["class"])
    x = states.astype(np.float64)
    m = x @ x.T / x.shape[1]

    if args.classes:
        unknown = set(args.classes) - set(names)
        if unknown:
            raise SystemExit(f"unknown classes: {sorted(unknown)}")
        sets = {f"{args.name}-{shape}": [names.index(c) for c in args.classes]}
    else:
        fg = info["foreground"].to_numpy()
        eligible = [
            k
            for k, c in enumerate(names)
            if args.min_foreground <= fg[k] <= args.max_foreground
            and c not in args.exclude
        ]
        largest = max(args.sizes)
        if len(eligible) < largest:
            raise SystemExit(
                f"only {len(eligible)} classes have foreground in "
                f"[{args.min_foreground}, {args.max_foreground}], need {largest}"
            )
        order = greedy(m, eligible, largest, args.criterion)
        sets = {
            f"{args.name}{size}-{shape}": order[:size] for size in sorted(args.sizes)
        }

    meta = {}
    for set_name, idx in sets.items():
        save_states(CUSTOM_DIR / f"{set_name}.csv", states[idx])
        meta[set_name] = {"classes": [names[k] for k in idx], **set_stats(states[idx])}
    (CUSTOM_DIR / f"{args.name}.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    print(f"-> {CUSTOM_DIR}")


if __name__ == "__main__":
    main()
