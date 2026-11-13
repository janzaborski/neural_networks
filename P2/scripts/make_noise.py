"""Damaged and random inputs ("queries") for retrieval experiments.

  python scripts/make_noise.py                       # all datasets, all kinds, seeds 0..4
  python scripts/make_noise.py --datasets small-7x7 --kinds flip mix --seeds 3

For every dataset, kind, level and seed it writes
  data/queries/<dataset>/<kind>_<level>_seed<seed>.csv          queries, same format as the patterns
  data/queries/<dataset>/<kind>_<level>_seed<seed>.targets.csv  index of the source pattern per line (-1 = none)

Kinds (level meaning in brackets):
  flip    exactly round(level * n) randomly chosen pixels are inverted        [fraction of pixels]
  mix     every pixel comes from the source pattern with probability `level`,
          otherwise from another, randomly chosen pattern; target = source    [probability]
  random  uniformly random states, target -1                                  [-]
Every pattern gets --per-pattern damaged copies. Output is deterministic for a given seed.
Only random changes are used, so the results do not depend on where the pixels lie in the image
(the network does not use pixel neighbourhood anyway). Capacity sets (capacity-*) get flip only.
"""

import argparse
import zlib

import numpy as np

from common import QUERY_DIR, Dataset, datasets, is_capacity, level_tag, save_states

KINDS = {
    "flip": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
    "mix": [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
    "random": [0.0],
}
FLIP_LEVELS = KINDS["flip"]
KIND_CODES = {"flip": 1, "mix": 6, "random": 7}
SEEDS = list(range(5))
PER_PATTERN = 4
N_RANDOM = 50


def rng_for(ds: Dataset, kind: str, level: float, seed: int) -> np.random.Generator:
    return np.random.default_rng(
        [seed, zlib.crc32(ds.name.encode()), KIND_CODES[kind], round(level * 1000)]
    )


def random_states(rng: np.random.Generator, shape) -> np.ndarray:
    return rng.choice(np.array([-1, 1], dtype=np.int8), size=shape)


def flip(pattern: np.ndarray, level: float, rng: np.random.Generator) -> np.ndarray:
    out = pattern.copy()
    idx = rng.choice(pattern.size, size=round(level * pattern.size), replace=False)
    out[idx] = -out[idx]
    return out


def mix(
    pattern: np.ndarray, other: np.ndarray, level: float, rng: np.random.Generator
) -> np.ndarray:
    return np.where(rng.random(pattern.size) < level, pattern, other).astype(
        pattern.dtype
    )


def damage(
    kind: str,
    patterns: np.ndarray,
    mu: int,
    ds: Dataset,
    level: float,
    rng: np.random.Generator,
) -> np.ndarray:
    p = patterns[mu]
    if kind == "flip":
        return flip(p, level, rng)
    if kind == "mix":
        others = [nu for nu in range(len(patterns)) if nu != mu]
        return mix(p, patterns[rng.choice(others)], level, rng)
    raise ValueError(f"unknown kind {kind}")


def applicable(kind: str, n_patterns: int, name: str = "") -> bool:
    """A mixture needs at least two patterns; capacity sets are only probed with flips."""
    if is_capacity(name):
        return kind == "flip"
    return kind != "mix" or n_patterns >= 2


def generate(
    ds: Dataset, kind: str, level: float, seed: int, per_pattern: int = PER_PATTERN
):
    rng = rng_for(ds, kind, level, seed)
    patterns = ds.load()
    if not applicable(kind, len(patterns), ds.name):
        raise ValueError(f"{ds.name}: kind {kind} needs at least two patterns")
    if kind == "random":
        return random_states(rng, (N_RANDOM, ds.n)), np.full(N_RANDOM, -1)
    rows, targets = [], []
    for mu in range(len(patterns)):
        for _ in range(per_pattern):
            rows.append(damage(kind, patterns, mu, ds, level, rng))
            targets.append(mu)
    return np.array(rows, dtype=np.int8), np.array(targets)


def paths(ds: Dataset, kind: str, level: float, seed: int):
    base = QUERY_DIR / ds.name / f"{kind}_{level_tag(level)}_seed{seed}"
    return base.with_name(base.name + ".csv"), base.with_name(
        base.name + ".targets.csv"
    )


def ensure(
    ds: Dataset,
    kind: str,
    level: float,
    seed: int,
    per_pattern: int = PER_PATTERN,
    force: bool = False,
):
    """Create the query files if missing; returns (queries, targets) paths."""
    q_path, t_path = paths(ds, kind, level, seed)
    if force or not (q_path.exists() and t_path.exists()):
        queries, targets = generate(ds, kind, level, seed, per_pattern)
        save_states(q_path, queries)
        np.savetxt(t_path, targets, fmt="%d")
    return q_path, t_path


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--datasets", nargs="*")
    p.add_argument("--kinds", nargs="*", choices=list(KINDS), default=list(KINDS))
    p.add_argument(
        "--seeds", type=int, default=len(SEEDS), help="number of seeds (0..n-1)"
    )
    p.add_argument("--per-pattern", type=int, default=PER_PATTERN)
    p.add_argument("--force", action="store_true", help="regenerate existing files")
    args = p.parse_args()

    count = 0
    for ds in datasets(args.datasets):
        n_patterns = len(ds.load())
        for kind in args.kinds:
            if not applicable(kind, n_patterns, ds.name):
                continue
            for level in KINDS[kind]:
                for seed in range(args.seeds):
                    ensure(ds, kind, level, seed, args.per_pattern, args.force)
                    count += 1
    print(f"{count} query sets in {QUERY_DIR}")


if __name__ == "__main__":
    main()
