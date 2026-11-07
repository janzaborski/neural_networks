import argparse
import zlib

import numpy as np

from common import QUERY_DIR, Dataset, datasets, level_tag, save_states

FLIP_LEVELS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
BLOCK_LEVELS = [0.10, 0.20, 0.30, 0.40, 0.50]
KINDS = {"flip": FLIP_LEVELS, "block": BLOCK_LEVELS, "random": [0.0]}
KIND_CODES = {"flip": 1, "block": 2, "random": 3}
SEEDS = list(range(5))
PER_PATTERN = 4
N_RANDOM = 50


def rng_for(ds: Dataset, kind: str, level: float, seed: int) -> np.random.Generator:
    return np.random.default_rng(
        [seed, zlib.crc32(ds.name.encode()), KIND_CODES[kind], round(level * 1000)]
    )


def flip(pattern: np.ndarray, level: float, rng: np.random.Generator) -> np.ndarray:
    out = pattern.copy()
    k = round(level * pattern.size)
    idx = rng.choice(pattern.size, size=k, replace=False)
    out[idx] = -out[idx]
    return out


def block(
    pattern: np.ndarray, ds: Dataset, level: float, rng: np.random.Generator
) -> np.ndarray:
    img = ds.image(pattern).copy()
    bh = max(1, min(ds.height, round(ds.height * np.sqrt(level))))
    bw = max(1, min(ds.width, round(ds.width * np.sqrt(level))))
    r = rng.integers(0, ds.height - bh + 1)
    c = rng.integers(0, ds.width - bw + 1)
    img[r : r + bh, c : c + bw] = rng.choice(
        np.array([-1, 1], dtype=np.int8), size=(bh, bw)
    )
    return img.reshape(-1)


def generate(
    ds: Dataset, kind: str, level: float, seed: int, per_pattern: int = PER_PATTERN
):
    rng = rng_for(ds, kind, level, seed)
    patterns = ds.load()
    if kind == "random":
        queries = rng.choice(np.array([-1, 1], dtype=np.int8), size=(N_RANDOM, ds.n))
        return queries, np.full(N_RANDOM, -1)
    rows, targets = [], []
    for mu, p in enumerate(patterns):
        for _ in range(per_pattern):
            rows.append(
                flip(p, level, rng) if kind == "flip" else block(p, ds, level, rng)
            )
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
        for kind in args.kinds:
            for level in KINDS[kind]:
                for seed in range(args.seeds):
                    ensure(ds, kind, level, seed, args.per_pattern, args.force)
                    count += 1
    print(f"{count} query sets in {QUERY_DIR}")


if __name__ == "__main__":
    main()
