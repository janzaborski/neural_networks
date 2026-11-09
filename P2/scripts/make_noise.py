import argparse
import zlib

import numpy as np

from common import QUERY_DIR, Dataset, datasets, level_tag, save_states

BACKGROUND = -1
KINDS = {
    "flip": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
    "block": [0.10, 0.20, 0.30, 0.40, 0.50],
    "erase": [0.25, 0.50, 0.75],
    "shift": [1.0, 2.0, 3.0],
    "mix": [0.50, 0.75],
    "invert": [0.0],
    "random": [0.0],
}
FLIP_LEVELS = KINDS["flip"]
BLOCK_LEVELS = KINDS["block"]
KIND_CODES = {kind: code for code, kind in enumerate(KINDS, 1)}
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


def block(
    pattern: np.ndarray, ds: Dataset, level: float, rng: np.random.Generator
) -> np.ndarray:
    img = ds.image(pattern).copy()
    bh = max(1, min(ds.height, round(ds.height * np.sqrt(level))))
    bw = max(1, min(ds.width, round(ds.width * np.sqrt(level))))
    r = rng.integers(0, ds.height - bh + 1)
    c = rng.integers(0, ds.width - bw + 1)
    img[r : r + bh, c : c + bw] = random_states(rng, (bh, bw))
    return img.reshape(-1)


def erase(
    pattern: np.ndarray, ds: Dataset, level: float, rng: np.random.Generator
) -> np.ndarray:
    img = ds.image(pattern).copy()
    side = rng.integers(4)
    rows = round(level * ds.height)
    cols = round(level * ds.width)
    if side == 0:
        img[:rows, :] = BACKGROUND
    elif side == 1:
        img[ds.height - rows :, :] = BACKGROUND
    elif side == 2:
        img[:, :cols] = BACKGROUND
    else:
        img[:, ds.width - cols :] = BACKGROUND
    return img.reshape(-1)


def shift(
    pattern: np.ndarray, ds: Dataset, level: float, rng: np.random.Generator
) -> np.ndarray:
    img = ds.image(pattern)
    k = int(level)
    dr, dc = [(-k, 0), (k, 0), (0, -k), (0, k)][rng.integers(4)]
    out = np.full_like(img, BACKGROUND)
    src_r = slice(max(0, -dr), ds.height - max(0, dr))
    dst_r = slice(max(0, dr), ds.height - max(0, -dr))
    src_c = slice(max(0, -dc), ds.width - max(0, dc))
    dst_c = slice(max(0, dc), ds.width - max(0, -dc))
    out[dst_r, dst_c] = img[src_r, src_c]
    return out.reshape(-1)


def mix(
    pattern: np.ndarray, other: np.ndarray, ds: Dataset, level: float
) -> np.ndarray:
    img = ds.image(other).copy()
    cols = round(level * ds.width)
    img[:, :cols] = ds.image(pattern)[:, :cols]
    return img.reshape(-1)


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
    if kind == "block":
        return block(p, ds, level, rng)
    if kind == "erase":
        return erase(p, ds, level, rng)
    if kind == "shift":
        return shift(p, ds, level, rng)
    if kind == "mix":
        others = [nu for nu in range(len(patterns)) if nu != mu]
        return mix(p, patterns[rng.choice(others)], ds, level)
    if kind == "invert":
        return -p
    raise ValueError(f"unknown kind {kind}")


def generate(
    ds: Dataset, kind: str, level: float, seed: int, per_pattern: int = PER_PATTERN
):
    rng = rng_for(ds, kind, level, seed)
    patterns = ds.load()
    if kind == "random":
        return random_states(rng, (N_RANDOM, ds.n)), np.full(N_RANDOM, -1)
    copies = 1 if kind == "invert" else per_pattern
    rows, targets = [], []
    for mu in range(len(patterns)):
        for _ in range(copies):
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
        for kind in args.kinds:
            for level in KINDS[kind]:
                for seed in range(args.seeds):
                    ensure(ds, kind, level, seed, args.per_pattern, args.force)
                    count += 1
    print(f"{count} query sets in {QUERY_DIR}")


if __name__ == "__main__":
    main()
