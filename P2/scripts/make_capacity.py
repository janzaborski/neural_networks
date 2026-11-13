"""Sets of uniformly random patterns for comparing stability and recall with theory.

  python scripts/make_capacity.py
  python scripts/make_capacity.py --width 20 --height 20 --counts 4 8 16 40 80

Writes data/custom/capacity-N<NNN>-<W>x<H>.csv: N patterns with independent states +-1 (p = 1/2),
all of the same size n = W * H, for every N in --counts. The default n = 400 and N from 4 to 360
cover the load N/n from 0.01 to 0.9, around the Hebb capacity 0.138 n (about 55 patterns).
Every set is drawn independently (seed --seed + N). make_noise.py generates only flips for them.
"""

import argparse

import numpy as np

from common import CUSTOM_DIR, save_states

COUNTS = [4, 8, 16, 24, 32, 40, 48, 56, 64, 80, 100, 140, 200, 280, 360]


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--width", type=int, default=20)
    p.add_argument("--height", type=int, default=20)
    p.add_argument("--counts", nargs="*", type=int, default=COUNTS)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    n = args.width * args.height
    CUSTOM_DIR.mkdir(parents=True, exist_ok=True)
    for count in args.counts:
        if not 1 <= count < n:
            raise SystemExit(f"N = {count} must be between 1 and n - 1 = {n - 1}")
        rng = np.random.default_rng([args.seed, count])
        patterns = rng.choice(np.array([-1, 1], dtype=np.int8), size=(count, n))
        path = CUSTOM_DIR / f"capacity-N{count:03d}-{args.width}x{args.height}.csv"
        save_states(path, patterns)
        print(f"{path.name}: N/n = {count / n:.3f}")


if __name__ == "__main__":
    main()
