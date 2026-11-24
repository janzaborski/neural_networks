import argparse

import numpy as np

from common import CUSTOM_DIR, save_states

COUNTS = [4, 8, 16, 24, 32, 40, 48, 56, 64, 80, 100, 140, 200, 280, 360]


def main():
    p = argparse.ArgumentParser()
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
