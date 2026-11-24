import argparse
import json

import numpy as np

from common import CUSTOM_DIR, DATA_DIR, dataset_from_path, save_states

LETTERS = DATA_DIR / "letters-14x20.csv"
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def overlaps(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    return x @ x.T / x.shape[1]


def hebb_margins(x: np.ndarray) -> np.ndarray:
    xi = x.astype(np.int64)
    n_patterns, n = xi.shape
    nw = xi.T @ xi - n_patterns * np.eye(n, dtype=np.int64)
    return (xi @ nw) * xi / n


def projection_margins(x: np.ndarray) -> np.ndarray:
    xf = x.astype(np.float64)
    p = xf.T @ np.linalg.pinv(xf.T)
    np.fill_diagonal(p, 0.0)
    return (xf @ p) * xf


def letter_names(idx) -> str:
    return "".join(ALPHABET[i] for i in idx)


def similar_groups(m: np.ndarray, n_groups: int, size: int) -> list[list[int]]:
    a = np.abs(m.copy())
    np.fill_diagonal(a, -np.inf)
    free = set(range(len(m)))
    groups = []
    for _ in range(n_groups):
        idx = sorted(free)
        sub = a[np.ix_(idx, idx)]
        i, j = np.unravel_index(np.argmax(sub), sub.shape)
        group = [idx[i], idx[j]]
        while len(group) < size:
            rest = [k for k in free if k not in group]
            group.append(max(rest, key=lambda k: a[k, group].mean()))
        groups.append(group)
        free -= set(group)
    return groups


def diverse_set(m: np.ndarray, size: int) -> list[int]:
    a = np.abs(m.copy())
    np.fill_diagonal(a, np.inf)
    i, j = np.unravel_index(np.argmin(a), a.shape)
    chosen = [int(i), int(j)]
    while len(chosen) < size:
        rest = [k for k in range(len(m)) if k not in chosen]
        chosen.append(min(rest, key=lambda k: a[k, chosen].max()))
    return chosen


def letters(meta: dict):
    patterns = dataset_from_path(LETTERS).load()
    m = overlaps(patterns)
    groups = similar_groups(m, n_groups=2, size=3)
    similar = [k for g in groups for k in g]
    diverse = diverse_set(m, size=len(similar))
    for name, idx in [
        ("letters-similar-14x20", similar),
        ("letters-diverse-14x20", diverse),
    ]:
        save_states(CUSTOM_DIR / f"{name}.csv", patterns[idx])
        off = np.abs(overlaps(patterns[idx])[~np.eye(len(idx), dtype=bool)])
        meta[name] = {
            "letters": letter_names(idx),
            "mean_abs_overlap": round(float(off.mean()), 3),
            "max_abs_overlap": round(float(off.max()), 3),
        }
    meta["letters-similar-14x20"]["groups"] = [letter_names(g) for g in groups]


def valid_set(x: np.ndarray) -> bool:
    m = np.abs(x.astype(np.int64) @ x.T.astype(np.int64))
    np.fill_diagonal(m, 0)
    return bool((m < x.shape[1]).all())


def find_stable(
    n_patterns: int, n: int, rng: np.random.Generator, sweeps: int
) -> np.ndarray | None:
    x = rng.choice(np.array([-1, 1], dtype=np.int8), size=(n_patterns, n))

    def cost(s):
        return int((hebb_margins(s) <= 0).sum()) + (
            0 if valid_set(s) else n * n_patterns
        )

    current = cost(x)
    temperature = 1.0
    for _ in range(sweeps * n_patterns * n):
        if current == 0:
            return x
        p, i = rng.integers(n_patterns), rng.integers(n)
        x[p, i] = -x[p, i]
        new = cost(x)
        if new <= current or rng.random() < np.exp((current - new) / temperature):
            current = new
        else:
            x[p, i] = -x[p, i]
        temperature = max(0.05, temperature * 0.999)
    return x if current == 0 else None


def largest_hebb_set(
    n: int, restarts: int, sweeps: int, rng: np.random.Generator, meta: dict
) -> np.ndarray:
    best = None
    tried = {}
    n_patterns = 2
    while True:
        found = None
        for r in range(restarts):
            found = find_stable(n_patterns, n, rng, sweeps)
            if found is not None:
                break
        tried[n_patterns] = (
            (r + 1) if found is not None else f"not found in {restarts} restarts"
        )
        print(
            f"  N = {n_patterns}: {'found' if found is not None else 'not found'} ({tried[n_patterns]})"
        )
        if found is None:
            break
        best = found
        n_patterns += 1
    meta["stable-hebb-5x5"] = {
        "N": len(best),
        "restarts_per_size": tried,
        "min_margin": round(float(hebb_margins(best).min()), 4),
    }
    return best


def random_independent(n_patterns: int, n: int, rng: np.random.Generator) -> np.ndarray:
    while True:
        x = rng.choice(np.array([-1, 1], dtype=np.int8), size=(n_patterns, n))
        if np.linalg.matrix_rank(x.astype(np.float64)) == n_patterns and valid_set(x):
            return x


def oscillation(meta: dict, rng: np.random.Generator):
    width = height = 4
    n = width * height
    xi = -np.ones(n, dtype=np.int8)
    xi.reshape(height, width)[1:3, 1:3] = 1
    query = xi.copy()
    flipped = rng.choice(n, size=n // 2, replace=False)
    query[flipped] = -query[flipped]
    assert int(xi.astype(int) @ query.astype(int)) == 0
    name = f"oscillation-{width}x{height}"
    save_states(CUSTOM_DIR / f"{name}.csv", xi[None, :])
    save_states(CUSTOM_DIR / "queries" / f"{name}.csv", query[None, :])
    (CUSTOM_DIR / "queries" / f"{name}.targets.csv").write_text("0\n")
    w = (np.outer(xi, xi) - np.eye(n)) / n
    h = w @ query
    meta[name] = {
        "overlap_query_pattern": 0.0,
        "field_equals_minus_x_over_n": bool(np.allclose(h, -query / n)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--restarts",
        type=int,
        default=200,
        help="search restarts per set size (stable-hebb-5x5)",
    )
    p.add_argument(
        "--sweeps", type=int, default=40, help="search length in sweeps over all pixels"
    )
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    CUSTOM_DIR.mkdir(parents=True, exist_ok=True)
    meta = {}

    letters(meta)

    print("searching for the largest 5x5 set stable under the Hebb rule")
    best = largest_hebb_set(25, args.restarts, args.sweeps, rng, meta)
    save_states(CUSTOM_DIR / "stable-hebb-5x5.csv", best)

    for n_patterns in (12, 24):
        x = random_independent(n_patterns, 25, rng)
        name = f"random{n_patterns}-5x5"
        save_states(CUSTOM_DIR / f"{name}.csv", x)
        meta[name] = {
            "N": n_patterns,
            "hebb_stable": int((hebb_margins(x) > 0).all(axis=1).sum()),
            "projection_stable": int((projection_margins(x) > 0).all(axis=1).sum()),
        }

    oscillation(meta, rng)

    (CUSTOM_DIR / "custom.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"-> {CUSTOM_DIR}")


if __name__ == "__main__":
    main()
