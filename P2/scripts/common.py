import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "projekt2"
QUERY_DIR = ROOT / "data" / "queries"
OUT_DIR = ROOT / "out"
FIG_DIR = ROOT / "figures"
DEFAULT_BINARY = ROOT / "build" / "release" / "app" / "hopfield_app"

SHAPE = re.compile(r"(\d+)x(\d+)")


@dataclass(frozen=True)
class Dataset:
    name: str
    path: Path
    width: int
    height: int

    @property
    def n(self) -> int:
        return self.width * self.height

    def load(self) -> np.ndarray:
        return load_states(self.path)

    def image(self, state: np.ndarray) -> np.ndarray:
        return np.asarray(state).reshape(self.height, self.width)


def dataset_from_path(path: Path) -> Dataset:

    name = path.name.removesuffix(".csv")
    match = SHAPE.search(name)
    if match is None:
        raise ValueError(f"{path}: no WxH shape in the file name")
    return Dataset(name, path, int(match.group(1)), int(match.group(2)))


def datasets(names: list[str] | None = None) -> list[Dataset]:
    found = [dataset_from_path(p) for p in sorted(DATA_DIR.glob("*.csv"))]
    if names:
        unknown = set(names) - {d.name for d in found}
        if unknown:
            raise SystemExit(
                f"unknown datasets: {sorted(unknown)}; available: {[d.name for d in found]}"
            )
        found = [d for d in found if d.name in names]
    return found


def load_states(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", dtype=np.int8, ndmin=2)


def save_states(path: Path, states: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, states, delimiter=",", fmt="%d")


def decode_state(encoded: str) -> np.ndarray:

    return np.frombuffer(encoded.encode(), dtype=np.uint8).astype(np.int8) * 2 - 97


def level_tag(level: float) -> str:
    return f"{level:.2f}"
