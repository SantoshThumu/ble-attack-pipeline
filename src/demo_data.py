"""
Build the demo dataset by running the whole pipeline with the demo settings.

Run from the repo root (the Docker container does this when it starts):
    python -m src.demo_data
"""

import time
from pathlib import Path

from src.detect import run_detection
from src.features import run_features
from src.ingest import run_ingest
from src.simulate import ATTACK_KINDS, run_simulation

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "ble.db"

# Settings of the demo dataset (the same defaults as the command-line tools)
DEMO_SEED = 42
DEMO_DEVICES = 8
DEMO_MINUTES = 10
DEMO_WINDOW_S = 5


def rebuild_data(seed: int, attacks: list[str]) -> tuple[list[dict], float]:
    """Run the whole pipeline on a fresh simulation and return its attacks and the seconds taken."""
    started = time.perf_counter()

    # Simulate new traffic, load it, build the features and score them (the same four stages as the CLI)
    packets, injected, n_sent = run_simulation(DEMO_DEVICES, DEMO_MINUTES, attacks, seed, DATA_DIR)
    run_ingest(DATA_DIR / "packets.csv", DB_PATH)
    run_features(DB_PATH, DEMO_WINDOW_S)
    run_detection(DB_PATH, DATA_DIR / "flags.csv")
    return injected, time.perf_counter() - started


def main() -> None:
    """Build the seed-42 demo data, unless a database is already there."""
    if DB_PATH.exists():
        print(f"Found {DB_PATH}, keeping it")
        return
    print(f"No database found, building the demo data (seed {DEMO_SEED})")
    injected, seconds = rebuild_data(DEMO_SEED, ATTACK_KINDS)
    print(f"Wrote {DB_PATH} in {seconds:.1f} seconds")


if __name__ == "__main__":
    main()
