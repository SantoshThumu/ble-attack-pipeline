"""
Quick checks that the pipeline keeps its promises.

Run from the repo root:
    python -m pytest
"""

import json
import sqlite3
from pathlib import Path

import pandas as pd

from src.detect import run_detection
from src.evaluate import find_attack_windows, read_windows
from src.features import run_features, write_features
from src.ingest import run_ingest
from src.simulate import ATTACK_KINDS, run_simulation

# A small run that is fast but still long enough to fit all three attacks
SMALL_DEVICES = 4
SMALL_MINUTES = 3
SMALL_SEED = 7

WINDOW_S = 5


def simulate_small(folder: Path) -> tuple[pd.DataFrame, list[dict]]:
    """Simulate the small run into folder and return its packets and its answer key."""
    run_simulation(SMALL_DEVICES, SMALL_MINUTES, ATTACK_KINDS, SMALL_SEED, folder)
    packets = pd.read_csv(folder / "packets.csv")
    ground_truth = json.loads((folder / "ground_truth.json").read_text())
    return packets, ground_truth


def run_pipeline(folder: Path, n_devices: int, minutes: float, attacks: list[str], seed: int) -> Path:
    """Run all four stages into folder and return the path of the database."""
    db_path = folder / "ble.db"
    run_simulation(n_devices, minutes, attacks, seed, folder)
    run_ingest(folder / "packets.csv", db_path)
    run_features(db_path, WINDOW_S)
    run_detection(db_path, folder / "flags.csv")
    return db_path


def test_same_seed_gives_identical_packets(tmp_path):
    """The same seed must write exactly the same file twice."""
    simulate_small(tmp_path / "first")
    simulate_small(tmp_path / "second")
    first = (tmp_path / "first" / "packets.csv").read_bytes()
    second = (tmp_path / "second" / "packets.csv").read_bytes()
    assert first == second


def test_simulated_data_is_well_formed(tmp_path):
    """No missing values, timestamps in order and inside the run, every requested attack present."""
    packets, ground_truth = simulate_small(tmp_path)
    assert not packets.isna().any().any()
    assert packets["timestamp"].is_monotonic_increasing
    assert packets["timestamp"].between(0, SMALL_MINUTES * 60).all()
    assert set(packets["label"]) == set(["none"] + ATTACK_KINDS)


def test_attack_packets_fall_inside_their_window(tmp_path):
    """Every packet labeled as an attack must match the answer key's address and time span."""
    packets, ground_truth = simulate_small(tmp_path)
    for attack in ground_truth:
        labeled = packets[packets["label"] == attack["type"]]
        assert len(labeled) > 0
        assert labeled["timestamp"].between(attack["start"], attack["end"]).all()
        assert (labeled["address"] == attack["address"]).all()


def test_packets_table_hides_the_answers(tmp_path):
    """The packets table must have no label or device_id column."""
    simulate_small(tmp_path)
    run_ingest(tmp_path / "packets.csv", tmp_path / "ble.db")
    conn = sqlite3.connect(tmp_path / "ble.db")
    columns = [row[1] for row in conn.execute("PRAGMA table_info(packets)")]
    conn.close()
    assert columns == ["id", "timestamp", "address", "pdu_type", "rssi", "channel"]


def test_detector_needs_only_the_features_table(tmp_path):
    """The detector must give the same result on a database that holds nothing but features."""
    simulate_small(tmp_path)
    run_ingest(tmp_path / "packets.csv", tmp_path / "ble.db")
    features, n_skipped = run_features(tmp_path / "ble.db", WINDOW_S)

    # Copy only the features into a second database (no packets table and no labels table)
    conn = sqlite3.connect(tmp_path / "features_only.db")
    write_features(conn, features)
    conn.close()

    full, flags = run_detection(tmp_path / "ble.db", tmp_path / "flags.csv")
    alone, flags = run_detection(tmp_path / "features_only.db", tmp_path / "flags.csv")
    assert alone.equals(full)


def test_demo_attacks_are_flagged(tmp_path):
    """On the seed-42 demo data, every window fully inside each attack must be flagged."""
    db_path = run_pipeline(tmp_path, n_devices=8, minutes=10, attacks=ATTACK_KINDS, seed=42)
    conn = sqlite3.connect(db_path)
    windows = read_windows(conn)
    conn.close()
    ground_truth = json.loads((tmp_path / "ground_truth.json").read_text())

    assert [attack["type"] for attack in ground_truth] == ATTACK_KINDS
    for attack in ground_truth:
        full, partly = find_attack_windows(windows, attack)
        assert full.sum() > 0
        assert (windows.loc[full, "flagged"] == 1).all(), f"missed a {attack['type']} window"


def test_small_pipeline_runs_end_to_end(tmp_path):
    """A tiny run (2 devices, 1 minute, one attack) must get through every stage."""
    db_path = run_pipeline(tmp_path, n_devices=2, minutes=1, attacks=["flood"], seed=1)
    conn = sqlite3.connect(db_path)
    n_features = conn.execute("SELECT COUNT(*) FROM features").fetchone()[0]
    n_detections = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
    conn.close()
    assert n_features > 0
    assert n_detections == n_features
    assert (tmp_path / "flags.csv").exists()
