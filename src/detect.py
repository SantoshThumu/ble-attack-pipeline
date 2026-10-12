"""
Score every feature window with an IsolationForest and explain the flagged ones.

Run from the repo root:
    python -m src.detect --db data/ble.db
"""

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# One behaviour feature per question: how often it sends, how regular it is, how steady its signal is
# (iat_median, iat_std and rssi_median are left out because their size depends on which device it is)
MODEL_FEATURES = ["packets_per_sec", "iat_deviation", "rssi_spread"]
N_TREES = 200        # number of random trees in the forest
RANDOM_STATE = 42    # fixed so the scores are the same on every run
MAD_TO_STD = 1.4826  # scales the MAD so it is comparable to a standard deviation

# One sentence per feature, filled in with the window's value and the typical value
REASONS = {
    "packets_per_sec": "Sent {value:.1f} packets/s vs. a typical {typical:.1f}/s",
    "iat_deviation": "{value:.0%} of gaps were off the device's usual rhythm vs. a typical {typical:.0%}",
    "rssi_spread": "Signal strength varied by {value:.0f} dB within {window:g} s vs. a typical "
                   "{typical:.0f} dB, possibly a second transmitter",
}

# One row per window: its anomaly score, whether it is flagged by default, and why
SCHEMA = """
DROP TABLE IF EXISTS detections;

CREATE TABLE detections (
    address      TEXT    NOT NULL,
    window_start REAL    NOT NULL,
    window_end   REAL    NOT NULL,
    score        REAL    NOT NULL,
    flagged      INTEGER NOT NULL,
    reason       TEXT    NOT NULL,
    PRIMARY KEY (address, window_start)
);
"""

COLUMNS = ["address", "window_start", "window_end", "score", "flagged", "reason"]


def read_features(conn: sqlite3.Connection) -> pd.DataFrame:
    """Read the features table (the only table the detector is allowed to see)."""
    return pd.read_sql_query("SELECT * FROM features ORDER BY address, window_start", conn)


def score_windows(features: pd.DataFrame) -> np.ndarray:
    """Return one anomaly score per window (higher = more unusual)."""
    # Fit the forest on the behaviour features of every window
    model = IsolationForest(n_estimators=N_TREES, random_state=RANDOM_STATE)
    model.fit(features[MODEL_FEATURES])

    # Flip the sign so that a higher score means more unusual (scikit-learn's is the reverse)
    return -model.score_samples(features[MODEL_FEATURES])


def robust_z_scores(features: pd.DataFrame) -> pd.DataFrame:
    """Return how far each value is from the typical window, in robust standard deviations."""
    values = features[MODEL_FEATURES]

    # Use the median as the typical value of each feature
    typical = values.median()

    # Use the MAD (median distance from the median) as the typical amount of variation
    mad = (values - typical).abs().median()
    return (values - typical) / (MAD_TO_STD * mad)


def explain_windows(features: pd.DataFrame) -> list[str]:
    """Return one plain-English sentence per window about its most unusual feature."""
    typical = features[MODEL_FEATURES].median()

    # Find the feature that is furthest above typical in each window
    top_features = robust_z_scores(features).idxmax(axis=1)

    reasons = []
    for index, row in features.iterrows():
        # Fill that feature's sentence with the window's value and the typical value
        feature = top_features[index]
        sentence = REASONS[feature].format(value=row[feature], typical=typical[feature],
                                           window=row["window_end"] - row["window_start"])
        reasons.append(sentence)
    return reasons


def build_detections(features: pd.DataFrame, contamination: float) -> pd.DataFrame:
    """Score, flag and explain every window."""
    detections = features[["address", "window_start", "window_end"]].copy()

    # Score every window
    detections["score"] = score_windows(features).round(4)

    # Flag the most unusual windows (contamination = the fraction we expect to be anomalous)
    threshold = detections["score"].quantile(1 - contamination)
    detections["flagged"] = (detections["score"] >= threshold).astype(int)

    # Explain every window, so a different threshold never needs a refit
    detections["reason"] = explain_windows(features)
    return detections[COLUMNS]


def write_detections(conn: sqlite3.Connection, detections: pd.DataFrame) -> None:
    """Rebuild the detections table and insert one row per window."""
    conn.executescript(SCHEMA)
    conn.executemany("INSERT INTO detections VALUES (?, ?, ?, ?, ?, ?)",
                     detections.itertuples(index=False))
    conn.commit()


def write_flags(detections: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
    """Save the flagged windows, most unusual first, as a CSV and return them."""
    flags = detections[detections["flagged"] == 1].sort_values("score", ascending=False)
    flags = flags.drop(columns="flagged")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    flags.to_csv(csv_path, index=False)
    return flags


def parse_args() -> argparse.Namespace:
    """Read the command-line options."""
    # Define each option: flag name, type, default value, and help
    parser = argparse.ArgumentParser(description="Flag unusual BLE feature windows.")
    parser.add_argument("--db", type=Path, default=Path("data/ble.db"), help="database file")
    parser.add_argument("--contamination", type=float, default=0.03,
                        help="fraction of windows to flag by default")
    parser.add_argument("--flags", type=Path, default=Path("data/flags.csv"), help="output CSV")

    # Parse and return them
    return parser.parse_args()


def run_detection(db_path: Path, contamination: float,
                  flags_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score every window, save the results, and return the detections and the flagged rows."""
    # Read the features and build the detections
    conn = sqlite3.connect(db_path)
    features = read_features(conn)
    detections = build_detections(features, contamination)

    # Write the table and the CSV, then close the database
    write_detections(conn, detections)
    flags = write_flags(detections, flags_path)
    conn.close()
    return detections, flags


def main() -> None:
    """Run the detector from the command line and print a summary."""
    # Parse arguments and stop early if the database is missing
    args = parse_args()
    if not args.db.exists():
        raise SystemExit(f"{args.db} not found (run 'python -m src.features' first)")

    # Score the windows, then print a summary and the most unusual windows
    detections, flags = run_detection(args.db, args.contamination, args.flags)
    print(f"Windows scored: {len(detections)}   flagged: {len(flags)}")
    print(f"Wrote table 'detections' to {args.db} and {args.flags}")
    for flag in flags.head(5).itertuples():
        print(f"{flag.score:.3f}  {flag.address}  {flag.window_start:g}-{flag.window_end:g} s  {flag.reason}")


if __name__ == "__main__":
    main()
