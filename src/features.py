"""
Build per-address, per-window features from the packets table.

Run from the repo root:
    python -m src.features --db data/ble.db --window 5
"""

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

IAT_TOLERANCE_MS = 10  # a gap within 10 ms of the usual interval is normal (BLE advDelay)
SMOOTH_PACKETS = 3     # number of packets in the rolling median that smooths RSSI
MIN_PACKETS = 3        # windows with fewer packets than this are skipped

# One row per address per time window (gap features are in milliseconds, RSSI in dBm)
SCHEMA = """
DROP TABLE IF EXISTS features;

CREATE TABLE features (
    address         TEXT    NOT NULL,
    window_start    REAL    NOT NULL,
    window_end      REAL    NOT NULL,
    packets         INTEGER NOT NULL,
    packets_per_sec REAL    NOT NULL,
    iat_median      REAL    NOT NULL,
    iat_std         REAL    NOT NULL,
    iat_deviation   REAL    NOT NULL,
    rssi_median     REAL    NOT NULL,
    rssi_spread     REAL    NOT NULL,
    PRIMARY KEY (address, window_start)
);
"""

COLUMNS = ["address", "window_start", "window_end", "packets", "packets_per_sec",
           "iat_median", "iat_std", "iat_deviation", "rssi_median", "rssi_spread"]


def read_packets(conn: sqlite3.Connection) -> pd.DataFrame:
    """Read the packets table (never the labels), sorted by address and then time."""
    query = "SELECT timestamp, address, rssi FROM packets ORDER BY address, timestamp"
    return pd.read_sql_query(query, conn)


def add_packet_columns(packets: pd.DataFrame, window_s: float) -> pd.DataFrame:
    """Add the per-packet values the features are built from."""
    # Assign each packet to a window by rounding its time down to a multiple of window_s
    packets["window_start"] = (packets["timestamp"] // window_s) * window_s

    # Compute the gap since the same address's previous packet (inter-arrival time, in ms)
    packets["iat"] = packets.groupby("address")["timestamp"].diff() * 1000

    # Estimate each address's usual interval as its median gap over the whole capture
    usual_iat = packets.groupby("address")["iat"].transform("median")

    # Measure how far each gap is from that usual interval
    packets["iat_error"] = (packets["iat"] - usual_iat).abs()

    # Smooth RSSI with a rolling median so a single multipath dip does not count as spread
    packets["rssi_smooth"] = packets.groupby("address")["rssi"].transform(
        lambda rssi: rssi.rolling(SMOOTH_PACKETS, center=True, min_periods=1).median())
    return packets


def share_off_interval(iat_errors: pd.Series) -> float:
    """Return the fraction of gaps that are more than 10 ms away from the usual interval."""
    # Ignore the empty value on an address's first packet (it has no previous packet)
    iat_errors = iat_errors.dropna()
    return (iat_errors > IAT_TOLERANCE_MS).mean()


def spread(values: pd.Series) -> float:
    """Return the range from the 10th to the 90th percentile of the values."""
    # Wider than the IQR (25th-75th), which missed a spoofer sending under a quarter of the packets
    return values.quantile(0.90) - values.quantile(0.10)


def build_features(packets: pd.DataFrame, window_s: float) -> pd.DataFrame:
    """Summarize every (address, window) group of packets into one row of features."""
    # Group the packets by address and window
    grouped = packets.groupby(["address", "window_start"])

    # Compute one value per group for each feature
    features = grouped.agg(
        packets=("timestamp", "count"),
        iat_median=("iat", "median"),
        iat_std=("iat", "std"),
        iat_deviation=("iat_error", share_off_interval),
        rssi_median=("rssi_smooth", "median"),
        rssi_spread=("rssi_smooth", spread),
    ).reset_index()

    # Add the window end and the packet rate
    features["window_end"] = features["window_start"] + window_s
    features["packets_per_sec"] = features["packets"] / window_s

    # Put the columns in table order and round for readability
    return features[COLUMNS].round(3)


def write_features(conn: sqlite3.Connection, features: pd.DataFrame) -> None:
    """Rebuild the features table and insert one row per window."""
    conn.executescript(SCHEMA)
    conn.executemany("INSERT INTO features VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                     features.itertuples(index=False))
    conn.commit()


def parse_args() -> argparse.Namespace:
    """Read the command-line options."""
    # Define each option: flag name, type, default value, and help
    parser = argparse.ArgumentParser(description="Build per-window features from BLE packets.")
    parser.add_argument("--db", type=Path, default=Path("data/ble.db"), help="database file")
    parser.add_argument("--window", type=float, default=5, help="window length in seconds")

    # Parse and return them
    return parser.parse_args()


def main() -> None:
    """Build the features table and print a summary."""
    # Parse arguments and stop early if the database is missing
    args = parse_args()
    if not args.db.exists():
        raise SystemExit(f"{args.db} not found (run 'python -m src.ingest' first)")

    # Read the packets and add the per-packet columns
    conn = sqlite3.connect(args.db)
    packets = read_packets(conn)
    packets = add_packet_columns(packets, args.window)

    # Build one row of features per address per window
    features = build_features(packets, args.window)

    # Skip windows with too few packets to measure gaps and spread
    is_usable = features["packets"] >= MIN_PACKETS
    n_skipped = len(features) - is_usable.sum()
    features = features[is_usable]

    # Write the table, close the database and print a summary
    write_features(conn, features)
    conn.close()
    print(f"Addresses: {features['address'].nunique()}   Window: {args.window:g} s")
    print(f"Windows written: {len(features)}   skipped (under {MIN_PACKETS} packets): {n_skipped}")
    print(f"Wrote table 'features' to {args.db}")


if __name__ == "__main__":
    main()
