"""
Load the simulated packets from the CSV into a SQLite database.

Run from the repo root:
    python -m src.ingest --csv data/packets.csv --db data/ble.db
"""

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

# Labels live in their own table so the detector can never read the answers
SCHEMA = """
DROP TABLE IF EXISTS labels;
DROP TABLE IF EXISTS packets;

CREATE TABLE packets (
    id        INTEGER PRIMARY KEY,
    timestamp REAL    NOT NULL,
    address   TEXT    NOT NULL,
    pdu_type  TEXT    NOT NULL,
    rssi      INTEGER NOT NULL,
    channel   INTEGER NOT NULL
);

CREATE TABLE labels (
    packet_id INTEGER PRIMARY KEY REFERENCES packets (id),
    label     TEXT    NOT NULL
);

CREATE INDEX idx_packets_address_time ON packets (address, timestamp);
"""


def read_packets(csv_path: Path) -> pd.DataFrame:
    """Read the packet CSV and give every packet an id."""
    packets = pd.read_csv(csv_path)

    # Number the packets from 1 in file order (the file is already sorted by time)
    packets["id"] = range(1, len(packets) + 1)
    return packets


def create_tables(conn: sqlite3.Connection) -> None:
    """Drop any old tables and create empty ones, so re-running never duplicates rows."""
    conn.executescript(SCHEMA)


def insert_packets(conn: sqlite3.Connection, packets: pd.DataFrame) -> None:
    """Insert every packet into the packets table, without its label."""
    # Keep the six packet columns in the same order as the table
    rows = packets[["id", "timestamp", "address", "pdu_type", "rssi", "channel"]]

    # Insert all rows with one statement (each ? is filled from one column)
    conn.executemany("INSERT INTO packets VALUES (?, ?, ?, ?, ?, ?)",
                     rows.itertuples(index=False))


def insert_labels(conn: sqlite3.Connection, packets: pd.DataFrame) -> None:
    """Insert every packet's label into the labels table, keyed by packet id."""
    rows = packets[["id", "label"]]
    conn.executemany("INSERT INTO labels VALUES (?, ?)", rows.itertuples(index=False))


def parse_args() -> argparse.Namespace:
    """Read the command-line options."""
    # Define each option: flag name, type, default value, and help
    parser = argparse.ArgumentParser(description="Load BLE packets into SQLite.")
    parser.add_argument("--csv", type=Path, default=Path("data/packets.csv"), help="input CSV")
    parser.add_argument("--db", type=Path, default=Path("data/ble.db"), help="output database")

    # Parse and return them
    return parser.parse_args()


def run_ingest(csv_path: Path, db_path: Path) -> tuple[int, int]:
    """Load the CSV into the database and return how many packets and labels were stored."""
    # Read the packets
    packets = read_packets(csv_path)

    # Open the database (the file is created if it does not already exist)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)

    # Rebuild the tables, load both of them, and save the changes
    create_tables(conn)
    insert_packets(conn, packets)
    insert_labels(conn, packets)
    conn.commit()

    # Count the rows that are actually in the database, then close it
    n_packets = conn.execute("SELECT COUNT(*) FROM packets").fetchone()[0]
    n_labels = conn.execute("SELECT COUNT(*) FROM labels").fetchone()[0]
    conn.close()
    return n_packets, n_labels


def main() -> None:
    """Load the CSV from the command line and print what was stored."""
    # Parse arguments and stop early if the CSV is missing
    args = parse_args()
    if not args.csv.exists():
        raise SystemExit(f"{args.csv} not found (run 'python -m src.simulate' first)")

    # Load the packets and print a summary
    n_packets, n_labels = run_ingest(args.csv, args.db)
    print(f"Loaded {n_packets} packets and {n_labels} labels into {args.db}")


if __name__ == "__main__":
    main()
