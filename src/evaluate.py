"""
Measure how well the detector finds the injected attacks, and compare it with simple rules.

This file reads the answers (the labels table and ground_truth.json) to score the detector.
The detector itself never does.

Run from the repo root:
    python -m src.evaluate --db data/ble.db      (score the current dataset)
    python -m src.evaluate --seeds 100-129       (score fresh simulated runs and write docs/results.md)
"""

import argparse
import json
import sqlite3
import tempfile
from pathlib import Path

import pandas as pd

from src.detect import run_detection
from src.features import run_features
from src.ingest import run_ingest
from src.simulate import ATTACK_KINDS, run_simulation

# Settings for the simulated runs (the same defaults as the command-line tools)
N_DEVICES = 8
MINUTES = 10
WINDOW_S = 5
CONTAMINATION = 0.03

# Rules baseline: flag a window if any one of these fixed limits is exceeded
RULE_PACKETS_PER_SEC = 20  # twice the fastest normal device (100 ms interval = 10 packets/s)
RULE_RSSI_SPREAD_DB = 15   # normal noise is a few dB, and a spoofer is at least 20 dB away
RULE_OFF_RHYTHM = 0.5      # the sniffer misses about 15% of packets, so over half is not normal


def detector_flags(windows: pd.DataFrame) -> pd.Series:
    """Return the windows the detector flagged (the saved flagged column)."""
    return windows["flagged"] == 1


def rule_flags(windows: pd.DataFrame) -> pd.Series:
    """Return the windows the rules baseline flags: any one limit exceeded."""
    too_fast = windows["packets_per_sec"] > RULE_PACKETS_PER_SEC
    unsteady_signal = windows["rssi_spread"] > RULE_RSSI_SPREAD_DB
    off_rhythm = windows["iat_deviation"] > RULE_OFF_RHYTHM
    return too_fast | unsteady_signal | off_rhythm


# The methods being compared: a name and the function that decides which windows to flag
METHODS = {
    "IsolationForest, top 3%": detector_flags,
    "Rules baseline": rule_flags,
}


def read_windows(conn: sqlite3.Connection) -> pd.DataFrame:
    """Read every window with the detector's flag and the three features the rules use."""
    query = """
        SELECT d.address, d.window_start, d.window_end, d.flagged,
               f.packets_per_sec, f.rssi_spread, f.iat_deviation
        FROM detections AS d
        JOIN features AS f ON f.address = d.address AND f.window_start = d.window_start
    """
    return pd.read_sql_query(query, conn)


def count_packets(conn: sqlite3.Connection) -> dict[str, int]:
    """Return how many packets carry each label (none, flood, spoof, timing)."""
    rows = conn.execute("SELECT label, COUNT(*) FROM labels GROUP BY label").fetchall()
    return dict(rows)


def find_attack_windows(windows: pd.DataFrame, attack: dict) -> tuple[pd.Series, pd.Series]:
    """Return which windows are fully inside one attack, and which are only partly inside it."""
    # An attack window has the attacked address and overlaps the attack's time span
    same_address = windows["address"] == attack["address"]
    overlaps = (windows["window_end"] > attack["start"]) & (windows["window_start"] < attack["end"])

    # It is fully inside when the whole window sits within that time span
    inside = (windows["window_start"] >= attack["start"]) & (windows["window_end"] <= attack["end"])
    return same_address & inside, same_address & overlaps & ~inside


def score_flags(windows: pd.DataFrame, flagged: pd.Series, ground_truth: list[dict],
                n_packets: int) -> tuple[list[dict], dict]:
    """Compare one set of flags with the answer key: one row per attack, plus totals for the run."""
    attack_rows = []
    is_attack = pd.Series(False, index=windows.index)
    for attack in ground_truth:
        # Count this attack's windows and how many of them were flagged
        full, partly = find_attack_windows(windows, attack)
        attack_rows.append({
            "type": attack["type"],
            "full": full.sum(),
            "full_caught": (full & flagged).sum(),
            "partly": partly.sum(),
            "partly_caught": (partly & flagged).sum(),
            "caught": int((flagged & (full | partly)).any()),
        })
        is_attack = is_attack | full | partly

    # A false alarm is a flagged window that belongs to no attack
    run_row = {
        "flagged": flagged.sum(),
        "false_alarms": (flagged & ~is_attack).sum(),
        "packets": n_packets,
    }
    return attack_rows, run_row


def evaluate_database(db_path: Path, truth_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score every method on one dataset and return the per-attack rows and the per-run rows."""
    # Read the windows, the packet count and the answer key
    conn = sqlite3.connect(db_path)
    windows = read_windows(conn)
    n_packets = sum(count_packets(conn).values())
    conn.close()
    with open(truth_path) as file:
        ground_truth = json.load(file)

    # Score each method's flags against the same answer key
    all_attack_rows, all_run_rows = [], []
    for method, choose_flags in METHODS.items():
        attack_rows, run_row = score_flags(windows, choose_flags(windows), ground_truth, n_packets)
        for row in attack_rows + [run_row]:
            row["method"] = method
        all_attack_rows += attack_rows
        all_run_rows.append(run_row)
    return pd.DataFrame(all_attack_rows), pd.DataFrame(all_run_rows)


def evaluate_seeds(seeds: list[int], one_attack: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate and score one fresh run per seed, without touching the data folder."""
    attack_tables, run_tables = [], []
    with tempfile.TemporaryDirectory() as folder:
        work_dir = Path(folder)
        db_path = work_dir / "ble.db"
        for seed in seeds:
            # Use all three attacks, or take turns so each run has exactly one
            attacks = [ATTACK_KINDS[seed % len(ATTACK_KINDS)]] if one_attack else ATTACK_KINDS

            # Run the whole pipeline for this seed (skip the seed if the attack cannot be placed)
            try:
                run_simulation(N_DEVICES, MINUTES, attacks, seed, work_dir)
            except ValueError as error:
                print(f"(seed {seed} skipped: {error})", end="", flush=True)
                continue
            run_ingest(work_dir / "packets.csv", db_path)
            run_features(db_path, WINDOW_S)
            run_detection(db_path, CONTAMINATION, work_dir / "flags.csv")

            # Score it and keep the rows
            attacks_table, runs_table = evaluate_database(db_path, work_dir / "ground_truth.json")
            attack_tables.append(attacks_table)
            run_tables.append(runs_table)
            print(".", end="", flush=True)
    print()
    return pd.concat(attack_tables, ignore_index=True), pd.concat(run_tables, ignore_index=True)


def share(caught: int, total: int) -> str:
    """Format a count out of a total, such as '45/48 (94%)'."""
    if total == 0:
        return "0/0"
    return f"{caught}/{total} ({caught / total:.0%})"


def overall_table(attacks: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """Return one row of headline numbers per method."""
    rows = []
    for method in METHODS:
        # Add up this method's counts over every attack and every run
        attack_rows = attacks[attacks["method"] == method]
        run_rows = runs[runs["method"] == method]
        caught_windows = attack_rows["full_caught"].sum() + attack_rows["partly_caught"].sum()
        attack_windows = attack_rows["full"].sum() + attack_rows["partly"].sum()
        false_alarms = run_rows["false_alarms"].sum()

        rows.append({
            "Method": method,
            "Whole attacks caught": share(attack_rows["caught"].sum(), len(attack_rows)),
            "Windows fully inside": share(attack_rows["full_caught"].sum(), attack_rows["full"].sum()),
            "Windows partly inside": share(attack_rows["partly_caught"].sum(), attack_rows["partly"].sum()),
            "Recall": share(caught_windows, attack_windows),
            "Precision": share(caught_windows, run_rows["flagged"].sum()),
            "False alarms per run": f"{false_alarms / len(run_rows):.1f}",
            "False alarms per 10,000 packets": f"{false_alarms / run_rows['packets'].sum() * 10000:.1f}",
        })
    return pd.DataFrame(rows)


def by_type_table(attacks: pd.DataFrame) -> pd.DataFrame:
    """Return one row per method and attack type."""
    rows = []
    for method in METHODS:
        for kind in ATTACK_KINDS:
            attack_rows = attacks[(attacks["method"] == method) & (attacks["type"] == kind)]
            caught_windows = attack_rows["full_caught"].sum() + attack_rows["partly_caught"].sum()
            attack_windows = attack_rows["full"].sum() + attack_rows["partly"].sum()
            rows.append({
                "Method": method,
                "Attack": kind,
                "Whole attacks caught": share(attack_rows["caught"].sum(), len(attack_rows)),
                "Windows fully inside": share(attack_rows["full_caught"].sum(), attack_rows["full"].sum()),
                "Windows partly inside": share(attack_rows["partly_caught"].sum(), attack_rows["partly"].sum()),
                "Recall": share(caught_windows, attack_windows),
            })
    return pd.DataFrame(rows)


def markdown_table(table: pd.DataFrame) -> str:
    """Format a table as Markdown, which also reads well in a terminal."""
    header = "| " + " | ".join(table.columns) + " |"
    divider = "|" + " --- |" * len(table.columns)
    rows = ["| " + " | ".join(str(value) for value in row) + " |"
            for row in table.itertuples(index=False)]
    return "\n".join([header, divider] + rows)


def report_section(title: str, attacks: pd.DataFrame, runs: pd.DataFrame) -> str:
    """Return one section of the report: the overall table, then the table by attack type."""
    if attacks.empty:
        return f"## {title}\n\nThis dataset contains no attacks, so there is nothing to score.\n"

    # Every scored run adds one row per method, so this gives the number of runs
    n_runs = len(runs) // len(METHODS)
    return "\n".join([
        f"## {title}",
        "",
        f"Runs scored: {n_runs}",
        "",
        markdown_table(overall_table(attacks, runs)),
        "",
        "By attack type:",
        "",
        markdown_table(by_type_table(attacks)),
        "",
    ])


def report_notes() -> str:
    """Return the fixed explanation that goes under the tables in the results file."""
    return f"""## How to read these tables

- A window is one device address during one {WINDOW_S}-second slice of time.
- An attack window is fully inside when the whole window falls within the attack, and partly
  inside when it only overlaps the start or the end of it.
- A whole attack counts as caught when at least one of its windows is flagged.
- Recall is the fraction of attack windows (fully or partly inside) that were flagged.
- Precision is the fraction of flagged windows that really were attack windows.
- A false alarm is a flagged window that belongs to no attack. False alarms have no attack
  type, so they and precision are only reported overall.

## The rules baseline

The baseline flags a window when any one of three fixed limits is exceeded. The limits come
from how the simulated traffic is specified, not from the runs scored above.

| Rule | Why this limit |
| --- | --- |
| More than {RULE_PACKETS_PER_SEC} packets per second | The fastest normal device sends every 100 ms, which is 10 packets per second. Twice that is not normal. |
| Signal strength varies by more than {RULE_RSSI_SPREAD_DB} dB | Normal noise is a few dB, and a spoofing transmitter is at least 20 dB away from the real one. |
| More than {RULE_OFF_RHYTHM:.0%} of gaps off the usual rhythm | The sniffer misses about 15% of packets, and each miss puts one gap off rhythm. Over half is not explained by that. |
"""


def parse_args() -> argparse.Namespace:
    """Read the command-line options."""
    # Define each option: flag name, type, default value, and help
    parser = argparse.ArgumentParser(description="Score the detector against the answer key.")
    parser.add_argument("--db", type=Path, default=Path("data/ble.db"), help="database file")
    parser.add_argument("--truth", type=Path, default=Path("data/ground_truth.json"),
                        help="answer key written by the simulator")
    parser.add_argument("--seeds", help="simulate and score fresh runs instead, for example 100-129")
    parser.add_argument("--out", type=Path, default=Path("docs/results.md"),
                        help="results file written when --seeds is used")
    args = parser.parse_args()

    # Turn the seed range (first-last, or a single number) into a list of seeds
    if args.seeds:
        parts = args.seeds.split("-")
        args.seeds = list(range(int(parts[0]), int(parts[-1]) + 1))
    return args


def main() -> None:
    """Score the current dataset, or score many fresh runs and write the results file."""
    args = parse_args()

    # Without --seeds, score the dataset that is already in the database
    if not args.seeds:
        if not args.db.exists() or not args.truth.exists():
            raise SystemExit(f"{args.db} or {args.truth} not found (run the pipeline first)")
        attacks, runs = evaluate_database(args.db, args.truth)
        print(report_section(f"Current dataset ({args.db})", attacks, runs))
        return

    # With --seeds, score fresh runs in two scenarios
    seed_range = f"{args.seeds[0]}-{args.seeds[-1]}"
    print("Scoring runs with all three attacks")
    attacks, runs = evaluate_seeds(args.seeds, one_attack=False)
    all_three = report_section("All three attacks in every run", attacks, runs)
    print("Scoring runs with one attack each")
    attacks, runs = evaluate_seeds(args.seeds, one_attack=True)
    single = report_section("One attack per run", attacks, runs)

    # Print the report and save it as a Markdown file
    intro = (f"# Detection results\n\nGenerated by `python -m src.evaluate --seeds {seed_range}`. "
             f"Each simulated run is {MINUTES} minutes long with {N_DEVICES} devices.\n")
    report = "\n".join([intro, all_three, single, report_notes()])
    print(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
