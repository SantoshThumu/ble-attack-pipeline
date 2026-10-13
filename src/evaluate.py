"""
Measure how well the detector finds the injected attacks, and compare it with simple rules.

This file reads the answers (the labels table and ground_truth.json) to score the detector.
The detector itself never does.

Run from the repo root:
    python -m src.evaluate --db data/ble.db      (score the current dataset)
    python -m src.evaluate --seeds 100-129       (score fresh simulated runs and write docs/results.md)
    python -m src.evaluate --tune 200-229        (show how the detector's two settings were chosen)
"""

import argparse
import json
import sqlite3
import tempfile
from pathlib import Path

import pandas as pd

from src.detect import MIN_PACKETS_TO_FLAG, SCORE_THRESHOLD, run_detection
from src.features import run_features
from src.ingest import run_ingest
from src.simulate import ATTACK_KINDS, run_simulation

# Settings for the simulated runs (the same defaults as the command-line tools)
N_DEVICES = 8
MINUTES = 10
WINDOW_S = 5

# The original approach, kept as the "before" in every report: flag the top 3% of each run
QUOTA = 0.03

# Rules baseline: flag a window if any one of these fixed limits is exceeded
RULE_PACKETS_PER_SEC = 20  # twice the fastest normal device (100 ms interval = 10 packets/s)
RULE_RSSI_SPREAD_DB = 15   # normal noise is a few dB, and a spoofer is at least 20 dB away
RULE_OFF_RHYTHM = 0.5      # the sniffer misses about 15% of packets, so over half is not normal

# The settings that --tune tries
TUNE_THRESHOLDS = [0.60, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67]
TUNE_MIN_PACKETS = [3, 4, 5]


def quota_flags(windows: pd.DataFrame) -> pd.Series:
    """Return the windows whose score is in the top 3% of the run (the original approach)."""
    # Worked out from the saved scores, so it does not depend on what detect.py flags now
    threshold = windows["score"].quantile(1 - QUOTA)
    return windows["score"] >= threshold


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
    "Before: top 3% of each run": quota_flags,
    f"After: score of {SCORE_THRESHOLD}+ and {MIN_PACKETS_TO_FLAG}+ packets": detector_flags,
    "Rules baseline": rule_flags,
}


def read_windows(conn: sqlite3.Connection) -> pd.DataFrame:
    """Read every window with its score, the detector's flag and the features the rules use."""
    query = """
        SELECT d.address, d.window_start, d.window_end, d.score, d.flagged,
               f.packets, f.packets_per_sec, f.rssi_spread, f.iat_deviation
        FROM detections AS d
        JOIN features AS f ON f.address = d.address AND f.window_start = d.window_start
    """
    return pd.read_sql_query(query, conn)


def count_packets(conn: sqlite3.Connection) -> dict[str, int]:
    """Return how many packets carry each label (none, flood, spoof, timing)."""
    rows = conn.execute("SELECT label, COUNT(*) FROM labels GROUP BY label").fetchall()
    return dict(rows)


def load_run(db_path: Path, truth_path: Path) -> tuple[pd.DataFrame, list[dict], int]:
    """Read one finished run: its windows, its answer key and its number of packets."""
    conn = sqlite3.connect(db_path)
    windows = read_windows(conn)
    n_packets = sum(count_packets(conn).values())
    conn.close()
    with open(truth_path) as file:
        ground_truth = json.load(file)
    return windows, ground_truth, n_packets


def simulate_runs(seeds: list[int], one_attack: bool) -> list[tuple]:
    """Simulate one fresh run per seed, without touching the data folder, and return them all."""
    runs = []
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
            run_detection(db_path, work_dir / "flags.csv")

            # Keep this run's windows and answers
            runs.append(load_run(db_path, work_dir / "ground_truth.json"))
            print(".", end="", flush=True)
    print()
    return runs


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


def score_runs(runs: list[tuple], choose_flags) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score one way of flagging on every run and return the per-attack rows and per-run rows."""
    all_attack_rows, all_run_rows = [], []
    for windows, ground_truth, n_packets in runs:
        attack_rows, run_row = score_flags(windows, choose_flags(windows), ground_truth, n_packets)
        all_attack_rows += attack_rows
        all_run_rows.append(run_row)
    return pd.DataFrame(all_attack_rows), pd.DataFrame(all_run_rows)


def share(caught: int, total: int) -> str:
    """Format a count out of a total, such as '45/48 (94%)'."""
    if total == 0:
        return "0/0"
    return f"{caught}/{total} ({caught / total:.0%})"


def attack_numbers(attacks: pd.DataFrame) -> dict:
    """Turn per-attack rows into the headline numbers about catching attacks."""
    caught_windows = attacks["full_caught"].sum() + attacks["partly_caught"].sum()
    attack_windows = attacks["full"].sum() + attacks["partly"].sum()
    return {
        "Whole attacks caught": share(attacks["caught"].sum(), len(attacks)),
        "Windows fully inside": share(attacks["full_caught"].sum(), attacks["full"].sum()),
        "Windows partly inside": share(attacks["partly_caught"].sum(), attacks["partly"].sum()),
        "Recall": share(caught_windows, attack_windows),
    }


def false_alarm_numbers(attacks: pd.DataFrame, per_run: pd.DataFrame) -> dict:
    """Turn per-run rows into the headline numbers about false alarms."""
    caught_windows = attacks["full_caught"].sum() + attacks["partly_caught"].sum()
    false_alarms = per_run["false_alarms"].sum()
    return {
        "Precision": share(caught_windows, per_run["flagged"].sum()),
        "False alarms per run": f"{false_alarms / len(per_run):.1f}",
        "False alarms per 10,000 packets": f"{false_alarms / per_run['packets'].sum() * 10000:.1f}",
    }


def overall_table(runs: list[tuple]) -> pd.DataFrame:
    """Return one row of headline numbers per method."""
    rows = []
    for method, choose_flags in METHODS.items():
        attacks, per_run = score_runs(runs, choose_flags)
        rows.append({"Method": method} | attack_numbers(attacks) | false_alarm_numbers(attacks, per_run))
    return pd.DataFrame(rows)


def by_type_table(runs: list[tuple]) -> pd.DataFrame:
    """Return one row per method and attack type."""
    rows = []
    for method, choose_flags in METHODS.items():
        attacks, per_run = score_runs(runs, choose_flags)
        for kind in ATTACK_KINDS:
            of_this_kind = attacks[attacks["type"] == kind]
            rows.append({"Method": method, "Attack": kind} | attack_numbers(of_this_kind))
    return pd.DataFrame(rows)


def tuning_table(runs: list[tuple]) -> pd.DataFrame:
    """Return one row per score threshold and minimum packet count that --tune tries."""
    rows = []
    for min_packets in TUNE_MIN_PACKETS:
        for threshold in TUNE_THRESHOLDS:
            # Flag with these two settings, using the scores the detector saved
            def choose_flags(windows: pd.DataFrame) -> pd.Series:
                return (windows["score"] >= threshold) & (windows["packets"] >= min_packets)

            attacks, per_run = score_runs(runs, choose_flags)
            settings = {"Minimum packets": min_packets, "Score threshold": f"{threshold:.2f}"}
            rows.append(settings | attack_numbers(attacks) | false_alarm_numbers(attacks, per_run))
    return pd.DataFrame(rows)


def markdown_table(table: pd.DataFrame) -> str:
    """Format a table as Markdown, which also reads well in a terminal."""
    header = "| " + " | ".join(table.columns) + " |"
    divider = "|" + " --- |" * len(table.columns)
    rows = ["| " + " | ".join(str(value) for value in row) + " |"
            for row in table.itertuples(index=False)]
    return "\n".join([header, divider] + rows)


def report_section(title: str, runs: list[tuple]) -> str:
    """Return one section of the report: the overall table, then the table by attack type."""
    n_attacks = sum(len(ground_truth) for windows, ground_truth, n_packets in runs)
    if n_attacks == 0:
        return f"## {title}\n\nThis dataset contains no attacks, so there is nothing to score.\n"
    return "\n".join([
        f"## {title}",
        "",
        f"Runs scored: {len(runs)}",
        "",
        markdown_table(overall_table(runs)),
        "",
        "By attack type:",
        "",
        markdown_table(by_type_table(runs)),
        "",
    ])


def report_notes() -> str:
    """Return the fixed explanation that goes under the tables in the results file."""
    return f"""## Before and after

- Before: the detector flagged the top {QUOTA:.0%} most unusual windows of every run, however
  many attacks the run contained. With a single attack, most of that quota was false alarms.
- After: the detector flags a window when its anomaly score is at least {SCORE_THRESHOLD} and it
  has at least {MIN_PACKETS_TO_FLAG} packets. The anomaly scores themselves are unchanged.
- The trade-off is far fewer false alarms for somewhat lower recall.
- Both settings were chosen on a separate set of runs (seeds 200-229, shown by
  `python -m src.evaluate --tune 200-229`), never on the runs scored above. The rule was:
  take the highest score threshold that still caught every attack in those runs, step back
  by 0.02 as a safety margin, then take the minimum packet count with the best balance of
  precision and recall that still caught every attack.

## How to read these tables

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


def parse_seeds(text: str) -> list[int]:
    """Turn a seed range (first-last, or a single number) into a list of seeds."""
    parts = text.split("-")
    return list(range(int(parts[0]), int(parts[-1]) + 1))


def parse_args() -> argparse.Namespace:
    """Read the command-line options."""
    # Define each option: flag name, type, default value, and help
    parser = argparse.ArgumentParser(description="Score the detector against the answer key.")
    parser.add_argument("--db", type=Path, default=Path("data/ble.db"), help="database file")
    parser.add_argument("--truth", type=Path, default=Path("data/ground_truth.json"),
                        help="answer key written by the simulator")
    parser.add_argument("--seeds", type=parse_seeds,
                        help="simulate and score fresh runs instead, for example 100-129")
    parser.add_argument("--tune", type=parse_seeds,
                        help="try score thresholds and minimum packet counts on these seeds")
    parser.add_argument("--out", type=Path, default=Path("docs/results.md"),
                        help="results file written when --seeds is used")

    # Parse and return them
    return parser.parse_args()


def main() -> None:
    """Score the current dataset, or score many fresh runs, or show the tuning table."""
    args = parse_args()

    # With --tune, try every setting on these seeds (both scenarios together) and stop
    if args.tune:
        print("Simulating the tuning runs")
        runs = simulate_runs(args.tune, one_attack=False) + simulate_runs(args.tune, one_attack=True)
        print(markdown_table(tuning_table(runs)))
        return

    # Without --seeds, score the dataset that is already in the database
    if not args.seeds:
        if not args.db.exists() or not args.truth.exists():
            raise SystemExit(f"{args.db} or {args.truth} not found (run the pipeline first)")
        run = load_run(args.db, args.truth)
        print(report_section(f"Current dataset ({args.db})", [run]))
        return

    # With --seeds, score fresh runs in two scenarios
    seed_range = f"{args.seeds[0]}-{args.seeds[-1]}"
    print("Scoring runs with all three attacks")
    all_three = report_section("All three attacks in every run", simulate_runs(args.seeds, one_attack=False))
    print("Scoring runs with one attack each")
    single = report_section("One attack per run", simulate_runs(args.seeds, one_attack=True))

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
