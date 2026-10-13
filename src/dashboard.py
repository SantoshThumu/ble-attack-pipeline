"""
Show the detection results in a browser dashboard.

Run from the repo root:
    streamlit run src/dashboard.py
"""

import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Add the repo root to the import path so "from src..." works under "streamlit run"
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.demo_data import DATA_DIR, DB_PATH, DEMO_SEED, rebuild_data
from src.detect import MIN_PACKETS_TO_FLAG, SCORE_THRESHOLD
from src.features import MIN_PACKETS

ANSWER_KEY_PATH = DATA_DIR / "ground_truth.json"

# Plain-English names for the attacks that can be injected
ATTACKS = {
    "flood": "Flood: far too many packets",
    "spoof": "Spoof: a copied address",
    "timing": "Timing: an irregular rhythm",
}

# Plain-English chart choices and the column behind each one
MEASURES = {
    "Packets sent per second": "packets_per_sec",
    "Signal strength variation (dB)": "rssi_spread",
    "Off-rhythm gaps (%)": "off_rhythm_pct",
}

GRAY = "#898781"    # normal windows stay in the background
ORANGE = "#eb6834"  # suspicious windows stand out
AQUA = "#1baf7a"    # answer key only: flagged windows that were not attacks
BLUE = "#2a78d6"    # answer key only: attack windows that were not flagged

PARTLY = " (partly inside)"  # added to the attack type of a window that only overlaps an attack


@st.cache_data
def load_windows(db_path: str, modified: float) -> pd.DataFrame:
    """Read every scored window with its features (never the labels table)."""
    # modified is not used below (it makes the cache reload when the database file changes)
    query = """
        SELECT d.address, d.window_start, d.window_end, d.score, d.reason,
               f.packets, f.packets_per_sec, f.rssi_spread, f.iat_deviation
        FROM detections AS d
        JOIN features AS f ON f.address = d.address AND f.window_start = d.window_start
        ORDER BY d.window_start, d.address
    """
    conn = sqlite3.connect(db_path)
    windows = pd.read_sql_query(query, conn)
    conn.close()

    # Turn the fraction of off-rhythm gaps into a percentage for the chart
    windows["off_rhythm_pct"] = windows["iat_deviation"] * 100
    return windows


@st.cache_data
def load_answer_key(json_path: str, modified: float) -> list[dict]:
    """Read the simulator's list of real attacks (evaluation only)."""
    with open(json_path) as file:
        return json.load(file)


def find_real_attacks(windows: pd.DataFrame, answer_key: list[dict]) -> pd.Series:
    """Return the real attack type of each window ("" = no attack). Evaluation only."""
    attack = pd.Series("", index=windows.index)
    for entry in answer_key:
        # A window belongs to an attack if it has the attacked address and overlaps its time span
        same_address = windows["address"] == entry["address"]
        overlaps = (windows["window_end"] > entry["start"]) & (windows["window_start"] < entry["end"])
        attack[same_address & overlaps] = entry["type"] + PARTLY

        # Drop the "partly" note where the whole window sits inside the attack
        inside = (windows["window_start"] >= entry["start"]) & (windows["window_end"] <= entry["end"])
        attack[same_address & inside] = entry["type"]
    return attack


def clock(seconds: float) -> str:
    """Format a time in seconds as minutes:seconds (575 becomes '9:35')."""
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes}:{seconds:02d}"


def show_demo_controls() -> None:
    """Show the sidebar buttons that rebuild the data: inject one attack, or reset to the demo data."""
    st.sidebar.header("Demo controls")
    kind = st.sidebar.selectbox("Attack to inject", list(ATTACKS), format_func=ATTACKS.get)
    inject = st.sidebar.button("Inject attack", type="primary")
    reset = st.sidebar.button("Reset to demo data")

    if inject:
        # Use the next seed each time, so every injection is new but repeatable
        seed = st.session_state.get("seed", DEMO_SEED) + 1
        st.session_state["seed"] = seed
        try:
            with st.spinner("Simulating new traffic and re-running the detector..."):
                injected, seconds = rebuild_data(seed, [kind])
            attack = injected[0]
            st.session_state["notice"] = (
                f"Injected a {kind} attack on device address {attack['address']}, from "
                f"{clock(attack['start'])} to {clock(attack['end'])}. New traffic (seed {seed}) was "
                f"simulated and re-checked in {seconds:.1f} seconds. Look for it near the top of "
                f"the suspicious windows table.")
        except ValueError as error:
            # The simulator could not place this attack, and nothing was changed
            st.session_state["notice"] = f"Could not inject with seed {seed}: {error}. Click again."

    if reset:
        with st.spinner("Restoring the demo data..."):
            injected, seconds = rebuild_data(DEMO_SEED, list(ATTACKS))
        st.session_state["seed"] = DEMO_SEED
        st.session_state["notice"] = (f"Reset to the original demo data (seed {DEMO_SEED}) "
                                      f"in {seconds:.1f} seconds.")


def show_summary(windows: pd.DataFrame) -> None:
    """Show the four headline numbers."""
    first, second, third, fourth = st.columns(4)
    first.metric("Device addresses seen", windows["address"].nunique(),
                 help="Some devices change address for privacy, so one device can appear more than once.")
    second.metric("Packets analysed", f"{windows['packets'].sum():,}")
    third.metric("Time windows checked", f"{len(windows):,}",
                 help="Each window is one device address during one short slice of time.")
    fourth.metric("Suspicious windows", int(windows["flagged"].sum()))


def show_chart(windows: pd.DataFrame, show_key: bool) -> None:
    """Plot one dot per window over time, with suspicious windows in a different colour."""
    st.subheader("Activity over time")
    measure = st.radio("Show on the chart", list(MEASURES), horizontal=True)
    values = windows[MEASURES[measure]]
    flagged = windows["flagged"]

    # Put each group of windows in its own column, so each group gets its own colour
    chart = pd.DataFrame({"Time (minutes)": windows["window_start"] / 60})
    if show_key:
        real = windows["attack"] != ""
        chart["Normal"] = values.where(~flagged & ~real)
        chart["Suspicious: real attack"] = values.where(flagged & real)
        chart["Suspicious: false alarm"] = values.where(flagged & ~real)
        chart["Missed attack"] = values.where(~flagged & real)
        colors = [GRAY, ORANGE, AQUA, BLUE]
    else:
        chart["Normal"] = values.where(~flagged)
        chart["Suspicious"] = values.where(flagged)
        colors = [GRAY, ORANGE]

    # Draw every column except the time as its own coloured series
    series = list(chart.columns[1:])
    st.scatter_chart(chart, x="Time (minutes)", y=series, color=colors, size=40, height=400,
                     x_label="Time in the capture (minutes)", y_label=measure)
    window_s = windows["window_end"].iloc[0] - windows["window_start"].iloc[0]
    st.caption(f"Each dot is one device address during one {window_s:g}-second window.")


def show_flags(windows: pd.DataFrame, show_key: bool) -> None:
    """List the suspicious windows, most unusual first, with the reason for each."""
    st.subheader("Suspicious windows")
    flags = windows[windows["flagged"]].sort_values("score", ascending=False)

    # Build the table with plain-English column names
    table = pd.DataFrame({
        "Device address": flags["address"],
        "Time window": [f"{clock(start)}-{clock(end)}"
                        for start, end in zip(flags["window_start"], flags["window_end"])],
        "Unusualness (0-1)": flags["score"],
        "Packets in window": flags["packets"],
        "Why it was flagged": flags["reason"],
    })
    if show_key:
        table["Answer key (evaluation only)"] = flags["attack"].replace("", "none (false alarm)")

    # Show the table, with three decimals and a tooltip on the score column
    score_help = "Higher means this window was easier to single out from normal traffic."
    score_column = st.column_config.NumberColumn(format="%.3f", help=score_help)
    st.dataframe(table, hide_index=True, column_config={"Unusualness (0-1)": score_column})
    st.caption("Windows with only a few packets are less reliable, so the Sensitivity section "
               "below can ignore them.")


def main() -> None:
    """Lay out the page from top to bottom."""
    st.set_page_config(page_title="BLE attack monitor", layout="wide")
    st.title("BLE attack monitor")
    st.caption("Bluetooth Low Energy traffic, checked for unusual behaviour in short time windows.")

    # Run the demo buttons first, so the page below always shows the newest data
    show_demo_controls()
    if "notice" in st.session_state:
        st.info(st.session_state["notice"])

    # Stop with instructions if the pipeline has not been run yet
    if not DB_PATH.exists():
        st.error("No results found. Run the pipeline first: python -m src.simulate, then "
                 "src.ingest, src.features and src.detect.")
        st.stop()
    windows = load_windows(str(DB_PATH), DB_PATH.stat().st_mtime)

    # Reserve the top three sections now, so the controls below can still change them
    summary_area = st.container()
    chart_area = st.container()
    flags_area = st.container()

    # Sensitivity: flag a window when it is unusual enough and has enough packets to judge
    st.subheader("Sensitivity")
    threshold = st.slider("Flag a window when its unusualness is at least", min_value=0.50,
                          max_value=0.80, value=SCORE_THRESHOLD, step=0.01)
    min_packets = st.slider("Ignore windows with fewer than this many packets",
                            min_value=MIN_PACKETS, max_value=10, value=MIN_PACKETS_TO_FLAG)
    windows["flagged"] = (windows["score"] >= threshold) & (windows["packets"] >= min_packets)
    st.caption(f"Flagging {windows['flagged'].sum()} of {len(windows)} windows. Lower the first "
               "slider to catch more attacks, at the cost of more false alarms.")

    # Evaluation only: compare the flags with the simulator's answer key (off by default)
    st.divider()
    st.subheader("Evaluation only")
    show_key = st.toggle("Show answer key", value=False,
                         help="Marks the real attack windows. The detector never sees this.")
    if show_key and not ANSWER_KEY_PATH.exists():
        st.warning("No answer key found (data/ground_truth.json).")
        show_key = False
    if show_key:
        answer_key = load_answer_key(str(ANSWER_KEY_PATH), ANSWER_KEY_PATH.stat().st_mtime)
        windows["attack"] = find_real_attacks(windows, answer_key)
        real = windows["attack"] != ""
        partly = windows["attack"].str.endswith(PARTLY)
        missed = real & ~windows["flagged"]
        caught = (real & windows["flagged"]).sum()
        false_alarms = (~real & windows["flagged"]).sum()
        st.caption(f"The answer key comes from the simulator and is used only to check the results. "
                   f"Real attack windows: {real.sum()}. Flagged: {caught}. Missed: {missed.sum()} "
                   f"({(missed & ~partly).sum()} fully inside an attack, "
                   f"{(missed & partly).sum()} only partly inside). False alarms: {false_alarms}.")

    # Fill the reserved sections with the current settings
    with summary_area:
        show_summary(windows)
    with chart_area:
        show_chart(windows, show_key)
    with flags_area:
        show_flags(windows, show_key)


if __name__ == "__main__":
    main()
