"""
Generate BLE advertising traffic, inject attacks, and save it as a labeled CSV.

Run from the repo root:
    python -m src.simulate --devices 8 --minutes 10 --attacks flood,spoof,timing --seed 42
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ADV_DELAY_MAX_S = 0.010  # BLE adds a random 0-10 ms delay to every advertising event
RSSI_NOISE_DB = 3.0      # standard deviation of the normal RSSI noise
DIP_CHANCE = 0.05        # fraction of packets hit by a brief multipath dip
ROTATE_CHANCE = 0.4      # fraction of devices that use rotating addresses
SCAN_RSP_CHANCE = 0.10   # fraction of an ADV_IND device's packets that are scan responses
CHANNELS = [37, 38, 39]  # the three BLE advertising channels

ATTACK_KINDS = ["flood", "spoof", "timing"]
WINDOW_MIN_S = 20        # shortest attack window
WINDOW_MAX_S = 40        # longest attack window
FLOOD_GAP_MIN_S = 0.005  # a flooding address sends every 5-10 ms
FLOOD_GAP_MAX_S = 0.010
SPOOF_RSSI_MIN_DB = 20   # a spoofer is heard 20-30 dB away from the real device
SPOOF_RSSI_MAX_DB = 30
TIMING_GAP_MIN = 0.2     # timing attack gaps are 0.2x-2x the device's normal interval
TIMING_GAP_MAX = 2.0
INJECTED_ID = -1         # device_id for packets added by an attack


@dataclass
class Device:
    """One simulated BLE advertiser and its fixed characteristics."""

    device_id: int            # never written to the CSV
    interval_s: float         # fixed advertising interval, 0.1 to 1.0 s
    base_rssi: float          # signal strength at the sniffer, -90 to -40 dBm
    pdu_type: str             # "ADV_IND" or "ADV_NONCONN_IND"
    rotation_s: float | None  # seconds between address changes (None = never rotates)


def random_address(rng: np.random.Generator) -> str:
    """Return a random 6-byte address such as 'A0:B2:C3:4D:5E:6F'."""
    # Generate 6 random integers between 0-255
    address_bytes = rng.integers(0, 256, size=6)

    # Format each as two uppercase hex digits and join them with ':'
    return ":".join(f"{byte:02X}" for byte in address_bytes)


def make_devices(n: int, rng: np.random.Generator) -> list[Device]:
    """Create n devices with their own respective characteristics (interval, signal strength and PDU type)."""
    devices = []
    for device_id in range(n):
        # Generate a fixed advertising interval between 100-1000 ms
        interval_s = rng.uniform(0.100, 1.000)

        # Generate a base RSSI between -90 and -40 dBm
        base_rssi = rng.uniform(-90, -40)

        # Choose the advertising type
        pdu_type = str(rng.choice(["ADV_IND", "ADV_NONCONN_IND"], p=[0.75, 0.25]))

        # Check if the device rotates its address
        rotation_s = None
        if rng.random() < ROTATE_CHANCE:
            # Decide rotation interval between 120-240 seconds
            rotation_s = rng.uniform(120, 240)

        # Put everything into a Device and add it to the list
        devices.append(Device(device_id, interval_s, base_rssi, pdu_type, rotation_s))
    return devices


def event_times(device: Device, duration_s: float, rng: np.random.Generator) -> np.ndarray:
    """Return the timestamps(s) when the device advertises."""
    # Work out how many events can fit
    n_events = int(np.ceil(duration_s / device.interval_s)) + 1

    # Build each gap as the fixed interval plus a random 0-10 ms advDelay
    gaps = device.interval_s + rng.uniform(0, ADV_DELAY_MAX_S, size=n_events)

    # Start at a random offset so devices are not synchronized at t=0
    gaps[0] = rng.uniform(0, device.interval_s)

    # Running total of the gaps gives each packet's time (drop any past the end of the run)
    times = np.cumsum(gaps)
    return times[times < duration_s]


def addresses_at(times: np.ndarray, rotation_s: float | None,
                 rng: np.random.Generator) -> np.ndarray:
    """Return the address the device uses at each timestamp."""
    # If the device never rotates then repeat one random address for every packet
    if rotation_s is None:
        return np.full(len(times), random_address(rng))

    # Else number the rotation periods
    periods = (times // rotation_s).astype(int)

    # Generate one random address per period and look up each packet's address
    addresses = np.array([random_address(rng) for _ in range(periods.max() + 1)])
    return addresses[periods]


def simulate_rssi(base_rssi: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Return n RSSI readings with base level, Gaussian noise, and multipath dips."""
    # Add Gaussian noise (3 dB standard deviation) around the base RSSI
    rssi = rng.normal(base_rssi, RSSI_NOISE_DB, size=n)

    # Choose about 5% of packets for a multipath dip and subtract 10-20 dB from them
    is_dip = rng.random(n) < DIP_CHANCE
    rssi[is_dip] -= rng.uniform(10, 20, size=is_dip.sum())

    # Clip to a realistic range (-100 to -30 dBm) and round
    return np.clip(rssi, -100, -30).round().astype(int)


def simulate_device(device: Device, duration_s: float,
                    rng: np.random.Generator) -> pd.DataFrame:
    """Build every packet one device transmits, before any sniffer losses."""
    # Get the transmit times
    times = event_times(device, duration_s, rng)
    n = len(times)

    # Get the address in use at each time
    addresses = addresses_at(times, device.rotation_s, rng)

    # Get the RSSI of each packet
    rssi = simulate_rssi(device.base_rssi, n, rng)

    # Set the PDU type: the device's own type, with about 10% of an ADV_IND device's packets relabeled SCAN_RSP
    is_scan_rsp = (device.pdu_type == "ADV_IND") & (rng.random(n) < SCAN_RSP_CHANCE)
    pdu_types = np.where(is_scan_rsp, "SCAN_RSP", device.pdu_type)

    # Choose a random advertising channel per packet
    channels = rng.choice(CHANNELS, size=n)

    # Build the DataFrame
    return pd.DataFrame({
        "timestamp": times,
        "address": addresses,
        "pdu_type": pdu_types,
        "rssi": rssi,
        "channel": channels,
        "label": "none",
        "device_id": device.device_id,
    })


def choose_windows(n: int, duration_s: float,
                   rng: np.random.Generator) -> list[tuple[float, float]]:
    """Return n non-overlapping (start, end) attack windows at random times."""
    # No attacks means no windows
    if n == 0:
        return []

    # Split the run into n equal slots, one window per slot, so windows never overlap
    slot_s = duration_s / n
    if slot_s < WINDOW_MAX_S:
        raise ValueError(f"A {duration_s:.0f} s run is too short for {n} attack windows")

    windows = []
    for i in range(n):
        # Choose a window length between 20-40 s
        length_s = rng.uniform(WINDOW_MIN_S, WINDOW_MAX_S)

        # Choose a random start that keeps the whole window inside slot i
        start = rng.uniform(i * slot_s, (i + 1) * slot_s - length_s)

        # Round to milliseconds so the ground-truth file matches the windows exactly
        windows.append((round(start, 3), round(start + length_s, 3)))

    # Shuffle so the attacks do not always happen in the order they were listed
    rng.shuffle(windows)
    return windows


def choose_victim(packets: pd.DataFrame, devices: list[Device],
                  rng: np.random.Generator) -> tuple[Device, str]:
    """Choose a random fixed-address device to attack and return it with its address."""
    # Keep the devices that never rotate, so one attack maps to exactly one address
    fixed = [device for device in devices if device.rotation_s is None]
    if not fixed:
        raise ValueError("No fixed-address device to attack (try another seed or more devices)")

    # Choose one of them at random
    victim = fixed[rng.integers(len(fixed))]

    # Look up its address from its first packet
    address = packets.loc[packets["device_id"] == victim.device_id, "address"].iloc[0]
    return victim, address


def window_times(start: float, end: float, gap_min_s: float, gap_max_s: float,
                 rng: np.random.Generator) -> np.ndarray:
    """Return timestamps inside a window, separated by random gaps of gap_min_s to gap_max_s."""
    # Work out how many packets can fit if every gap is the shortest one
    n = int(np.ceil((end - start) / gap_min_s))

    # Generate the gaps and add them up from the start of the window
    gaps = rng.uniform(gap_min_s, gap_max_s, size=n)
    times = start + np.cumsum(gaps)

    # Drop any past the end of the window
    return times[times < end]


def build_attack_packets(times: np.ndarray, address: str, base_rssi: float, pdu_type: str,
                         label: str, rng: np.random.Generator) -> pd.DataFrame:
    """Build the packets an attack adds, all from one address with one label."""
    n = len(times)
    return pd.DataFrame({
        "timestamp": times,
        "address": address,
        "pdu_type": pdu_type,
        "rssi": simulate_rssi(base_rssi, n, rng),
        "channel": rng.choice(CHANNELS, size=n),
        "label": label,
        "device_id": INJECTED_ID,
    })


def inject_flood(packets: pd.DataFrame, start: float, end: float,
                 rng: np.random.Generator) -> tuple[pd.DataFrame, str]:
    """Add a new address that sends a packet every 5-10 ms for the whole window."""
    # Generate the attacker's address and signal strength
    address = random_address(rng)
    base_rssi = rng.uniform(-90, -40)

    # Generate packet times 5-10 ms apart
    times = window_times(start, end, FLOOD_GAP_MIN_S, FLOOD_GAP_MAX_S, rng)

    # Build the flood packets and add them to the capture
    flood = build_attack_packets(times, address, base_rssi, "ADV_IND", "flood", rng)
    return pd.concat([packets, flood], ignore_index=True), address


def inject_spoof(packets: pd.DataFrame, devices: list[Device], start: float, end: float,
                 rng: np.random.Generator) -> tuple[pd.DataFrame, str]:
    """Add a second transmitter that reuses a real device's address during the window."""
    # Choose the device to impersonate
    victim, address = choose_victim(packets, devices, rng)

    # Move the attacker's RSSI 20-30 dB away from the victim's (toward the middle of the range)
    rssi_gap = rng.uniform(SPOOF_RSSI_MIN_DB, SPOOF_RSSI_MAX_DB)
    if victim.base_rssi > -65:
        attacker_rssi = victim.base_rssi - rssi_gap
    else:
        attacker_rssi = victim.base_rssi + rssi_gap

    # Generate the attacker's packet times by copying the victim's interval and advDelay
    gap_min_s = victim.interval_s
    gap_max_s = victim.interval_s + ADV_DELAY_MAX_S
    times = window_times(start, end, gap_min_s, gap_max_s, rng)

    # Build the spoofed packets and add them to the capture (the victim keeps sending too)
    spoofed = build_attack_packets(times, address, attacker_rssi, victim.pdu_type, "spoof", rng)
    return pd.concat([packets, spoofed], ignore_index=True), address


def inject_timing(packets: pd.DataFrame, devices: list[Device], start: float, end: float,
                  rng: np.random.Generator) -> tuple[pd.DataFrame, str]:
    """Replace a real device's packets in the window with irregularly timed ones."""
    # Choose the device whose timing goes wrong
    victim, address = choose_victim(packets, devices, rng)

    # Remove the victim's normal packets inside the window
    in_window = (packets["device_id"] == victim.device_id) & packets["timestamp"].between(start, end)
    packets = packets[~in_window]

    # Generate packet times with gaps between 0.2x-2x the victim's normal interval
    gap_min_s = TIMING_GAP_MIN * victim.interval_s
    gap_max_s = TIMING_GAP_MAX * victim.interval_s
    times = window_times(start, end, gap_min_s, gap_max_s, rng)

    # Build the irregular packets (same address and signal strength) and add them to the capture
    irregular = build_attack_packets(times, address, victim.base_rssi, victim.pdu_type, "timing", rng)
    return pd.concat([packets, irregular], ignore_index=True), address


def inject_attacks(packets: pd.DataFrame, devices: list[Device], kinds: list[str],
                   duration_s: float, rng: np.random.Generator) -> tuple[pd.DataFrame, list[dict]]:
    """Inject each requested attack in its own window and return the packets and ground truth."""
    # Choose one time window per attack
    windows = choose_windows(len(kinds), duration_s, rng)

    ground_truth = []
    for kind, (start, end) in zip(kinds, windows):
        # Run the matching attack
        if kind == "flood":
            packets, address = inject_flood(packets, start, end, rng)
        elif kind == "spoof":
            packets, address = inject_spoof(packets, devices, start, end, rng)
        else:
            packets, address = inject_timing(packets, devices, start, end, rng)

        # Record what happened, where and when
        ground_truth.append({"type": kind, "address": address, "start": start, "end": end})
    return packets, ground_truth


def drop_packets(packets: pd.DataFrame, rng: np.random.Generator,
                 miss_rate: float = 0.15) -> pd.DataFrame:
    """Randomly remove the packets a single-radio sniffer would have missed."""
    # Generate one uniform random number per packet
    draws = rng.random(len(packets))

    # Keep the rows whose number is at or above miss_rate
    return packets[draws >= miss_rate]


def write_packets(packets: pd.DataFrame, out_dir: Path) -> Path:
    """Sort packets by time and save them as packets.csv."""
    # Create the output folder if it does not already exist
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sort by timestamp and drop the internal device_id column
    packets = packets.sort_values("timestamp").drop(columns="device_id")

    # Round timestamps to microseconds and write the CSV without the index
    packets["timestamp"] = packets["timestamp"].round(6)
    path = out_dir / "packets.csv"
    packets.to_csv(path, index=False)
    return path


def write_ground_truth(ground_truth: list[dict], out_dir: Path) -> Path:
    """Save the attack windows as ground_truth.json."""
    # Create the output folder if it does not already exist
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write the list of windows as indented JSON
    path = out_dir / "ground_truth.json"
    with open(path, "w") as file:
        json.dump(ground_truth, file, indent=2)
    return path


def parse_args() -> argparse.Namespace:
    """Read the command-line options."""
    # Define each option: flag name, type, default value, and help
    parser = argparse.ArgumentParser(description="Simulate BLE advertising traffic.")
    parser.add_argument("--devices", type=int, default=8, help="number of devices")
    parser.add_argument("--minutes", type=float, default=10, help="length of the capture")
    parser.add_argument("--attacks", default="flood,spoof,timing",
                        help="comma-separated attacks to inject, or 'none'")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--out-dir", type=Path, default=Path("data"), help="output folder")
    args = parser.parse_args()

    # Turn the attack text into a list ("none" = empty list)
    if args.attacks == "none":
        args.attacks = []
    else:
        args.attacks = args.attacks.split(",")

    # Stop with a clear message if an attack name is not recognized
    for kind in args.attacks:
        if kind not in ATTACK_KINDS:
            parser.error(f"unknown attack '{kind}' (choose from: {', '.join(ATTACK_KINDS)})")
    return args


def main() -> None:
    """Run the simulation end to end and print a summary."""
    # Parse arguments and create the seeded random generator
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    duration_s = args.minutes * 60

    # Build the devices
    devices = make_devices(args.devices, rng)

    # Simulate each device and concatenate the results
    per_device = [simulate_device(device, duration_s, rng) for device in devices]
    packets = pd.concat(per_device, ignore_index=True)

    # Inject the attacks and record where they happened
    packets, ground_truth = inject_attacks(packets, devices, args.attacks, duration_s, rng)
    n_sent = len(packets)

    # Drop missed packets
    packets = drop_packets(packets, rng)

    # Write both files and print a summary
    packets_path = write_packets(packets, args.out_dir)
    truth_path = write_ground_truth(ground_truth, args.out_dir)
    print(f"Devices: {len(devices)}   Duration: {duration_s:.0f} s")
    print(f"Packets sent: {n_sent}   captured: {len(packets)}")
    print(packets["label"].value_counts().to_string())
    for window in ground_truth:
        print(f"{window['type']:<7}{window['address']}  {window['start']:.1f}-{window['end']:.1f} s")
    print(f"Wrote {packets_path} and {truth_path}")


if __name__ == "__main__":
    main()
