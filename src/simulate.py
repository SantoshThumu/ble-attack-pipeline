"""
Generate BLE advertising traffic and save it as a labeled CSV.

Run from the repo root:
    python -m src.simulate --devices 8 --minutes 10 --seed 42
"""

import argparse
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
    bytes = rng.integers(0, 256, size=6)

    # Format each as two uppercase hex digits and join them with ':'
    return ":".join(f"{byte:02X}" for byte in bytes)


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


def parse_args() -> argparse.Namespace:
    """Read the command-line options."""
    # Define each option: flag name, type, default value, and help
    parser = argparse.ArgumentParser(description="Simulate BLE advertising traffic.")
    parser.add_argument("--devices", type=int, default=8, help="number of devices")
    parser.add_argument("--minutes", type=float, default=10, help="length of the capture")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--out-dir", type=Path, default=Path("data"), help="output folder")

    # Parse and return them
    return parser.parse_args()


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
    n_sent = len(packets)

    # Drop missed packets
    packets = drop_packets(packets, rng)

    # Write the CSV and print packet counts
    path = write_packets(packets, args.out_dir)
    print(f"Devices: {len(devices)}   Duration: {duration_s:.0f} s")
    print(f"Packets sent: {n_sent}   captured: {len(packets)}")
    print(packets["label"].value_counts().to_string())
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()