import os
import sys
import numpy as np
import pandas as pd
import h5py  # HDF5 library
from obspy.clients.fdsn import Client
from obspy import UTCDateTime

# Fix imports for utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import cal_snr

# Configuration
config = {
    "region": "Indonesia",
    "center": (107.67, -7.19),
    "xlim_degree": [107.67 - 50.0 / 2, 107.67 + 50.0 / 2],
    "ylim_degree": [-7.19 - 50.0 / 2, -7.19 + 50.0 / 2],
    "networks": ["GE"],
    "channels": "BHE,BHN,BHZ",
    "client": "http://geofon.gfz-potsdam.de",
    "starttime": UTCDateTime("2018-01-01"),
    "endtime": UTCDateTime("2018-01-02"),
}

client = Client(config["client"])
print(f"Fetching stations from {config['starttime']} to {config['endtime']}...")

# Fetch station metadata
stations = client.get_stations(
    network=",".join(config["networks"]),
    station="*",
    starttime=config["starttime"],
    endtime=config["endtime"],
    minlongitude=config["xlim_degree"][0],
    maxlongitude=config["xlim_degree"][1],
    minlatitude=config["ylim_degree"][0],
    maxlatitude=config["ylim_degree"][1],
    channel=config["channels"],
    level="response",
)

print("Fetching earthquake events...")
catalog = client.get_events(
    starttime=config["starttime"],
    endtime=config["endtime"],
    minlongitude=config["xlim_degree"][0],
    maxlongitude=config["xlim_degree"][1],
    minlatitude=config["ylim_degree"][0],
    maxlatitude=config["ylim_degree"][1],
    includearrivals=True,
)

# Ensure directories exist
os.makedirs("source_dataset", exist_ok=True)

# Metadata CSV
metadata_file = "source_dataset/obspy_metadata.csv"
metadata_exists = os.path.exists(metadata_file)

# Open metadata file in append mode
with open(metadata_file, mode="a") as f:
    if not metadata_exists:  # Only write header if file doesn't exist
        f.write("source_origin_time,source_end_time,source_magnitude,p_arrival_sample,s_arrival_sample,snr,waveform_index,station\n")

# HDF5 file for waveforms
waveform_file = "source_dataset/obspy_waveforms.h5"

# Check if HDF5 dataset exists
if os.path.exists(waveform_file):
    with h5py.File(waveform_file, "a") as hf:
        if "waveforms" in hf:
            waveform_index = hf["waveforms"].shape[0]  # Get last index
        else:
            waveform_index = 0
else:
    waveform_index = 0

# Process each event
with h5py.File(waveform_file, "a") as hf:  # Open HDF5 in append mode
    if "waveforms" not in hf:
        max_shape = (None, 3, 1200)  # Shape: (samples, 3 channels, 1200 time points)
        hf.create_dataset("waveforms", shape=(0, 3, 1200), maxshape=max_shape, dtype=np.float32, compression="gzip")

    for event in catalog:
        origin = event.origins[0]
        event_time = origin.time
        end_event_time = origin.time + 60
        magn = event.magnitudes[0].mag if event.magnitudes else None

        for network in stations:
            for station in network:
                station_code = station.code

                # Get station-specific pick times
                ppk_actual, spk_actual = None, None
                for pick in event.picks:
                    if pick.waveform_id and pick.waveform_id.station_code == station_code:
                        if pick.phase_hint == "P":
                            ppk_actual = int((pick.time - origin.time) * 20)  # 20 Hz sampling rate
                        elif pick.phase_hint == "S":
                            spk_actual = int((pick.time - origin.time) * 20)

                # Fetch waveform data for the station
                try:
                    waveforms = client.get_waveforms(
                        network=network.code,
                        station=station_code,
                        location="*",
                        channel=config["channels"],
                        starttime=event_time,
                        endtime=end_event_time,
                    )
                except Exception as e:
                    print(f"Failed to fetch waveforms for station {station_code}: {e}")
                    continue

                traces = [tr.data.astype(np.float32) for tr in waveforms]

                if traces:
                    min_len = min(len(tr) for tr in traces)
                    data = np.stack([tr[:min_len] for tr in traces])  # Shape: (3, time_samples)

                    # **FIX: Pad or truncate to 1200 samples**
                    target_length = 1200
                    padded_data = np.zeros((3, target_length), dtype=np.float32)  # Create empty padded array
                    padded_data[:, :min(target_length, min_len)] = data[:, :target_length]  # Fill with actual data

                    # Compute SNR
                    snr_value = cal_snr(padded_data, ppk_actual / 20) if ppk_actual else 0.0

                    # Write metadata immediately
                    with open(metadata_file, "a") as f:
                        f.write(f"{event_time},{end_event_time},{magn},{ppk_actual},{spk_actual},{snr_value},{waveform_index},{station_code}\n")

                    # Append new waveform data to HDF5
                    waveforms_ds = hf["waveforms"]
                    waveforms_ds.resize((waveforms_ds.shape[0] + 1, 3, target_length))  # Expand dataset only in first dimension
                    waveforms_ds[-1] = padded_data  # Add new waveform

                    # Increment index
                    waveform_index += 1

print("Waveforms and metadata updated successfully.")