import h5py
import pandas as pd
import numpy as np

# File paths
waveform_file = "source_dataset/obspy_waveforms_1month.h5"
metadata_file = "source_dataset/obspy_metadata_1month.csv"

# Load HDF5 file
with h5py.File(waveform_file, "r") as hf:
    waveforms_ds = hf["waveforms"]

    # Print dataset shape
    print(f"Waveform dataset shape: {waveforms_ds.shape}")  # (num_samples, 3, 1200)

    # Get the max length of waveforms
    max_waveform_length = waveforms_ds.shape[2]
    print(f"Max waveform length: {max_waveform_length}")

    # Select one waveform (first one)
    sample_waveform = waveforms_ds[0]  # Shape: (3, time_samples)

# Load metadata CSV
metadata_df = pd.read_csv(metadata_file)

# Get metadata for the first waveform
first_metadata = metadata_df.iloc[0]

# Print metadata
print("\nFirst Waveform Metadata:")
print(first_metadata.to_string())  # Print full metadata

# Print the entire waveform data
print("\nFirst Waveform Data (Full):")
print(sample_waveform)  # Print all samples for all channels