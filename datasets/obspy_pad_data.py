import numpy as np
import os

# File path for waveforms
waveform_file = "source_dataset/obspy_waveforms.npy"

# Check if waveform file exists
if not os.path.exists(waveform_file):
    print("No waveform file found.")
    exit()

# Load waveform data
waveforms = np.load(waveform_file, allow_pickle=True)

if waveforms.size == 0:
    print("Waveform file is empty.")
    exit()

# Determine the maximum waveform length
max_length = max(data.shape[1] for data in waveforms)

# Pad all waveforms to the same length
padded_waveforms = np.array([
    np.pad(data, ((0, 0), (0, max_length - data.shape[1])), mode="constant") for data in waveforms
], dtype=np.float32)

# Save the padded waveforms
np.save(waveform_file, padded_waveforms)
print(f"All waveforms padded to {max_length} samples and saved.")