import os
import torch
import sys
import numpy as np
import pandas as pd
# Get the absolute path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from models import create_model, load_checkpoint
from training.postprocess import process_outputs
import argparse

# Path to local dataset
metadata_path = "../datasets/source_dataset/obspy_metadata_1month.csv"
waveform_path = "../datasets/source_dataset/obspy_waveforms.npy"

# Normalize function
def normalize(data: np.ndarray, mode: str):
    data -= np.mean(data, axis=1, keepdims=True)
    if mode == "max":
        max_data = np.max(data, axis=1, keepdims=True)
        max_data[max_data == 0] = 1
        data /= max_data
    elif mode == "std":
        std_data = np.std(data, axis=1, keepdims=True)
        std_data[std_data == 0] = 1
        data /= std_data
    return data

# Load model
def load_model(model_name: str, ckpt_path: str, device: torch.device, in_channels: int = 3):
    model = create_model(model_name=model_name, in_channels=in_channels)
    ckpt = load_checkpoint(ckpt_path, device=device)
    model.load_state_dict(ckpt["model_dict"] if "model_dict" in ckpt else ckpt)
    model.to(device)
    return model

# Load all metadata from local CSV
def load_local_events():
    return pd.read_csv(metadata_path)

# Load waveforms from local .npy file
def load_local_waveforms():
    return np.load(waveform_path, allow_pickle=True)

# Run inference on all available data
def run_local_inference(model, device, csv_file="./data_prediction/local_seismic_predictions.csv"):
    metadata = load_local_events()
    waveforms = load_local_waveforms()
    results_list = []

    print(f"Loaded {len(metadata)} events from local dataset.")

    for _, event in metadata.iterrows():
        waveform_index = int(event["waveform_index"])
        if waveform_index >= len(waveforms):
            print(f"Skipping event {event['waveform_index']} - Invalid waveform index")
            continue

        waveform = waveforms[waveform_index]
        waveform = np.nan_to_num(waveform, nan=0)  # Replace NaNs with 0
        waveform = normalize(waveform, mode="std")

        waveform_tensor = torch.from_numpy(waveform).reshape(1, 3, -1).to(device)

        print(f"Running inference for event {event['waveform_index']}")
        preds_tensor = model(waveform_tensor)
        print("Inference done")

        results = process_outputs(
            args=argparse.Namespace(
                ppk_threshold=0.3, spk_threshold=0.3, det_threshold=0.5,
                min_peak_dist=10, max_detect_event_num=1
            ),
            outputs=preds_tensor,
            label_names=[["ppk", "spk"]],
            sampling_rate=20,
        )

        print(results)

        ppk_pred = results["ppk"][0][0] / 20.0 if results["ppk"][0][0] > 0 else None
        spk_pred = results["spk"][0][0] / 20.0 if results["spk"][0][0] > 0 else None

        ppk_actual = event["p_arrival_sample"] / 20.0 if pd.notna(event["p_arrival_sample"]) else None
        spk_actual = event["s_arrival_sample"] / 20.0 if pd.notna(event["s_arrival_sample"]) else None

        ppk_error = (ppk_pred - ppk_actual) if ppk_pred is not None and ppk_actual is not None else None
        spk_error = (spk_pred - spk_actual) if spk_pred is not None and spk_actual is not None else None

        print(f"Event: {event['waveform_index']}, P Pred: {ppk_pred}, P Actual: {ppk_actual}, Error: {ppk_error}")
        print(f"Event: {event['waveform_index']}, S Pred: {spk_pred}, S Actual: {spk_actual}, Error: {spk_error}")

        results_list.append({
            "waveform_index": event["waveform_index"],
            "ppk_pred": ppk_pred.item() if isinstance(ppk_pred, torch.Tensor) else ppk_pred,
            "spk_pred": spk_pred.item() if isinstance(spk_pred, torch.Tensor) else spk_pred,
            "ppk_actual": ppk_actual,
            "spk_actual": spk_actual,
            "ppk_error": ppk_error.item() if isinstance(ppk_error, torch.Tensor) else ppk_error,
            "spk_error": spk_error.item() if isinstance(spk_error, torch.Tensor) else spk_error
        })

    # Save results to CSV
    df = pd.DataFrame(results_list)
    df.to_csv(csv_file, index=False)
    print(f"Predictions saved to {csv_file}")

# Example usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("seist_s_dpk", "../pretrained/seist_s_dpk_diting.pth", device)

    # Run inference on all data
    run_local_inference(model, device)