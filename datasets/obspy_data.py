import numpy as np
import pandas as pd
import h5py  # NEW: HDF5 support
from .base import DatasetBase
from utils import logger, cal_snr
from ._factory import register_dataset


class ObsPy(DatasetBase):
    """ObsPy-based Dataset that fetches seismic event data from pre-saved HDF5 dataset."""

    _name = "obspy"
    _channels = ["e", "n", "z"]  # Map BHE -> e, BHN -> n, BHZ -> z
    _sampling_rate = 20

    def __init__(
            self,
            seed: int,
            mode: str,
            shuffle: bool = True,
            data_split: bool = True,
            train_size: float = 0.8,
            val_size: float = 0.1,
            **kwargs
    ):

        super().__init__(
            seed=seed,
            mode=mode,
            data_dir=None,  # No local dataset directory
            shuffle=shuffle,
            data_split=data_split,
            train_size=train_size,
            val_size=val_size,
        )
        self._meta_data = self._load_meta_data()

    def _load_meta_data(self):
        """Loads metadata from earthquake events."""

        metadata = pd.read_csv("datasets/source_dataset/obspy_metadata.csv")
        if self._shuffle:
            metadata = metadata.sample(frac=1, random_state=self._seed).reset_index(drop=True)

        return metadata

    def _load_event_data(self, idx: int):
        """Loads waveform data from HDF5 instead of .npy."""

        # Load metadata from CSV
        target_event = self._meta_data.iloc[idx]

        # Load waveform index
        waveform_index = int(target_event["waveform_index"])

        # Read waveform from HDF5
        with h5py.File("datasets/source_dataset/obspy_waveforms.h5", "r") as hf:
            if "waveforms" not in hf:
                raise ValueError("Waveforms dataset not found in HDF5 file!")
            data = hf["waveforms"][waveform_index]  # Load only required waveform

        # Ensure correct dtype & replace NaNs
        data = np.array(data, dtype=np.float32)
        data = np.nan_to_num(data)

        # Create event dictionary
        event = {
            "data": data,
            "ppks": [target_event["p_arrival_sample"]] if pd.notna(target_event["p_arrival_sample"]) else [],
            "spks": [target_event["s_arrival_sample"]] if pd.notna(target_event["s_arrival_sample"]) else [],
            "emg": [target_event["source_magnitude"]] if pd.notna(target_event["source_magnitude"]) else [],
            "clr": [0],  # For compatibility
            "snr": target_event["snr"],
        }

        return event, target_event.to_dict()


@register_dataset
def obspy(**kwargs):
    logger.info("Registering ObsPy dataset with HDF5 support...")
    return ObsPy(**kwargs)
