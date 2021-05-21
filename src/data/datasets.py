from io import BytesIO
import pickle
import gzip
from pathlib import Path
from functools import cached_property
from dataclasses import dataclass
from PIL import Image
import json
import torch
from collections import Counter

from torch.nn.utils.rnn import pad_sequence
from src.data.extract_data import RAW_FILE_NAME, get_file_tree, get_trace_data
import zipfile
import pandas as pd
from tqdm import tqdm
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.data import DataLoader, Dataset

project_dir = Path(__file__).resolve().parents[2]

raw_path = project_dir / "data" / "raw"
interim_path = project_dir / "data" / "interim"
processed_path = project_dir / "data" / "processed"


def get_loader(dataset, batch_size, pin_memory=False, generator=None):

    sampler = BatchSampler(
        RandomSampler(dataset, generator=generator), batch_size=batch_size, drop_last=False
    )
    return DataLoader(
        dataset,
        batch_size=None,
        sampler=sampler,
        pin_memory=pin_memory,
    )


@dataclass
class TestDataset:
    pass


class SiteDataset(Dataset):
    def __init__(self, site_id: str, **kwargs) -> None:

        self.site_id = site_id

        file_tree = get_file_tree()
        floor_ids = file_tree["train"][self.site_id]
        self.floors = [FloorDataset(self.site_id, floor_id, **kwargs) for floor_id in floor_ids]


class FloorDataset(Dataset):
    def __init__(
        self, site_id: str, floor_id: str, sampling_interval=100, wifi_threshold=100
    ) -> None:

        self.site_id = site_id
        self.floor_id = floor_id

        self.sampling_interval = sampling_interval
        self.wifi_threshold = wifi_threshold

        file_tree = get_file_tree()
        trace_ids = file_tree["train"][self.site_id][self.floor_id]

        self.traces = [
            TraceData(self.site_id, self.floor_id, trace_id, sampling_interval)
            for trace_id in trace_ids
        ]

    @cached_property
    def image(self):

        image_path = Path("metadata") / self.site_id / self.floor_id / "floor_image.png"
        with zipfile.ZipFile(raw_path / RAW_FILE_NAME) as zip_file:
            file_path = zipfile.Path(zip_file) / image_path

            with file_path.open("rb") as f:
                bytes_ = BytesIO(f.read())

        return Image.open(bytes_)

    @cached_property
    def info(self):

        info_path = Path("metadata") / self.site_id / self.floor_id / "floor_info.json"

        with zipfile.ZipFile(raw_path / RAW_FILE_NAME) as zip_file:

            file_path = zipfile.Path(zip_file) / info_path
            with file_path.open("r") as f:
                return json.load(f)

    def __len__(self):
        return len(self.traces)

    def __getitem__(self, indices):

        time_unpadded, position_unpadded, wifi_unpadded = self._generate_tensors()

        mini_batch_index = indices
        mini_batch_length = torch.tensor([len(time_unpadded[i]) for i in indices])

        mini_batch_time = pad_sequence(
            [time_unpadded[i] for i in indices], batch_first=True
        )

        mini_batch_position = pad_sequence(
            [position_unpadded[i] for i in indices], batch_first=True
        )
        mini_batch_position_mask = ~mini_batch_position.isnan().any(dim=-1)
        for i, length in enumerate(mini_batch_length):
            mini_batch_position_mask[i, length:] = False
        mini_batch_position[~mini_batch_position_mask] = 0

        mini_batch_wifi = pad_sequence(
            [wifi_unpadded[i] for i in indices], batch_first=True
        )
        mini_batch_wifi_mask = ~mini_batch_wifi.isnan()
        for i, length in enumerate(mini_batch_length):
            mini_batch_wifi_mask[i, length:, :] = False
        mini_batch_wifi[~mini_batch_wifi_mask] = 0

        return (
            mini_batch_index,
            mini_batch_length,
            mini_batch_time,
            mini_batch_position,
            mini_batch_position_mask,
            mini_batch_wifi,
            mini_batch_wifi_mask,
        )

    @property
    def K(self):
        if hasattr(self, "bssids_"):
            return len(self.bssids_)
        else:
            self._generate_tensors()
            return len(self.bssids_)

    def _generate_tensors(self):

        sub_path = Path("train") / self.site_id / self.floor_id
        # cached_path = (processed_path / sub_path).with_suffix(".pkl.gz")
        cached_path = (processed_path / sub_path).with_suffix(".pt")

        if cached_path.exists():
            # with gzip.open(cached_path, "rb") as f:
            #     sampling_interval, bssids, data_tensors_unpadded = pickle.load(f)
            data_parameters, bssids, data_tensors_unpadded = torch.load(cached_path)
            if data_parameters == (self.sampling_interval, self.wifi_threshold):
                self.bssids_ = bssids
                return data_tensors_unpadded

        time_unpadded = []
        position_unpadded = []

        wifi_unaligned = []

        for trace in tqdm(self.traces):

            time, position, wifi = trace[0]
            time_unpadded.append(time)
            position_unpadded.append(position)

            wifi_unaligned.append((trace.bssids_, wifi))

        bssid_counter = Counter()
        for bssids_, wifi in wifi_unaligned:
            bssid_counter.update(dict(zip(bssids_, (~wifi.isnan()).sum(0))))

        self.bssids_ = sorted(
            i for i, j in bssid_counter.items() if j >= self.wifi_threshold
        )

        bssid_to_index = {j: i for i, j in enumerate(self.bssids_)}

        wifi_unpadded = []
        for bssids, wifi in wifi_unaligned:
            wifi_aligned = torch.full(
                (wifi.shape[0], len(self.bssids_)), float("nan"), dtype=wifi.dtype
            )
            
            old_index, old_bssid,  = zip(*[(i, bssid) for i, bssid in enumerate(bssids) if bssid in bssid_to_index])
            new_index = [bssid_to_index[bssid] for bssid in old_bssid]
   
            wifi_aligned[:, new_index] = wifi[:, old_index]
            wifi_unpadded.append(wifi_aligned)

        data_tensors_unpadded = (time_unpadded, position_unpadded, wifi_unpadded)

        cached_path.parent.mkdir(parents=True, exist_ok=True)
        data_parameters = (self.sampling_interval, self.wifi_threshold)
        torch.save((data_parameters, self.bssids_, data_tensors_unpadded), cached_path)
        return data_tensors_unpadded


class TraceData:
    """Data for a single trace"""

    def __init__(self, site_id, floor_id, trace_id, sampling_interval=100) -> None:

        self.site_id = site_id
        self.floor_id = floor_id
        self.trace_id = trace_id

        self.sampling_interval = sampling_interval

    @property
    def data(self):
        """Data in pandas format"""
        return self._get_zipped_data()

    def _get_zipped_data(self, cache=True):
        """Loads data from zip file into pandas format"""

        sub_path = Path("train") / self.site_id / self.floor_id / self.trace_id
        cached_path = (interim_path / sub_path).with_suffix(".pkl.gz")

        if cached_path.exists():
            with gzip.open(cached_path, "rb") as f:
                return pickle.load(f)

        data = get_trace_data(sub_path.with_suffix(".txt"))

        if cache:
            cached_path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(cached_path, "wb") as f:
                pickle.dump(data, f)

        return data

    def __len__(self):
        return 1

    def __getitem__(self, idx):

        if idx != 0:
            raise IndexError

        data_tensors = self._generate_tensors()
        return data_tensors

    def _generate_tensors(self):

        sub_path = Path("train") / self.site_id / self.floor_id / self.trace_id
        cached_path = (processed_path / sub_path).with_suffix(".pkl.gz")

        if cached_path.exists():
            with gzip.open(cached_path, "rb") as f:
                sampling_interval, bssids, data_tensors = pickle.load(f)
            if sampling_interval == self.sampling_interval:
                self.bssids_ = bssids
                return data_tensors

        data_frames = self._get_zipped_data(cache=False)

        position_df = data_frames["TYPE_WAYPOINT"]
        position_df = position_df.rename(columns=lambda x: f"pos:{x}")

        wifi_df = data_frames["TYPE_WIFI"]

        def _apply(group):
            bssid = group["bssid"].iloc[0]
            return pd.Series(group["rssi"], name=f"wifi:{bssid}")

        wifi_split = pd.DataFrame()
        wifi_grouped = wifi_df.groupby("bssid").apply(_apply)

        self.bssids_ = wifi_grouped.index.get_level_values(0).unique().to_list()
        for bssid in self.bssids_:
            try:
                wifi_split[bssid] = wifi_grouped[bssid]
            except ValueError:
                # Sometimes more than one observation per time
                wifi_split[bssid] = wifi_grouped[bssid][
                    ~wifi_grouped[bssid].index.duplicated()
                ]
                pass

        end_time = max(position_df.index[-1], wifi_df.index[-1])
        new_index = pd.timedelta_range(0, end_time, freq=f"{self.sampling_interval}ms")
        new_index.name = "time"

        resampled_data = (
            pd.DataFrame(index=new_index)
            .pipe(pd.merge_ordered, position_df, "time")
            .pipe(pd.merge_ordered, wifi_split, "time")
            .bfill(limit=1)
            .set_index("time")
            .loc[new_index]
        )

        time = resampled_data.index.total_seconds()
        position = resampled_data[position_df.columns].values
        wifi = resampled_data[wifi_split.columns].values

        data_tensors = (
            torch.tensor(time, dtype=torch.float64),
            torch.tensor(position, dtype=torch.float64),
            torch.tensor(wifi, dtype=torch.float64),
        )

        cached_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(cached_path, "wb") as f:
            pickle.dump((self.sampling_interval, self.bssids_, data_tensors), f)

        return data_tensors


if __name__ == "__main__":
    pass