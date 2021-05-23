import functools
from io import BytesIO
import pickle
import gzip
from pathlib import Path
from functools import cached_property
from dataclasses import dataclass
from PIL import Image
import json
from pandas._libs.tslibs import Timedelta
import torch
from collections import Counter
import functools
import random

from torch.nn.utils.rnn import pad_sequence
from src.data.extract_data import RAW_FILE_NAME, get_file_tree, get_trace_data
import zipfile
import pandas as pd
from tqdm import tqdm
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.data import DataLoader, Dataset

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

project_dir = Path(__file__).resolve().parents[2]
#project_dir = Path("/work3/s164221")

raw_path = project_dir / "data" / "raw"
interim_path = project_dir / "data" / "interim"
processed_path = project_dir / "data" / "processed"


def get_loader(dataset, batch_size, pin_memory=False, generator=None):

    sampler = BatchSampler(
        RandomSampler(dataset, generator=generator),
        batch_size=batch_size,
        drop_last=False,
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
        self.floors = [
            FloorDataset(self.site_id, floor_id, **kwargs) for floor_id in floor_ids
        ]


class FloorDataset(Dataset):
    def __init__(
        self,
        site_id: str,
        floor_id: str,
        sampling_interval=100,
        wifi_threshold=100,
        include_wifi=True,
        include_beacon=False,
        validation_percent=None,
        test_percent=None,
        split_seed=123,
    ) -> None:
        self.unpadded_tensors = None
        self.site_id = site_id
        self.floor_id = floor_id

        self.sampling_interval = sampling_interval
        self.wifi_threshold = wifi_threshold

        self.include_wifi = include_wifi
        self.include_beacon = include_beacon

        file_tree = get_file_tree()
        trace_ids = file_tree["train"][self.site_id][self.floor_id]

        self.traces = [
            TraceData(self.site_id, self.floor_id, trace_id, sampling_interval)
            for trace_id in trace_ids
        ]

        # ---- TEST TRAIN SPLIT -----
        
        trace_indices = set(range(len(self.traces)))
        self.validation_mask = torch.full((len(self.traces),), False)
        self.test_mask = torch.full((len(self.traces),), False)

        if validation_percent is not None and test_percent is not None:
            random.seed(split_seed)

        if validation_percent is not None:
            validation_indices = random.choices(
                list(trace_indices), k=int(len(self.traces) * validation_percent)
            )
            trace_indices.difference_update(validation_indices)
            self.validation_mask[validation_indices] = True

        if test_percent is not None:
            test_indices = random.choices(
                list(trace_indices), k=int(len(self.traces) * test_percent)
            )
            trace_indices.difference_update(test_indices)
            self.test_mask[test_indices] = True

        


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

        (
            time_unpadded,
            position_unpadded,
            wifi_unpadded,
            beacon_unpadded,
        ) = self._generate_tensors()

        mini_batch_index = indices
        mini_batch_length = torch.tensor([len(time_unpadded[i]) for i in indices], device=device)

        mini_batch_time = pad_sequence(
            [time_unpadded[i] for i in indices], batch_first=True
        )

        mini_batch_position = pad_sequence(
            [position_unpadded[i] for i in indices], batch_first=True
        )
        mini_batch_position_mask = ~mini_batch_position.isnan().any(dim=-1)
        for i, length in enumerate(mini_batch_length):
            mini_batch_position_mask[i, length:] = False

        mini_batch_validation_mask = self.validation_mask[mini_batch_index]
        mini_batch_test_mask = self.test_mask[mini_batch_index]

        mini_batch_position_mask[mini_batch_validation_mask, :] = False
        mini_batch_position_mask[mini_batch_test_mask, :] = False

        mini_batch_position[~mini_batch_position_mask] = 0

        out_tensors = [
            mini_batch_index,
            mini_batch_length,
            mini_batch_time,
            mini_batch_position,
            mini_batch_position_mask,
        ]

        if self.include_wifi:

            mini_batch_wifi = pad_sequence(
                [wifi_unpadded[i] for i in indices], batch_first=True
            )
            mini_batch_wifi_mask = ~mini_batch_wifi.isnan()
            for i, length in enumerate(mini_batch_length):
                mini_batch_wifi_mask[i, length:, :] = False
            mini_batch_wifi[~mini_batch_wifi_mask] = 0

            out_tensors.extend([mini_batch_wifi, mini_batch_wifi_mask])

        if self.include_beacon:
            mini_batch_beacon = pad_sequence(
                [beacon_unpadded[i] for i in indices], batch_first=True
            )
            mini_batch_beacon_mask = ~mini_batch_beacon.isnan()
            for i, length in enumerate(mini_batch_length):
                mini_batch_beacon_mask[i, length:, :] = False
            mini_batch_beacon[~mini_batch_beacon_mask] = 0

            out_tensors.extend([mini_batch_beacon, mini_batch_beacon_mask])

        return out_tensors

    @property
    def K(self):
        if hasattr(self, "bssids_"):
            return len(self.bssids_)
        else:
            self._generate_tensors()
            return len(self.bssids_)

    @property
    def B(self):
        if hasattr(self, "beacon_ids_"):
            return len(self.beacon_ids_)
        else:
            self._generate_tensors()
            return len(self.beacon_ids_)

    def _generate_tensors(self):
        if self.unpadded_tensors is not None:
            return self.unpadded_tensors
        sub_path = Path("train") / self.site_id / self.floor_id
        # cached_path = (processed_path / sub_path).with_suffix(".pkl.gz")
        cached_path = (processed_path / sub_path).with_suffix(".pt")

        if cached_path.exists():
            # with gzip.open(cached_path, "rb") as f:
            #     sampling_interval, bssids, data_tensors_unpadded = pickle.load(f)
            data_parameters, bssids, beacon_ids, data_tensors_unpadded = torch.load(
                cached_path, map_location=device
            )
            if data_parameters == (self.sampling_interval, self.wifi_threshold):
                self.bssids_ = bssids
                self.beacon_ids_ = beacon_ids
                return data_tensors_unpadded

        time_unpadded = []
        position_unpadded = []

        wifi_unaligned = []
        beacon_unaligned = []

        for trace in tqdm(self.traces):

            time, position, wifi, beacon = trace[0]
            time_unpadded.append(time)
            position_unpadded.append(position)

            wifi_unaligned.append((trace.bssids_, wifi))
            beacon_unaligned.append((trace.beacon_ids_, beacon))

        ## Aligning floor wide wifi signals
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

            old_index, old_bssid, = zip(
                *[
                    (i, bssid)
                    for i, bssid in enumerate(bssids)
                    if bssid in bssid_to_index
                ]
            )
            new_index = [bssid_to_index[bssid] for bssid in old_bssid]

            wifi_aligned[:, new_index] = wifi[:, old_index]
            wifi_unpadded.append(wifi_aligned)

        ## Aligning floor wide beacon signals
        self.beacon_ids_ = sorted(
            set(
                beacon_id
                for (beacon_ids, beacon) in beacon_unaligned
                for beacon_id in beacon_ids
            )
        )

        beacon_id_to_index = {j: i for i, j in enumerate(self.beacon_ids_)}

        beacon_unpadded = []
        for (beacon_ids, beacon) in beacon_unaligned:
            beacon_aligned = torch.full(
                (beacon.shape[0], len(self.beacon_ids_)),
                float("nan"),
                dtype=beacon.dtype,
            )
            beacon_aligned[
                :, [beacon_id_to_index[beacon_id] for beacon_id in beacon_ids]
            ] = beacon
            beacon_unpadded.append(beacon_aligned)

        data_tensors_unpadded = (
            time_unpadded,
            position_unpadded,
            wifi_unpadded,
            beacon_unpadded,
        )

        cached_path.parent.mkdir(parents=True, exist_ok=True)
        data_parameters = (self.sampling_interval, self.wifi_threshold)
        torch.save(
            (data_parameters, self.bssids_, self.beacon_ids_, data_tensors_unpadded),
            cached_path,
        )
        data_tensors_unpadded = [x.to(device=device) for x in data_tensors_unpadded]
        self.unpadded_tensors = data_tensors_unpadded

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
                sampling_interval, bssids, beacon_ids, data_tensors = pickle.load(f)
            if self.sampling_interval == sampling_interval:
                self.bssids_ = bssids
                self.beacon_ids_ = beacon_ids
                return data_tensors

        data_frames = self._get_zipped_data(cache=False)

        position_df = data_frames["TYPE_WAYPOINT"]
        position_df = position_df.rename(columns=lambda x: f"pos:{x}")

        # ---- WIFI ----

        wifi_df = data_frames["TYPE_WIFI"]

        def _apply(group):
            bssid = group["bssid"].iloc[0]
            return pd.Series(group["rssi"], name=f"wifi:{bssid}")

        wifi_grouped = wifi_df.groupby("bssid").apply(_apply)

        wifi_timestamps = sorted(wifi_grouped.index.get_level_values(1).unique())
        wifi_split = pd.DataFrame(index=wifi_timestamps)
        wifi_split.index.name = "time"

        self.bssids_ = wifi_grouped.index.get_level_values(0).unique().to_list()
        for bssid in self.bssids_:
            try:
                wifi_split[bssid] = wifi_grouped[bssid]
            except ValueError:
                # Sometimes more than one observation per time
                wifi_split[bssid] = wifi_grouped[bssid][
                    ~wifi_grouped[bssid].index.duplicated()
                ]

        # ---- Beacons ----
        beacon_df = data_frames.get("TYPE_BEACON")

        if beacon_df is not None:

            def _apply(group):
                beacon_id = group["uuid"].iloc[0]
                return pd.Series(group["distance"], name=f"beacon:{beacon_id}")

            beacon_grouped = beacon_df.groupby("uuid").apply(_apply)

        if beacon_df is None:
            self.beacon_ids_ = []
            beacon_split = pd.DataFrame()
            beacon_split.index.name = "time"

        elif (
            hasattr(beacon_grouped, "columns") and beacon_grouped.columns.name == "time"
        ):
            # Fix for only one beacon
            beacon_timestamps = sorted(beacon_grouped.columns)
            beacon_split = pd.DataFrame(index=beacon_timestamps)
            beacon_split.index.name = "time"
            beacon_split[beacon_grouped.index[0]] = beacon_grouped.values.flatten()

            self.beacon_ids_ = [beacon_grouped.index[0]]
        else:
            beacon_timestamps = sorted(
                beacon_grouped.index.get_level_values(1).unique()
            )
            beacon_split = pd.DataFrame(index=beacon_timestamps)
            beacon_split.index.name = "time"

            self.beacon_ids_ = (
                beacon_grouped.index.get_level_values(0).unique().to_list()
            )

            for i, beacon_id in enumerate(self.beacon_ids_):
                try:
                    beacon_split[beacon_id] = beacon_grouped[beacon_id]
                except ValueError:
                    # Sometimes more than one observation per time
                    beacon_split[beacon_id] = beacon_grouped[beacon_id][
                        ~beacon_grouped[beacon_id].index.duplicated()
                    ]

        end_time = max(
            position_df.index[-1],
            wifi_df.index[-1],
            beacon_split.index[-1] if len(beacon_split.index) > 0 else pd.Timedelta(0),
        )
        new_index = pd.timedelta_range(0, end_time, freq=f"{self.sampling_interval}ms")
        new_index.name = "time"

        resampled_data = (
            pd.DataFrame(index=new_index)
            .pipe(pd.merge_ordered, position_df, "time")
            .pipe(pd.merge_ordered, wifi_split, "time")
            .pipe(pd.merge_ordered, beacon_split, "time")
            .bfill(limit=1)
            .set_index("time")
            .loc[new_index]
        )

        time = resampled_data.index.total_seconds()
        position = resampled_data[position_df.columns].values
        wifi = resampled_data[wifi_split.columns].values
        beacon = resampled_data[beacon_split.columns].values

        data_tensors = (
            torch.tensor(time, dtype=torch.float64),
            torch.tensor(position, dtype=torch.float64),
            torch.tensor(wifi, dtype=torch.float64),
            torch.tensor(beacon, dtype=torch.float64),
        )

        cached_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(cached_path, "wb") as f:
            pickle.dump(
                (self.sampling_interval, self.bssids_, self.beacon_ids_, data_tensors),
                f,
            )

        return data_tensors


if __name__ == "__main__":

    site_id = "5a0546857ecc773753327266"
    floor_id = "F1"

    floor_data = FloorDataset(
        site_id,
        floor_id,
        wifi_threshold=200,
        sampling_interval=100,
        validation_percent=0.2,
        test_percent=0.1,
    )

    floor_data[[1, 2, 3, 4]]
