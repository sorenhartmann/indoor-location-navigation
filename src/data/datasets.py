from io import BytesIO
import pickle
import gzip
from pathlib import Path
from functools import cached_property
from dataclasses import dataclass, InitVar
import re
from PIL import Image
import json
from src.data.extract_data import RAW_FILE_NAME, get_file_tree, get_trace_data
import zipfile
import pandas as pd
from contextlib import ExitStack
import functools
from tqdm import tqdm
import jax.numpy as jnp

project_dir = Path(__file__).resolve().parents[2]

raw_path = project_dir / "data" / "raw"
interim_path = project_dir / "data" / "interim"
processed_path = project_dir / "data" / "processed"


@dataclass
class TestDataset:
    pass


@dataclass
class SiteDataset:

    site_id: str

    def __post_init__(self) -> None:

        file_tree = get_file_tree()
        floor_ids = file_tree["train"][self.site_id]
        self.floors = [FloorDataset(self.site_id, floor_id) for floor_id in floor_ids]


@dataclass
class FloorDataset:

    site_id: str
    floor_id: str

    def __post_init__(self) -> None:

        file_tree = get_file_tree()
        trace_ids = file_tree["train"][self.site_id][self.floor_id]

        # TODO: Train test split
        self.traces = [
            TraceData(self.site_id, self.floor_id, trace_id) for trace_id in trace_ids
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

    def extract_traces(self):
        for trace in tqdm(self.traces):
            trace.data

    
    def _get_matrices(self, ms=100):

        data = self._get_data(cache=False)

        position = data["TYPE_WAYPOINT"]
        position = position.rename(columns=lambda x: f"pos:{x}")

        wifi = data["TYPE_WIFI"]

        def _apply(group):
            bssid = group["bssid"].iloc[0]
            return pd.Series(group["rssi"], name=f"wifi:{bssid}")

        wifi_split = pd.DataFrame()
        wifi_series = wifi.groupby("bssid").apply(_apply)
        for bssid in wifi_series.index.get_level_values(0).unique():
            wifi_split[bssid] = wifi_series[bssid]

        end_point = max(position.index[-1], wifi.index[-1])
        new_index = pd.timedelta_range(0, end_point, freq=f"{ms}ms")
        new_index.name = "time"
        df = pd.DataFrame(index=new_index)

        resampled_data = (
            df.pipe(pd.merge_ordered, position, "time")
            .pipe(pd.merge_ordered, wifi_split, "time")
            .bfill(limit=1)
            .set_index("time")
            .loc[new_index]
        )

        position = jnp.array(resampled_data[position.columns].values)
        wifi = jnp.array(resampled_data[wifi_split.columns].values)

        return {
            "position" : position,
            "wifi" : wifi,
        }



@dataclass
class TraceData:

    site_id: str
    floor_id: str
    trace_id: str

    @property
    def data(self):
        return self._get_data()

    def _get_data(self, cache=True):

        sub_path = Path("train") / self.site_id / self.floor_id / self.trace_id
        cached_path = (interim_path / sub_path).with_suffix(".pkl.gz")

        try:
            with gzip.open(cached_path, "rb") as f:
                data = pickle.load(f)
        except FileNotFoundError:
            data = get_trace_data(sub_path.with_suffix(".txt"))
            if cache:
                cached_path.parent.mkdir(parents=True, exist_ok=True)
                with gzip.open(cached_path, "wb") as f:
                    pickle.dump(data, f)

        return data

    @property
    def matrices(self):

        sub_path = Path("train") / self.site_id / self.floor_id / self.trace_id
        cached_path = (processed_path / sub_path).with_suffix(".pkl.gz")

        try:
            with gzip.open(cached_path, "rb") as f:
                matrices = pickle.load(f)
        except FileNotFoundError:
            matrices = self._get_matrices()
            cached_path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(cached_path, "wb") as f:
                pickle.dump(matrices, f)

        return matrices

    def _get_matrices(self, ms=100):

        data = self._get_data(cache=False)

        position = data["TYPE_WAYPOINT"]
        position = position.rename(columns=lambda x: f"pos:{x}")

        wifi = data["TYPE_WIFI"]

        def _apply(group):
            bssid = group["bssid"].iloc[0]
            return pd.Series(group["rssi"], name=f"wifi:{bssid}")

        wifi_split = pd.DataFrame()
        wifi_series = wifi.groupby("bssid").apply(_apply)
        for bssid in wifi_series.index.get_level_values(0).unique():
            wifi_split[bssid] = wifi_series[bssid]

        end_point = max(position.index[-1], wifi.index[-1])
        new_index = pd.timedelta_range(0, end_point, freq=f"{ms}ms")
        new_index.name = "time"
        df = pd.DataFrame(index=new_index)

        resampled_data = (
            df.pipe(pd.merge_ordered, position, "time")
            .pipe(pd.merge_ordered, wifi_split, "time")
            .bfill(limit=1)
            .set_index("time")
            .loc[new_index]
        )

        position = resampled_data[position.columns].values
        wifi = resampled_data[wifi_split.columns].values
        time  = resampled_data.index.total_seconds()

        return {
            "position" : position,
            "wifi" : wifi,
            "time" : time,
        }


if __name__ == "__main__":

    site_data = SiteDataset("5a0546857ecc773753327266")
    # site_data.floors[0].traces[0].data
    # site_data.floors[0].image
    # site_data.floors[0].extract_traces()
    site_data.floors[0].traces[0].matrices