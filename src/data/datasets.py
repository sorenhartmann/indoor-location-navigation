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

project_dir = Path(__file__).resolve().parents[2]

raw_path = project_dir / "data" / "raw"
interim_path = project_dir / "data" / "interim"

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

        info_path =  Path("metadata") / self.site_id / self.floor_id / "floor_info.json"

        with zipfile.ZipFile(raw_path / RAW_FILE_NAME) as zip_file:

            file_path = zipfile.Path(zip_file) / info_path
            with file_path.open("r") as f:
                return json.load(f)

    def extract_traces(self):
        for trace in tqdm(self.traces):
            trace.data

@dataclass
class TraceData:

    site_id: str
    floor_id: str
    trace_id: str
        
    @cached_property
    def data(self):

        sub_path = Path("train") / self.site_id / self.floor_id / self.trace_id
        cached_path = (interim_path / sub_path).with_suffix(".pkl.gz")
        
        try:
            with gzip.open(cached_path, "rb") as f:
                data = pickle.load(f)
        except FileNotFoundError:
            data = get_trace_data(sub_path.with_suffix(".txt"))
            cached_path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(cached_path, "wb") as f:
                pickle.dump(data, f)

        return data

if __name__ == "__main__":

    site_data = SiteDataset("5a0546857ecc773753327266")
    site_data.floors[0].traces[0].data
    site_data.floors[0].image
    site_data.floors[0].extract_traces()