# -*- coding: utf-8 -*-
import functools
import gzip
import pickle
import re
import io
import warnings
from itertools import tee
from multiprocessing import Pool
from pathlib import Path

from zipfile import ZipFile, ZipInfo
import zipfile

import pandas as pd
from tqdm import tqdm

RAW_FILE_NAME = "indoor-location-navigation.zip"

project_dir = Path(__file__).resolve().parents[2]
#project_dir = Path("/work3/s164221")

raw_path = project_dir / "data" / "raw"
interim_path = project_dir / "data" / "interim"

# None means ignore field
DATA_DESCRIPTIONS = {
    "TYPE_WAYPOINT": {
        "x": float,
        "y": float,
    },
    "TYPE_ACCELEROMETER": {
        "x": float,
        "y": float,
        "z": float,
        "accuracy": int,
    },
    "TYPE_MAGNETIC_FIELD": {
        "x": float,
        "y": float,
        "z": float,
        "accuracy": int,
    },
    "TYPE_GYROSCOPE": {
        "ang_speed_x": float,
        "ang_speed_y": float,
        "ang_speed_z": float,
        "accuracy": int,
    },
    "TYPE_ROTATION_VECTOR": {
        "a": float,
        "b": float,
        "c": float,
        "accuracy": int,
    },
    "TYPE_MAGNETIC_FIELD_UNCALIBRATED": None,
    "TYPE_GYROSCOPE_UNCALIBRATED": None,
    "TYPE_ACCELEROMETER_UNCALIBRATED": None,
    "TYPE_WIFI": {
        "ssid": str,
        "bssid": str,
        "rssi": float,
        "frequency": int,
        "lastseen_ts": None,
    },
    "TYPE_BEACON": {
        "uuid": str,
        "major": str,
        "minor": str,
        "tx_power": int,
        "rssi": float,
        "distance": float,
        "mac_id": None,
        "lastseen_ts": None,
    },
}


@functools.lru_cache(None)
def get_file_tree():

    cached_file = interim_path / "file_tree.pkl"

    if cached_file.exists():
        with open(cached_file, "rb") as f:
            return(pickle.load(f))

    with zipfile.ZipFile(raw_path / RAW_FILE_NAME) as zip_file:
        file_names = pd.Series([file_.filename for file_ in zip_file.filelist])

    grouped = {}
    for name, values in file_names.str.split("/", n=1, expand=True).groupby(0):
        grouped[name] = values[1]

    train_files = grouped["train"]
    train_files_indexed = train_files.str.split("/", expand=True).set_index([0, 1])

    file_tree = {"train": {}}
    for site in train_files_indexed.index.get_level_values(0).unique():
        file_tree["train"][site] = {}
        for floor in train_files_indexed.loc[site].index.unique():
            file_tree["train"][site][floor] = (
                train_files_indexed[2][site, floor]
                .str.replace(".txt", "", regex=False)
                .tolist()
            )

    cached_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cached_file, "wb") as f:
        pickle.dump(file_tree, f)

    return file_tree


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def fix_newlines(series):
    """
    Fixes missing newlines
    """

    # Finding length of timestamp
    lengths = series[0:5].str.extract("^(\d+)\t", expand=False).str.len()
    ts_length = lengths[0]
    assert (ts_length == lengths).all()

    pattern = f"(?<!^)([0-9]{{{ts_length}}}\\tTYPE_[A-Z_]+)"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        missing_nl = series.str.contains(pattern)

    if missing_nl.any():

        def _sub(string):
            matches = re.finditer(pattern, string)
            start_indices = [0] + [x.start(0) for x in matches] + [-1]
            return [string[a:b] for a, b in pairwise(start_indices)]

        return (
            series[~missing_nl]
            .append(series[missing_nl].map(_sub).explode())
            .reset_index(drop=True)
        )
    else:
        return series


def get_trace_data(trace_path):

    with zipfile.ZipFile(raw_path / RAW_FILE_NAME) as zip_file:
        file_path = zipfile.Path(zip_file) / trace_path

        with file_path.open("r") as f:
            first_line = next(f)
            if type(first_line) is bytes:
                first_line = first_line.decode()
            start_time = int(re.search("(\d+)", first_line).group(0))
            raw = pd.read_csv(f, comment="#", header=None)

    observations = fix_newlines(raw[0])

    data = (
        observations.str.split("\t", n=2, expand=True)
        .rename(columns={0: "time", 1: "type", 2: "data"})
        .set_index(["time", "type"])
    )

    data_frames = {}
    for type_, data_frame in data.groupby(level=1):
        if DATA_DESCRIPTIONS[type_] is not None:
            data_frames[type_] = extract_data(type_, data_frame, start_time)

    return data_frames


def extract_data(type_, data_frame, start_time):
    """
    Extracts data of given type from `data_frame`
    """

    data_frame = data_frame["data"].str.split("\t", expand=True)

    time = data_frame.index.droplevel(1).astype(int)
    data_frame.index = pd.to_timedelta(time - start_time, unit="ms")

    data_description = DATA_DESCRIPTIONS[type_]
    keys = list(data_description.keys())

    n_cols = len(data_frame.columns)
    found_keys = keys[:n_cols]

    data_frame.columns = found_keys

    drop_keys = [key for key in found_keys if data_description[key] is None]
    keep_keys = [key for key in found_keys if data_description[key] is not None]

    data_frame = data_frame.drop(columns=drop_keys)
    data_frame = data_frame.astype({key: data_description[key] for key in keep_keys})

    return data_frame