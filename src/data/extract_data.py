# -*- coding: utf-8 -*-
import gzip
import pickle
import re
import warnings
from itertools import tee
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from tqdm import tqdm

project_dir = Path(__file__).resolve().parents[2]

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
    "TYPE_MAGNETIC_FIELD_UNCALIBRATED": {
        "x_uncalib": float,
        "y_uncalib": float,
        "z_uncalib": float,
        "x_bias": float,
        "y_bias": float,
        "z_bias": float,
        "accuracy": int,
    },
    "TYPE_GYROSCOPE_UNCALIBRATED": {
        "ang_speed_x": float,
        "ang_speed_y": float,
        "ang_speed_z": float,
        "drift_x": float,
        "drift_y": float,
        "drift_z": float,
        "accuracy": int,
    },
    "TYPE_ACCELEROMETER_UNCALIBRATED": {
        "x_uncalib": float,
        "y_uncalib": float,
        "z_uncalib": float,
        "x_bias": float,
        "y_bias": float,
        "z_bias": float,
        "accuracy": int,
    },
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


def load_trace_file(trace_path: Path):

    with open(trace_path) as f:
        start_time = int(re.search("(\d+)", next(f)).group(0))
        raw = pd.read_csv(f, comment="#", header=None)

    observations = fix_newlines(raw[0])

    data = (
        observations.str.split("\t", n=2, expand=True)
        .rename(columns={0: "time", 1: "type", 2: "data"})
        .set_index(["time", "type"])
    )

    data_frames = {}
    for type_, data_frame in data.groupby(level=1):
        data_frames[type_] = extract_data(type_, data_frame, start_time)

    return data_frames


def _process_trace_file(trace_path: Path):

    sub_path = trace_path.relative_to(raw_path)
    new_path = (interim_path / sub_path).with_suffix(".pkl.gz")
    if new_path.exists():
        return  # Dont recreate file

    new_path.parent.mkdir(parents=True, exist_ok=True)

    data_frames = load_trace_file(trace_path)
    with gzip.open(new_path, "wb") as f:
        pickle.dump(data_frames, f)


def iter_train_files():
    for site_dir in (raw_path / "train").iterdir():
        if not site_dir.is_dir():
            continue
        for floor_dir in site_dir.iterdir():
            if not floor_dir.is_dir():
                continue
            for trace_path in floor_dir.iterdir():
                if trace_path.suffix == ".txt":
                    yield trace_path


def iter_test_files():
    for trace_path in (raw_path / "test").iterdir():
        if trace_path.suffix == ".txt":
            yield trace_path


if __name__ == "__main__":

    with Pool() as pool:

        files = list(iter_train_files())
        iter_ = pool.imap_unordered(_process_trace_file, files)

        for result in tqdm(iter_, desc="Extracting training files", total=len(files)):
            pass

        files = list(iter_test_files())
        iter_ = pool.imap_unordered(_process_trace_file, files)

        for result in tqdm(iter_, desc="Extracting test files", total=len(files)):
            pass
