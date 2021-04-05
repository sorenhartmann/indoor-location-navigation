import pickle
import gzip
from pathlib import Path
from functools import cached_property
from dataclasses import dataclass

project_dir = Path(__file__).resolve().parents[2]

raw_path = project_dir / "data" / "raw"
interim_path = project_dir / "data" / "interim"

@dataclass
class SiteDataset:

    site_id : str

    def __post_init__(self) -> None:

        # TODO: Train test split
        site_dir = interim_path / "train" / self.site_id
        assert site_dir.is_dir(), "Site not found"

        floor_ids = [dir_.name for dir_ in site_dir.iterdir() if dir_.is_dir()]

        self.floors = {
            floor_id: FloorDataset(self.site_id, floor_id) for floor_id in floor_ids
        }

        pass


@dataclass
class FloorDataset:

    site_id : str
    floor_id : str

    def __post_init__(self) -> None:

        floor_dir = interim_path / "train" / self.site_id / self.floor_id
        assert floor_dir.is_dir(), "Invalid site- and/or floor-id"

        trace_ids = [
            file_.with_suffix("").stem
            for file_ in floor_dir.iterdir()
            if file_.suffix == ".gz"
        ]

        self.traces = {
            trace_id: TraceData(self.site_id, self.floor_id, trace_id) for trace_id in trace_ids
        }

@dataclass
class TraceData:

    site_id : str
    floor_id : str
    trace_id : str

    def __post_init__(self):

        trace_dir = interim_path / "train" / self.site_id / self.floor_id / f"{self.trace_id}.pkl.gz"
        assert trace_dir.exists(), "Invalid site- and/or floor- and/or trace-id"
        
        self._trace_dir = trace_dir

    @cached_property
    def data(self):
        with gzip.open(self._trace_dir) as f:
            return pickle.load(f)


