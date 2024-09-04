from __future__ import annotations

import dataclasses
import uuid
from collections import UserList
from dataclasses import dataclass, field, make_dataclass
from pathlib import Path
from typing import Optional, Tuple, TypedDict

import numpy as np
import polars as pl

from dont_fret.config.config import BurstColor, BurstFilterItem
from dont_fret.fileIO import PhotonFile
from dont_fret.models import PhotonData

BURST_SCHEMA = {
    "E_app": pl.Float64,
    "S_app": pl.Float64,
    "n_photons": pl.UInt16,
    "time_length": pl.Float64,
    "time_mean": pl.Float64,
    "time_min": pl.Float64,
    "time_max": pl.Float64,
    "n_DD": pl.UInt16,
    "n_DA": pl.UInt16,
    "n_AA": pl.UInt16,
    "n_AD": pl.UInt8,
    "tau_DD": pl.Float64,
    "tau_DA": pl.Float64,
    "tau_AA": pl.Float64,
    "tau_AD": pl.Float64,
}


# refactor: move to methods
def get_info(photons: PhotonData) -> dict:
    info = {}
    info["creation_time"] = photons.metadata["creation_time"]
    info["number_of_photons"] = len(photons)
    info["acquisition_duration"] = photons.metadata["acquisition_duration"]
    info["power_diode"] = photons.metadata["tags"]["UsrPowerDiode"]["value"]

    info["cps"] = photons.cps
    t_max = photons.photon_times.max()
    counts = photons.data["stream"].value_counts(sort=True)
    info["stream_cps"] = {k: v / t_max for k, v in counts.iter_rows()}

    if comment := photons.comment:
        info["comment"] = comment
    return info


@dataclasses.dataclass
class PhotonFileItem:
    file_path: Path
    info: Optional[dict] = None
    photons: Optional[PhotonData] = None
    id: uuid.UUID = dataclasses.field(default_factory=lambda: uuid.uuid4())

    @property
    def name(self) -> str:
        return self.file_path.name

    @property
    def size(self) -> int:
        return self.file_path.stat().st_size

    def get_info(self) -> dict:
        if self.info is None:
            self.get_photons()
            assert self.info is not None
            return self.info
        else:
            return self.info

    def get_photons(self) -> PhotonData:
        """read the file if it hasn't been read yet, otherwise checks for cached rsult"""

        if self.photons is not None:
            return self.photons
        # todo cachhe
        # elif self.id in CHACHE:
        #     return CHACHE[self.id]
        else:
            self.photons = PhotonData.from_file(PhotonFile(self.file_path))
            self.info = get_info(self.photons)
        return self.photons

    def to_cache(self):
        pass

    def from_cache(self):
        pass


# move to config ?
class SearchParams(TypedDict):
    L: int
    M: int
    T: float


# todo maybe don't use this class anymore with the new config settings
class BurstColorList(UserList[BurstColor]):
    @classmethod
    def from_dict(cls, search_spec: dict[str, SearchParams]) -> BurstColorList:
        """Takes a dict of burst settings from cfg.burst_search and returns
        a dict of list of BurstColor instances."""

        colors = [
            BurstColor(streams=[s.strip() for s in streams.split("+")], **search_params)
            for streams, search_params in search_spec.items()
        ]

        return cls(colors)

    def to_dict(self) -> dict[str, SearchParams]:
        """Takes a list of `BurstColor` instnaes and returns a dict of burst
        settings for burst search."""

        spec = {}
        for color in self:
            d = dataclasses.asdict(color)
            streams = " + ".join(d.pop("streams"))
            spec[streams] = d

        return spec


@dataclasses.dataclass
class BurstItem:
    name: str

    df: pl.DataFrame

    selected_files: list[str] = dataclasses.field(default_factory=list)

    search_spec: Optional[dict[str, SearchParams]] = None
    """Burst search settings used to generate the bursts"""

    metadata: dict = field(default_factory=dict)

    id: uuid.UUID = dataclasses.field(default_factory=lambda: uuid.uuid4())

    def __post_init__(self):
        if not self.selected_files:
            self.selected_files = list(self.df["filename"].unique())

    @property
    def duration(self) -> Optional[float]:
        durations = [m.get("acquisition_duration", None) for m in self.metadata.values()]
        if len(set(durations)) == 1:
            return durations[0]
        else:
            return None

    @classmethod
    def from_path(cls, path: Path) -> BurstItem:
        # todo add support for hdf5 files
        if path.suffix == ".csv":
            df = pl.read_csv(path)
        elif path.suffix == ".pq":
            df = pl.read_parquet(path)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")

        # convert the filename column to Enum dtype
        df = df.with_columns(pl.col("filename").cast(pl.Enum(df["filename"].unique().sort())))

        return cls(name=path.stem, df=df)


@dataclasses.dataclass
class BurstPlotSettings:
    # = burst plot settings
    # TODO these state classes should be in a separate file; one per page
    x_name: str = "E_app"
    y_name: str = "S_app"

    x_min: float = 0.0
    x_max: float = 1.0

    y_min: float = 0.0
    y_max: float = 1.0

    z_min: Optional[float] = None
    z_max: Optional[float] = None

    nbinsx: int = 40
    nbinsy: int = 40

    @property
    def xbins(self) -> dict:
        return dict(start=self.x_min, end=self.x_max, size=(self.x_max - self.x_min) / self.nbinsx)

    @property
    def ybins(self) -> dict:
        return dict(start=self.y_min, end=self.y_max, size=(self.y_max - self.y_min) / self.nbinsy)

    @property
    def x_range(self) -> Tuple[float, float]:
        return self.x_min, self.x_max

    @property
    def y_range(self) -> Tuple[float, float]:
        return self.y_min, self.y_max


@dataclasses.dataclass(frozen=True)
class BinnedImage:
    img_data: np.ndarray
    x: np.ndarray
    y: np.ndarray

    @classmethod
    def from_settings(cls, df: pl.DataFrame, settings: BurstPlotSettings) -> BinnedImage:
        x, y = df[settings.x_name], df[settings.y_name]

        x_edges = np.linspace(
            settings.x_min, settings.x_max, num=settings.nbinsx + 1, endpoint=True
        )
        y_edges = np.linspace(
            settings.y_min, settings.y_max, num=settings.nbinsy + 1, endpoint=True
        )

        img_data, *_ = np.histogram2d(x, y, bins=(x_edges, y_edges))  # type: ignore

        img = BinnedImage(
            img_data=img_data,
            x=(x_edges[:-1] + x_edges[1:]) / 2,
            y=(y_edges[:-1] + y_edges[1:]) / 2,
        )

        return img


FrozenBurstFilterItem = make_dataclass(
    "FrozenBurstFilterItem", BurstFilterItem.__dataclass_fields__, frozen=True
)


@dataclasses.dataclass
class TraceSettings:
    t_min: float = 0.0
    """Lower bound of the time window to plot in seconds."""

    t_max: float = 10.0
    """Upper bound of the time window to plot in seconds."""

    t_bin: float = 1.0
    """Binning time in milliseconds."""

    @property
    def num_dpts(self) -> int:
        """Number of datapoints in the currently plotted traces."""
        return int((self.t_max - self.t_min) / (self.t_bin / 1000))


@dataclasses.dataclass
class TCSPCSettings:
    log_y: bool = True


@dataclasses.dataclass
class FRETNode:
    name: str  # displayed name
    id: str = dataclasses.field(default_factory=lambda: uuid.uuid4().hex)  # unique id
    description: str = ""  # description of the node
    photons: list[PhotonFileItem] = dataclasses.field(default_factory=list)
    bursts: list[BurstItem] = dataclasses.field(default_factory=list)


@dataclass
class SnackbarMessage:
    message: str = ""
    color: str = "primary"
    timeout: int = 3000
    btn_color: str = "text-primary-color"

    show: bool = False
