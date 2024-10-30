from __future__ import annotations

import dataclasses
import uuid
from collections import UserList
from dataclasses import dataclass, make_dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple, TypedDict, TypeVar

import numpy as np
import polars as pl
import solara

from dont_fret.config.config import BurstColor, BurstFilterItem

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


@dataclasses.dataclass
class PhotonNode:
    file_path: Path
    info: Optional[dict] = None
    id: uuid.UUID = dataclasses.field(default_factory=lambda: uuid.uuid4())

    @property
    def name(self) -> str:
        return self.file_path.name

    @property
    def size(self) -> int:
        return self.file_path.stat().st_size

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, PhotonNode):
            return False
        return self.id == value.id


@dataclasses.dataclass
class BurstNode:
    name: str
    df: pl.DataFrame
    colors: list[BurstColor] = dataclasses.field(default_factory=list)
    id: uuid.UUID = dataclasses.field(default_factory=lambda: uuid.uuid4())

    photon_nodes: list[PhotonNode] = dataclasses.field(default_factory=list)
    duration: Optional[float] = None


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


T = TypeVar("T")


def reactive_factory(factory: Callable[[], T]) -> solara.Reactive[T]:
    return solara.reactive(factory())


# @dataclasses.dataclass
# class FRETNode:
#     name: str  # displayed name
#     id: str = dataclasses.field(default_factory=lambda: uuid.uuid4().hex)  # unique id
#     description: str = ""  # description of the node
#     photons: list[PhotonFileItem] = dataclasses.field(default_factory=list)
#     bursts: list[BurstItem] = dataclasses.field(default_factory=list)


@dataclass
class SnackbarMessage:
    message: str = ""
    color: str = "primary"
    timeout: int = 3000
    btn_color: str = "text-primary-color"

    show: bool = False
