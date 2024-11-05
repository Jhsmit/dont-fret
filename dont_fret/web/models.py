from __future__ import annotations

import dataclasses
import uuid
from collections import UserList
from dataclasses import dataclass, field, make_dataclass
from pathlib import Path
from typing import Callable, ContextManager, Generic, Optional, Tuple, TypedDict, TypeVar

import numpy as np
import polars as pl
import solara
from solara.toestand import merge_state

from dont_fret.config.config import BurstColor, BurstFilterItem

T = TypeVar("T")

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
# todo remove
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


def reactive_factory(factory: Callable[[], T]) -> solara.Reactive[T]:
    return solara.reactive(factory())


@dataclass
class SnackbarMessage:
    message: str = ""
    color: str = "primary"
    timeout: int = 3000
    btn_color: str = "text-primary-color"

    show: bool = False


class _NoDefault:
    """Sentinel class to distinguish between no default and None as default"""

    pass


NO_DEFAULT = _NoDefault()


class ListStore(Generic[T]):
    """baseclass for reactive list"""

    def __init__(self, items: Optional[list[T]] = None):
        self._items = solara.reactive(items if items is not None else [])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> T:
        return self.items[idx]

    def __iter__(self):
        return iter(self.items)

    @property
    def items(self):
        return self._items.value

    def get_item(self, idx: int, default: R = NO_DEFAULT) -> T | R:
        try:
            return self._items.value[idx]
        except IndexError:
            if default is NO_DEFAULT:
                raise IndexError(f"Index {idx} is out of range")
            return default

    def set(self, items: list[T]) -> None:
        self._items.value = items

    def set_item(self, idx: int, item: T) -> None:
        new_items = self._items.value.copy()
        if idx == len(new_items):
            new_items.append(item)
        elif idx < len(new_items):
            new_items[idx] = item
        else:
            raise IndexError(f"Index {idx} is out of range")
        self._items.value = new_items

    def append(self, item: T) -> None:
        self._items.value = [*self._items.value, item]

    def extend(self, items: list[T]) -> None:
        new_value = self.items.copy()
        new_value.extend(items)
        self._items.value = new_value

    def pop(self, idx: int) -> T:
        item = self.items[idx]
        self._items.value = self.items[:idx] + self.items[idx + 1 :]
        return item

    def remove(self, item: T) -> None:
        self._items.value = [it for it in self.items if it != item]

    def update(self, idx: int, **kwargs):
        new_value = self.items.copy()
        updated_item = merge_state(new_value[idx], **kwargs)
        new_value[idx] = updated_item
        self._items.value = new_value

    def index(self, item: T) -> int:
        return self.items.index(item)

    def subscribe(
        self, listener: Callable[[list[T]], None], scope: Optional[ContextManager] = None
    ):
        return self._items.subscribe(listener, scope=scope)

    def subscribe_change(
        self, listener: Callable[[list[T], list[T]], None], scope: Optional[ContextManager] = None
    ):
        return self._items.subscribe_change(listener, scope=scope)


def use_liststore(value: list[T] | ListStore[T]) -> ListStore[T]:
    """use_reactive for liststore"""

    def make_liststore():
        if not isinstance(value, ListStore):
            return ListStore(value)

    store = solara.use_memo(make_liststore, [])
    if isinstance(value, ListStore):
        store = value
    assert store is not None

    return store


@dataclasses.dataclass
class FRETNode:
    name: solara.Reactive[str]  # displayed name
    id: uuid.UUID = dataclasses.field(default_factory=lambda: uuid.uuid4())
    description: solara.Reactive[str] = dataclasses.field(
        default_factory=lambda: solara.reactive("")
    )
    photons: ListStore[PhotonNode] = dataclasses.field(default_factory=lambda: ListStore([]))
    bursts: ListStore[BurstNode] = dataclasses.field(default_factory=lambda: ListStore([]))


class FRETStore(ListStore[FRETNode]):
    def __init__(self, nodes: list[FRETNode]):
        super().__init__(nodes)

    def new_node(self, name: solara.Optional[str] = None) -> None:
        name = name or self.get_default_name()
        node = FRETNode(name=solara.Reactive(name))
        self.append(node)

    def get_node(self, node_id: str) -> FRETNode:
        for node in self.items:
            if node.id == node_id:
                return node

        raise ValueError(f"Node with id {node_id} not found.")

    def get_default_name(self) -> str:
        num_dic = {
            1: "TOO",
            2: "THREE",
            3: "FOUR",
            4: "FIVE",
            5: "SIX",
            6: "SEVEN",
            7: "EIGHT",
            8: "NINE",
            9: "TEN",
        }

        if len(self) == 0:
            return "FRET NOT"
        else:
            return f"FRET {num_dic.get(len(self), len(self) + 1)}"


@dataclass
class SelectorNode:
    value: str
    text: Optional[str] = None
    children: list[SelectorNode] = field(default_factory=list)

    def __post_init__(self):
        if self.text is None:
            self.text = self.value

    @property
    def record(self) -> dict:
        return {"text": self.text, "value": self.value}
