import asyncio
import copy
import dataclasses
import json
import threading
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Generic, Optional, ParamSpec, Type, TypeVar

import numpy as np
import polars as pl
import solara
from attrs import define, field

from dont_fret.config import cfg
from dont_fret.config.config import BurstColor
from dont_fret.fileIO import PhotonFile
from dont_fret.models import Bursts, PhotonData
from dont_fret.web.methods import get_info
from dont_fret.web.models import BurstColorList, BurstNode, PhotonNode

if TYPE_CHECKING:
    from dont_fret.web.reactive import BurstSettingsReactive, ReactiveFRETNodes, SnackbarReactive

T = TypeVar("T")


class ListStore(Generic[T]):
    """baseclass for reactive list"""

    def __init__(self, items: list[T]):
        self._items = solara.reactive(items)

    def __len__(self):
        return len(self.items)

    @property
    def items(self):
        return self._items.value

    def set(self, items: list[T]) -> None:
        self._items.value = items

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
        self._items.value = [node for node in self.items if item != item]


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


class DaskDataManager:
    pass


K = TypeVar("K")
V = TypeVar("V")


class Cache(Generic[K, V]):
    """superclass for caches"""

    def __init__(self):
        self._cache: OrderedDict[K, V] = OrderedDict()
        self.lock = threading.Lock()

    def __contains__(self, key: K) -> bool:
        with self.lock:
            return key in self._cache

    def __getitem__(self, key: K) -> V:
        with self.lock:
            return self._cache[key]

    def get(self, key: K) -> Optional[V]:
        with self.lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            return None

    def set(self, key: K, value: V) -> None:
        with self.lock:
            self._cache[key] = value
            self._cache.move_to_end(key)  # in case it already existed

    def clear(self) -> None:
        with self.lock:
            self._cache.clear()


# class PhotonCache:
#     """responsible for loading a caching photon data"""

#     def __init__(self):
#         self._cache: OrderedDict[uuid.UUID, PhotonData] = OrderedDict()
#         self.lock = threading.Lock()

#     def __contains__(self, node: PhotonNode) -> bool:
#         # or should we check for node.id?
#         with self.lock:
#             return node.id in self._cache

#     def get(self, key: uuid.UUID) -> Optional[PhotonData]:
#         with self.lock:
#             if key in self._cache:
#                 self._cache.move_to_end(key)
#                 return self._cache[key]
#             return None

#     # TODO: this should be decoupled from the cache object
#     # burst / photon cache should be the same object
#     def get_photons(self, node: PhotonNode) -> PhotonData:
#         photons = self.get(node.id)
#         if photons is None:
#             photons = PhotonData.from_file(PhotonFile(node.file_path))
#             self.set(node.id, photons)

#         return photons

#     def get_info(self, node: PhotonNode) -> dict:
#         if node.info is not None:
#             return node.info
#         else:
#             photons = self.get_photons(node)
#             info = get_info(photons)
#             node.info = info
#             return info


# class BurstCache:
#     def __init__(self):
#         pass
#         self.lock = threading.Lock()

#         self._cache: OrderedDict[tuple[uuid.UUID, str], Bursts] = OrderedDict()

#     # todo perhaps get/set should be in the form of:
#     # get(node, burst_colors) -> Bursts
#     def get(self, key: tuple[uuid.UUID, str]) -> Optional[Bursts]:
#         with self.lock:
#             if key in self._cache:
#                 self._cache.move_to_end(key)
#                 return self._cache[key]
#             return None

#     def set(self, key: tuple[uuid.UUID, str], value: Bursts) -> None:
#         with self.lock:
#             self._cache[key] = value
#             self._cache.move_to_end(key)  # in case it already existed

#     def clear(self) -> None:
#         with self.lock:
#             self._cache.clear()

#     # TODO the key depends on color order while burst search (should) not depend on order
#     @staticmethod
#     def make_key(photon_node: PhotonNode, burst_colors: list[BurstColor]) -> tuple[uuid.UUID, str]:
#         s = json.dumps([dataclasses.asdict(c) for c in burst_colors])
#         return (photon_node.id, s)


T = TypeVar("T")
P = ParamSpec("P")


class ThreadedDataManager:
    def __init__(self) -> None:
        self.photon_cache = Cache[uuid.UUID, PhotonData]()
        self.burst_cache = Cache[tuple[uuid.UUID, str], Bursts]()
        self.executor = ThreadPoolExecutor(max_workers=4)  # todo config wokers

    # todo allow passing loop to init
    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        return asyncio.get_running_loop()

    def run(self, func, *args):
        # TODO typing / implement
        return self.loop.run_in_executor(self.executor, func, *args)

    async def get_photons(self, node: PhotonNode) -> PhotonData:
        photons = self.photon_cache.get(node.id)
        if photons is None:
            photons = await self.run(PhotonData.from_file, PhotonFile(node.file_path))
            self.photon_cache.set(node.id, photons)

        return photons

    async def get_info(self, node: PhotonNode) -> dict:
        if node.info is not None:
            return node.info
        else:
            photons = await self.get_photons(node)
            info = get_info(photons)
            node.info = info
            return info

    @staticmethod
    def burst_key(node: PhotonNode, burst_colors: list[BurstColor]) -> tuple[uuid.UUID, str]:
        s = json.dumps([dataclasses.asdict(c) for c in burst_colors])
        return (node.id, s)

    async def get_bursts(
        self,
        photon_node: PhotonNode,
        burst_colors: list[BurstColor],
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> Bursts:
        loop = loop or asyncio.get_running_loop()

        key = self.burst_key(photon_node, burst_colors)
        bursts = self.burst_cache.get(key)
        if bursts is None:
            bursts = await self.search(photon_node, burst_colors)
            self.burst_cache.set(key, bursts)

        return bursts

    async def burst_search(
        self,
        photon_nodes: list[PhotonNode],
        burst_colors: list[BurstColor],
        on_progress: Optional[Callable[[float | bool], None]] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        loop = loop or asyncio.get_running_loop()
        on_progress = on_progress or (lambda _: None)

        # TODO we are checking todo / done twice, get_dataframe and here
        all_keys = [self.burst_key(ph_node, burst_colors) for ph_node in photon_nodes]
        todo_keys = [k for k in all_keys if k not in self.burst_cache]

        tasks = []
        for k in todo_keys:
            idx = all_keys.index(k)
            ph_node = photon_nodes[idx]
            task = asyncio.create_task(self.search(ph_node, burst_colors))
            tasks.append(task)

        for i, f in enumerate(asyncio.as_completed(tasks)):
            await f
            progress = (i + 1) * (100 / len(tasks))
            on_progress(progress)

    async def search(self, node: PhotonNode, colors: list[BurstColor]) -> Bursts:
        """performs burst search and stores the result in the burst cache"""
        photon_data = await self.get_photons(node)
        bursts = photon_data.burst_search(colors)

        self.burst_cache.set(self.burst_key(node, colors), bursts)
        return bursts

    # then get the results and concatenate
    async def get_dataframe(
        self,
        photon_nodes: list[PhotonNode],
        burst_colors: list[BurstColor],
        on_progress: Optional[Callable[[float | bool], None]] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> pl.DataFrame:
        on_progress = on_progress or (lambda _: None)
        on_progress(True)

        # check for missing bursts, do search for those
        # TODO we dont need to check for missing keys, we can just use get_bursts which will try
        # to find it in the cache
        all_keys = [self.burst_key(ph_node, burst_colors) for ph_node in photon_nodes]
        todo_keys = [k for k in all_keys if k not in self.burst_cache]
        todo_nodes = [photon_nodes[all_keys.index(k)] for k in todo_keys]

        if todo_nodes:
            await self.burst_search(todo_nodes, burst_colors, on_progress, loop)
        on_progress(True)

        # gather all bursts, combine into a dataframe
        all_bursts = [self.burst_cache[key] for key in all_keys]
        assert not any(burst is None for burst in all_bursts), "some burst data is missing"

        names = [ph_node.name for ph_node in photon_nodes]
        lens = [len(burst) for burst in all_bursts]

        dtype = pl.Enum(categories=names)
        filenames = pl.Series(name="filename", values=np.repeat(names, lens), dtype=dtype)

        df = pl.concat([b.burst_data for b in all_bursts], how="vertical_relaxed").with_columns(
            filenames
        )

        return df
