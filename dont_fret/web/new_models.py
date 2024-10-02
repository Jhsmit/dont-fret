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
from dont_fret.web.methods import get_duration, get_info
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

    def __getitem__(self, idx: int) -> T:
        return self.items[idx]

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
    ) -> Bursts:
        key = self.burst_key(photon_node, burst_colors)
        bursts = self.burst_cache.get(key)
        if bursts is None:
            bursts = await self.search(photon_node, burst_colors)
            self.burst_cache.set(key, bursts)

        return bursts

    async def get_bursts_batch(
        self,
        photon_nodes: list[PhotonNode],
        burst_colors: list[BurstColor],
        on_progress: Optional[Callable[[float | bool], None]] = None,
    ) -> list[Bursts]:
        on_progress = on_progress or (lambda _: None)

        tasks = []
        for ph_node in photon_nodes:
            task = asyncio.create_task(self.get_bursts(ph_node, burst_colors))
            tasks.append(task)

        results = []
        for i, f in enumerate(asyncio.as_completed(tasks)):
            result = await f
            results.append(result)
            progress = (i + 1) * (100 / len(tasks))
            on_progress(progress)

        return results

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
            await self.get_bursts_batch(todo_nodes, burst_colors, on_progress)
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

    async def get_burst_node(
        self,
        photon_nodes: list[PhotonNode],
        burst_colors: list[BurstColor],
        name: str = "",
        on_progress: Optional[Callable[[float | bool], None]] = None,
    ) -> BurstNode:
        burst_df = await self.get_dataframe(photon_nodes, burst_colors, on_progress=on_progress)
        info_list = [await self.get_info(node) for node in photon_nodes]

        duration = get_duration(info_list)
        uu_id = uuid.uuid4()

        node = BurstNode(
            name=name or f"burst_node-{id}",
            df=burst_df,
            colors=burst_colors,
            photon_nodes=photon_nodes,
            duration=duration,
            id=uu_id,
        )

        return node


import asyncio
import threading
from functools import wraps
from typing import Callable, Optional, Union


class SyncDataManager:
    def __init__(self):
        self._async_manager = ThreadedDataManager()
        self._loop = None
        self._loop_thread = None

    def _ensure_loop(self):
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
            self._loop_thread = threading.Thread(target=self._run_loop, daemon=True)
            self._loop_thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _run_async(self, coro):
        self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    @staticmethod
    def _sync_wrapper(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            return self._run_async(getattr(self._async_manager, func.__name__)(*args, **kwargs))

        return wrapper

    @_sync_wrapper
    def get_photons(self, node: PhotonNode) -> PhotonData:
        pass

    @_sync_wrapper
    def get_info(self, node: PhotonNode) -> dict:
        pass

    @_sync_wrapper
    def get_bursts(self, photon_node: PhotonNode, burst_colors: list[BurstColor]) -> Bursts:
        pass

    @_sync_wrapper
    def get_bursts_batch(
        self,
        photon_nodes: list[PhotonNode],
        burst_colors: list[BurstColor],
        on_progress: Optional[Callable[[float | bool], None]] = None,
    ) -> list[Bursts]:
        pass

    @_sync_wrapper
    def search(self, node: PhotonNode, colors: list[BurstColor]) -> Bursts:
        pass

    @_sync_wrapper
    def get_dataframe(
        self,
        photon_nodes: list[PhotonNode],
        burst_colors: list[BurstColor],
        on_progress: Optional[Callable[[float | bool], None]] = None,
    ) -> "pl.DataFrame":
        pass

    @_sync_wrapper
    def get_burst_node(
        self,
        photon_nodes: list[PhotonNode],
        burst_colors: list[BurstColor],
        name: str = "",
        on_progress: Optional[Callable[[float | bool], None]] = None,
    ) -> "BurstNode":
        pass

    def burst_key(self, node: PhotonNode, burst_colors: list[BurstColor]) -> tuple[uuid.UUID, str]:
        return self._async_manager.burst_key(node, burst_colors)

    def __del__(self):
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._loop_thread:
                self._loop_thread.join()
            self._loop.close()
