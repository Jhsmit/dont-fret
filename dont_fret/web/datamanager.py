from __future__ import annotations

import asyncio
import dataclasses
import json
import threading
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import (
    Callable,
    Generic,
    Optional,
    ParamSpec,
    TypeVar,
)

import numpy as np
import polars as pl

from dont_fret.config.config import BurstColor
from dont_fret.fileIO import PhotonFile
from dont_fret.models import Bursts, PhotonData
from dont_fret.web.methods import get_duration, get_info
from dont_fret.web.models import BurstNode, PhotonNode

T = TypeVar("T")
R = TypeVar("R")


class DaskDataManager:
    pass


K = TypeVar("K")
V = TypeVar("V")


# TODO make async, lock per key
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


# altnernative for threadpool
# however, we do need control over the number of workers

# Current implementation with ThreadPoolExecutor
# import asyncio
# from concurrent.futures import ThreadPoolExecutor

# class ThreadedDataManager:
#     def __init__(self, max_workers: int = 4) -> None:
#         self.executor = ThreadPoolExecutor(max_workers=max_workers)
#         # ... other initializations ...

#     def run(self, func, *args):
#         return self.loop.run_in_executor(self.executor, func, *args)

#     async def get_photons(self, node: PhotonNode) -> PhotonData:
#         # ... (lock handling code) ...
#         photons = await self.run(PhotonData.from_file, PhotonFile(node.file_path))
#         # ... (cache update code) ...
#         return photons

# # Equivalent implementation using threading.Thread
# import threading

# class ThreadBasedDataManager:
#     def __init__(self) -> None:
#         self.active_threads = set()
#         # ... other initializations ...

#     def run(self, func, *args):
#         future = asyncio.Future()

#         def thread_func():
#             try:
#                 result = func(*args)
#                 self.loop.call_soon_threadsafe(future.set_result, result)
#             except Exception as e:
#                 self.loop.call_soon_threadsafe(future.set_exception, e)
#             finally:
#                 self.active_threads.remove(thread)

#         thread = threading.Thread(target=thread_func)
#         self.active_threads.add(thread)
#         thread.start()
#         return future

#     async def get_photons(self, node: PhotonNode) -> PhotonData:
#         # ... (lock handling code) ...
#         photons = await self.run(PhotonData.from_file, PhotonFile(node.file_path))
#         # ... (cache update code) ...
#         return photons

#     def __del__(self):
#         for thread in self.active_threads:
#             thread.join()

# per-key lock will also solve the problem of two threads asking for the same photon item object
# claude suggestion for cache:
# class AsyncCache(Generic[K, V]):
#     def __init__(self):
#         self._data: Dict[K, V] = {}
#         self._locks: Dict[K, asyncio.Lock] = {}

#     async def get(self, key: K) -> V | None:
#         if key not in self._locks:
#             self._locks[key] = asyncio.Lock()

#         async with self._locks[key]:
#             return self._data.get(key)

#     async def set(self, key: K, value: V) -> None:
#         if key not in self._locks:
#             self._locks[key] = asyncio.Lock()

#         async with self._locks[key]:
#             self._data[key] = value

# claude suggestiosn for locks on photon items:
# import asyncio
# import uuid
# from concurrent.futures import ThreadPoolExecutor
# from typing import Dict

# from your_module import Cache, PhotonData, PhotonNode, PhotonFile

# class ThreadedDataManager:
#     def __init__(self, max_workers: int = 4) -> None:
#         self.photon_cache = Cache[uuid.UUID, PhotonData]()
#         self.burst_cache = Cache[tuple[uuid.UUID, str], Bursts]()
#         self.executor = ThreadPoolExecutor(max_workers=max_workers)
#         self._loop = None
#         self.photon_locks: Dict[uuid.UUID, asyncio.Lock] = {}

#     @property
#     def loop(self) -> asyncio.AbstractEventLoop:
#         if self._loop is None:
#             self._loop = asyncio.get_event_loop()
#         return self._loop

#     def run(self, func, *args):
#         return self.loop.run_in_executor(self.executor, func, *args)

#     async def get_photons(self, node: PhotonNode) -> PhotonData:
#         if node.id not in self.photon_locks:
#             self.photon_locks[node.id] = asyncio.Lock()

#         async with self.photon_locks[node.id]:
#             photons = self.photon_cache.get(node.id)
#             if photons is None:
#                 photons = await self.run(PhotonData.from_file, PhotonFile(node.file_path))
#                 self.photon_cache.set(node.id, photons)

#         return photons

# ... (rest of the class remains the same)

# testing:
# import asyncio

# class SyncDataManager:
#     def __init__(self):
#         self.async_manager = AsyncDataManager()
#         self.loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(self.loop)

#     def __getattr__(self, name):
#         async_attr = getattr(self.async_manager, name)
#         if callable(async_attr):
#             def sync_wrapper(*args, **kwargs):
#                 return self.loop.run_until_complete(async_attr(*args, **kwargs))
#             return sync_wrapper
#         return async_attr

#     def __del__(self):
#         self.loop.close()

# # Usage in tests:
# def test_get_photons():
#     manager = SyncDataManager()
#     node = PhotonNode(id=uuid.uuid4(), file_path="test_file.photon")
#     photons = manager.get_photons(node)
#     assert isinstance(photons, PhotonData)
#     # ... more assertions

# move functional approach:
# import asyncio
# import uuid
# from typing import Dict, Callable, Any

# # Type aliases for clarity
# PhotonCache = Dict[uuid.UUID, PhotonData]
# BurstCache = Dict[tuple[uuid.UUID, str], Bursts]

# async def get_photons(photon_cache: PhotonCache, node: PhotonNode) -> PhotonData:
#     if node.id not in photon_cache:
#         photons = await PhotonData.from_file_async(PhotonFile(node.file_path))
#         photon_cache[node.id] = photons
#     return photon_cache[node.id]

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
        # TODO if another thread is already generating these photons it should wait for
        # that task to finish
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
from typing import Callable, Optional


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