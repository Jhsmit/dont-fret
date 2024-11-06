from __future__ import annotations

import asyncio
import dataclasses
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import (
    Callable,
    Dict,
    Optional,
)

import numpy as np
import polars as pl

from dont_fret.config.config import BurstColor
from dont_fret.fileIO import PhotonFile
from dont_fret.models import Bursts, PhotonData
from dont_fret.web.methods import get_duration, get_info
from dont_fret.web.models import BurstNode, PhotonNode


class ThreadedDataManager:
    def __init__(self) -> None:
        self.photon_cache: Dict[uuid.UUID, asyncio.Future[PhotonData]] = {}
        self.burst_cache: Dict[tuple[uuid.UUID, str], asyncio.Future[Bursts]] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)  # todo config wokers
        self.running_jobs = {}

    # todo allow passing loop to init
    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        return asyncio.get_running_loop()

    def run(self, func, *args):
        # TODO typing
        return self.loop.run_in_executor(self.executor, func, *args)

    async def get_photons(self, node: PhotonNode) -> PhotonData:
        if node.id not in self.photon_cache:
            future = self.loop.create_future()
            self.photon_cache[node.id] = future

            try:
                photons = await self.run(PhotonData.from_file, PhotonFile(node.file_path))
                future.set_result(photons)
            except Exception as e:
                self.photon_cache.pop(node.id)
                future.set_exception(e)
                raise

        return await self.photon_cache[node.id]

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

        if key not in self.burst_cache:
            future = self.loop.create_future()
            self.burst_cache[key] = future

            try:
                bursts = await self.search(photon_node, burst_colors)
                future.set_result(bursts)
            except Exception as e:
                self.burst_cache.pop(key)
                future.set_exception(e)
                raise

        return await self.burst_cache[key]

    async def search(self, node: PhotonNode, colors: list[BurstColor]) -> Bursts:
        photon_data = await self.get_photons(node)
        bursts = photon_data.burst_search(colors)

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

        # results = []
        for i, f in enumerate(asyncio.as_completed(tasks)):
            # result = await f
            # results.append(result)
            progress = (i + 1) * (100 / len(tasks))
            on_progress(progress)

        results = await asyncio.gather(*tasks)

        return results

    async def get_dataframe(
        self,
        photon_nodes: list[PhotonNode],
        burst_colors: list[BurstColor],
        on_progress: Optional[Callable[[float | bool], None]] = None,
    ) -> pl.DataFrame:
        on_progress = on_progress or (lambda _: None)
        on_progress(True)

        results = await self.get_bursts_batch(photon_nodes, burst_colors, on_progress)
        on_progress(True)

        names = [ph_node.name for ph_node in photon_nodes]
        lens = [len(burst) for burst in results]

        dtype = pl.Enum(categories=names)
        filenames = pl.Series(name="filename", values=np.repeat(names, lens), dtype=dtype)

        df = pl.concat([b.burst_data for b in results], how="vertical_relaxed").with_columns(
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
