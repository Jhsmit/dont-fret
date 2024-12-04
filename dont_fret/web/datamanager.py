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

from dont_fret.config.config import BurstColor, DontFRETConfig, cfg
from dont_fret.fileIO import PhotonFile
from dont_fret.models import Bursts, PhotonData
from dont_fret.process import process_photon_data
from dont_fret.web.methods import get_duration, get_info, make_burst_dataframe
from dont_fret.web.models import BurstNode, PhotonNode


class ThreadedDataManager:
    def __init__(self, cfg: DontFRETConfig = cfg) -> None:
        self.photon_cache: Dict[uuid.UUID, asyncio.Future[PhotonData]] = {}
        self.burst_cache: Dict[tuple[uuid.UUID, str], asyncio.Future[Bursts]] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)  # todo config wokers
        self.running_jobs = {}
        self.cfg = cfg

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
                photon_data = await self.get_photons(photon_node)
                bursts = await self.run(
                    process_photon_data,
                    photon_data,
                    burst_colors,
                    self.cfg.aggregations,
                    self.cfg.transforms,
                )

                future.set_result(bursts)
            except Exception as e:
                self.burst_cache.pop(key)
                future.set_exception(e)
                raise

        return await self.burst_cache[key]

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

        for i, f in enumerate(asyncio.as_completed(tasks)):
            await f
            progress = (i + 1) * (100 / len(tasks))
            on_progress(progress)

        results = await asyncio.gather(*tasks)

        return results

    async def alex_2cde(self, photon_node: PhotonNode, bursts: Bursts) -> Bursts:
        photons = await self.get_photons(photon_node)
        new_bursts = self.run(bursts.alex_2cde, photons)
        return await new_bursts

    async def fret_2cde(self, photon_node: PhotonNode, bursts: Bursts) -> Bursts:
        photons = await self.get_photons(photon_node)
        new_bursts = self.run(bursts.fret_2cde, photons)
        return await new_bursts

    async def get_burst_node(
        self,
        photon_nodes: list[PhotonNode],
        burst_colors: list[BurstColor],
        name: str = "",
        on_progress: Optional[Callable[[float | bool], None]] = None,
    ) -> BurstNode:
        bursts = await self.get_bursts_batch(photon_nodes, burst_colors, on_progress=on_progress)
        bursts_df = make_burst_dataframe(bursts, names=[ph_node.name for ph_node in photon_nodes])
        info_list = [await self.get_info(node) for node in photon_nodes]

        duration = get_duration(info_list)
        uu_id = uuid.uuid4()

        node = BurstNode(
            name=name or f"burst_node-{id}",
            df=bursts_df,
            colors=burst_colors,
            photon_nodes=photon_nodes,
            duration=duration,
            id=uu_id,
        )

        return node
