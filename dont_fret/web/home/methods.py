import asyncio
from concurrent.futures import ThreadPoolExecutor

import polars as pl
import solara
import solara.lab

import dont_fret.web.state as state
from dont_fret.web.methods import burst_search, get_duration
from dont_fret.web.models import BurstNode, PhotonNode


# todo pass callable to add item on done
@solara.lab.task(prefer_threaded=False)
async def task_burst_search(name: str, photon_nodes: list[PhotonNode], burst_store) -> None:
    # name: name of burst search settings as well as its output name

    def on_progress(progress: float | bool) -> None:
        task_burst_search.progress = progress

    burst_colors = list(state.burst_settings.value[name])
    df = await state.data_manager.get_dataframe(photon_nodes, burst_colors, on_progress)

    if len(df) == 0:
        state.snackbar.warning("No bursts found", timeout=0)
    else:
        # getting info should be fast / non_blocking since photons are cached
        info_list = [await state.data_manager.get_info(ph_node) for ph_node in photon_nodes]
        duration = get_duration(info_list)
        burst_node = BurstNode(
            name=name, df=df, colors=burst_colors, photon_nodes=photon_nodes, duration=duration
        )
        burst_store.append(burst_node)

        state.snackbar.success(
            f"Burst search completed, found {len(burst_node.df)} bursts", timeout=0
        )

    task_burst_search.progress = False


@solara.lab.task(prefer_threaded=False)
async def task_burst_search_depr(name: str, measurement_id: str) -> None:
    # name: name of burst search settings as well as its output name
    task_burst_search.progress = True
    photon_file_items = state.fret_nodes.get_node(measurement_id).photons
    burst_colors = list(state.burst_settings.value[name])

    categories = [item.name for item in photon_file_items]
    dtype = pl.Enum(categories)

    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for f in photon_file_items:
            fut = loop.run_in_executor(executor, burst_search, f, burst_colors, dtype)
            futures.append(fut)

        results = []
        for i, f in enumerate(asyncio.as_completed(futures)):
            results.append(await f)
            task_burst_search.progress = (i + 1) * (100 / len(futures))

    task_burst_search.progress = True

    df = pl.concat(
        results, how="vertical_relaxed"
    )  # this should be very short such that the UI remains responsive
    if len(df) == 0:
        state.snackbar.warning("No bursts found", timeout=0)
    else:
        metadata = {fi.name: fi.get_info() for fi in photon_file_items}
        burst_item = BurstNode(name=name, df=df, metadata=metadata)
        state.fret_nodes.add_burst_items(measurement_id, [burst_item])
        state.snackbar.success(
            f"Burst search completed, found {len(burst_item.df)} bursts", timeout=0
        )

    task_burst_search.progress = False
