import solara
import solara.lab

import dont_fret.web.state as state
from dont_fret.web.methods import get_duration
from dont_fret.web.models import BurstNode, PhotonNode


# todo pass callable to add item on done
@solara.lab.task(prefer_threaded=False)  # type: ignore
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
