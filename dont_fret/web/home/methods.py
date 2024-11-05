import solara
import solara.lab

import dont_fret.web.state as state
from dont_fret.web.methods import get_duration
from dont_fret.web.models import BurstNode, PhotonNode
from dont_fret.web.utils import has_bursts


# todo pass callable to add item on done
# todo pass callable to add item on done
@solara.lab.task(prefer_threaded=False)  # type: ignore
async def task_burst_search(name: str, photon_nodes: list[PhotonNode], burst_store) -> None:
    # name: name of burst search settings as well as its output name

    def on_progress(progress: float | bool) -> None:
        task_burst_search.progress = progress

    burst_colors = list(state.burst_settings.value[name])

    try:
        burst_node = await state.data_manager.get_burst_node(
            photon_nodes, burst_colors, name, on_progress
        )
    except ValueError:
        state.snackbar.warning("No bursts found", timeout=0)
        task_burst_search.progress = False
        return

    burst_store.append(burst_node)
    state.disable_burst_page.set(not has_bursts(state.fret_nodes.items))

    state.snackbar.success(f"Burst search completed, found {len(burst_node.df)} bursts", timeout=0)

    task_burst_search.progress = False
