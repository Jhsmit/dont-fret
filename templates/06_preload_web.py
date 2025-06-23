# %%

"""
launch dont-fret web with preloaded data
"""

from __future__ import annotations

from pathlib import Path

import solara

from dont_fret.__main__ import load_config
from dont_fret.web.methods import make_burst_nodes
import dont_fret.web.state as state
from dont_fret.web.main import Page as MainPage
from dont_fret.web.models import FRETNode, ListStore, PhotonNode
from dont_fret.config import cfg


# %%
# optionally change config
load_config(Path("default_testing.yaml"))

# %%

ROOT = Path(__file__).parent.parent
pth = ROOT / "tests" / "test_data" / "input" / "ds2"
photon_nodes = [PhotonNode(file_path=ptu_pth) for ptu_pth in pth.glob("*.ptu")]
burst_nodes = make_burst_nodes(photon_nodes, cfg.burst_search)


# %%
node_1 = FRETNode(
    name=solara.Reactive("my_node"),
    photons=ListStore(photon_nodes),
    bursts=ListStore(burst_nodes),
)

# %%


@solara.component
def Page():
    counter = solara.use_reactive(0)

    def preload():
        state.fret_nodes.set([])
        if len(state.fret_nodes) == 0:
            state.fret_nodes.extend([node_1])
        state.disable_burst_page.set(False)
        state.disable_trace_page.set(False)

    solara.use_effect(preload, dependencies=[counter.value])

    with solara.AppBar():
        solara.Button("Reload", on_click=lambda: counter.set(counter.get() + 1))

    if len(state.fret_nodes) != 0:
        MainPage()
    else:
        solara.Text("Loading fret nodes...")


# %%
