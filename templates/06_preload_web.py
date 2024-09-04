# %%

"""
launch dont-fret web with preloaded data
"""

from __future__ import annotations

from pathlib import Path

import solara

import dont_fret.web.state as state
from dont_fret.web.bursts.components import BurstFigure
from dont_fret.web.main import Page as MainPage
from dont_fret.web.methods import batch_burst_search
from dont_fret.web.models import FRETNode, PhotonFileItem

# %%

ROOT = Path(__file__).parent.parent
pth = ROOT / "tests" / "test_data" / "input" / "ds2"
photon_file_items = [PhotonFileItem(file_path=ptu_pth) for ptu_pth in pth.glob("*.ptu")]

# %%

burst_settings = ["DCBS", "APBS"]
burst_items = [batch_burst_search(photon_file_items, name) for name in burst_settings]
# %%
# %%
node = FRETNode(
    name="FRET NOT",
    photons=photon_file_items,
    bursts=burst_items,
)

# %%


@solara.component
def Page():
    def preload():
        state.fret_nodes.set([])
        if len(state.fret_nodes.value) == 0:
            state.fret_nodes.extend([node])

    solara.use_effect(preload, dependencies=[])
    nodes = state.fret_nodes.value

    if len(nodes) != 0:
        MainPage()
    else:
        solara.Text("Loading fret nodes...")


# %%
