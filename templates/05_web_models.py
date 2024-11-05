# %%
"""template for creating web models and using them to launch solara parts
for interactive use only

"""

from __future__ import annotations

import asyncio
from pathlib import Path

import solara

from dont_fret.config import cfg
from dont_fret.web.bursts.components import BurstFigure
from dont_fret.web.datamanager import ThreadedDataManager

# SyncDataManager,
from dont_fret.web.models import FRETNode, ListStore, PhotonNode

# %%
ROOT = Path(__file__).parent.parent
pth = ROOT / "tests" / "test_data" / "input" / "ds2"
photon_nodes = [PhotonNode(file_path=ptu_pth) for ptu_pth in pth.glob("*.ptu")]

dm = ThreadedDataManager()

# %%


# %%
burst_settings = ["DCBS", "APBS"]
burst_nodes = [
    await dm.get_burst_node(photon_nodes, cfg.burst_search[name], name=name)
    for name in burst_settings
]

node_1 = FRETNode(
    name=solara.Reactive("my_node"),
    photons=ListStore(photon_nodes),
    bursts=ListStore(burst_nodes),
)


# %%

node_2 = FRETNode(
    name=solara.Reactive("my_node_2"),
)
node_2.photons.extend(photon_nodes[2:])
fret_nodes = [node_1, node_2]

fret_nodes

node_1.bursts
# %%

BurstFigure(fret_nodes)


# %%
