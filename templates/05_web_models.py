# %%
"""template for creating web models and using them to launch solara parts
for interactive use only

"""

from __future__ import annotations

from pathlib import Path

import solara

from dont_fret.config import cfg
from dont_fret.web.bursts.components import BurstFigure

# SyncDataManager,
from dont_fret.web.methods import make_burst_nodes
from dont_fret.web.models import FRETNode, ListStore, PhotonNode

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

node_2 = FRETNode(
    name=solara.Reactive("my_node_2"),
)
node_2.photons.extend(photon_nodes[2:])
fret_nodes = [node_1, node_2]
fret_nodes
# %%

BurstFigure(fret_nodes)

# %%
