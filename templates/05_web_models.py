# %%

"""template for creating web models and using them to launch solara parts
for interactive use only

"""

from __future__ import annotations

from pathlib import Path

import dont_fret.web.state as state
from dont_fret.web.bursts.components import BurstFigure
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
state.fret_nodes.set([])
node = FRETNode(
    name="FRET NOT",
    photons=photon_file_items,
    bursts=burst_items,
)
state.fret_nodes.append(node)

# %%

BurstFigure(state.fret_nodes, state.filters)

# %%
