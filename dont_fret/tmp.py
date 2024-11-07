# %%
import uuid
from dataclasses import replace
from functools import reduce
from itertools import chain
from operator import and_
from pathlib import Path
from typing import Callable, Dict, List, Optional, Type, TypedDict, Union

import altair as alt
import numpy as np
import solara
import solara.lab
import yaml

import dont_fret.web.state as state
from dont_fret.config.config import BurstFilterItem, cfg
from dont_fret.web.bursts import BurstPage
from dont_fret.web.bursts.components import BurstFigure, BurstFigureSelection
from dont_fret.web.components import RegexSelectDialog
from dont_fret.web.home import HomePage
from dont_fret.web.main import Page as MainPage
from dont_fret.web.methods import make_burst_nodes
from dont_fret.web.models import BurstNode, FRETNode, ListStore, PhotonNode
from dont_fret.web.trace import TracePage
from dont_fret.web.utils import (
    find_index,
    find_object,
    get_bursts,
    make_selector_nodes,
    wrap_callback,
)

# %%

ROOT = Path(__file__).parent.parent
ROOT
# %%


data = yaml.safe_load((ROOT / "default_testing.yaml").read_text())
cfg.update(data)

style = """
.vega-embed {
    overflow: visible;
    width: 100% !important;
}
"""

pth = ROOT / "tests" / "test_data" / "input" / "ds2"

# %%
pth.exists()
# %%


def on_start():
    photon_nodes = [PhotonNode(file_path=ptu_pth) for ptu_pth in pth.glob("*.ptu")]
    burst_nodes = make_burst_nodes(photon_nodes, cfg.burst_search)

    node_1 = FRETNode(
        name=solara.Reactive("my_node"),
        photons=ListStore(photon_nodes),
        bursts=ListStore(burst_nodes),
    )

    state.fret_nodes.set([node_1])
    state.disable_burst_page.set(False)
    state.disable_trace_page.set(False)


import solara.lifecycle

solara.lifecycle.on_kernel_start(on_start)


@solara.component
def Page():
    solara.Style(style)
    MainPage()


# %%
