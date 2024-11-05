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
from dont_fret.web.models import BurstNode, ListStore, PhotonNode
from dont_fret.web.trace import TracePage
from dont_fret.web.utils import (
    find_index,
    find_object,
    get_bursts,
    make_selector_nodes,
    wrap_callback,
)

data = yaml.safe_load(Path("default_testing.yaml").read_text())
cfg.update(data)

style = """
.vega-embed {
    overflow: visible;
    width: 100% !important;
}
"""

my_selection = ListStore([])

values = ["a", "b1", "c1", "d3"]

state.disable_burst_page.set(False)
state.disable_trace_page.set(False)


@solara.component
def Page():
    solara.Style(style)
    MainPage()
