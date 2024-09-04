from pathlib import Path

import solara
import solara.lab
import yaml

import dont_fret.web.state as state
from dont_fret.config.config import cfg
from dont_fret.web.dev import load_burst_items, load_file_items
from dont_fret.web.home import HomePage
from dont_fret.web.main import Page as MainPage
from dont_fret.web.models import FRETNode

data = yaml.safe_load(Path("default_testing.yaml").read_text())
cfg.update(data)


# TODO default burst searches from config
DCBS_TEST = {"DD + DA": {"L": 50, "M": 35, "T": 0.0005}, "AA": {"L": 50, "M": 35, "T": 0.0005}}
APBS_TEST = {"DD + DA + AA": {"L": 50, "M": 35, "T": 0.0005}}

pth = "ds2"
photon_file_items = load_file_items(pth)
# burst_items = load_burst_items(pth, suffix=".csv")

fret_nodes = [
    FRETNode(
        name="FRET NOT",
        photons=photon_file_items,
        # bursts=burst_items,
    ),
]

# %%


@solara.component
def Page():
    def preload():
        # state.FRET_NODES.set([])
        if len(state.fret_nodes.value) == 0:
            state.fret_nodes.extend(fret_nodes)

    solara.use_effect(preload, dependencies=[])
    nodes = state.fret_nodes.value

    if len(nodes) != 0:
        with solara.Column(style={"height": "100%"}):
            MainPage()
    else:
        solara.Text("Loading fret nodes...")
