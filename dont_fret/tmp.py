from pathlib import Path

import solara
import solara.lab
import yaml

import dont_fret.web.state as state
from dont_fret.config.config import cfg
from dont_fret.web.home import HomePage
from dont_fret.web.main import Page as MainPage
from dont_fret.web.models import PhotonNode
from dont_fret.web.new_models import FRETNode, ListStore

data = yaml.safe_load(Path("default_testing.yaml").read_text())
cfg.update(data)

ROOT = Path(__file__).parent.parent
pth = ROOT / "tests" / "test_data" / "input" / "ds2"
photon_file_items = [PhotonNode(file_path=ptu_pth) for ptu_pth in pth.glob("*.ptu")]

# TODO default burst searches from config
DCBS_TEST = {"DD + DA": {"L": 50, "M": 35, "T": 0.0005}, "AA": {"L": 50, "M": 35, "T": 0.0005}}
APBS_TEST = {"DD + DA + AA": {"L": 50, "M": 35, "T": 0.0005}}

pth = "ds2"
# photon_file_items = []
# burst_items = load_burst_items(pth, suffix=".csv")

fret_nodes = [
    FRETNode(
        name=solara.reactive("FRET TOO"),
        photons=ListStore(photon_file_items),
        # bursts=burst_items,
    ),
]

# %%


@solara.component
def Page():
    def preload():
        # state.FRET_NODES.set([])
        if len(state.fret_nodes) == 0:
            print("len zero")
            state.fret_nodes.extend(fret_nodes)

    solara.use_effect(preload, dependencies=[])
    # nodes = state.fret_nodes.value

    if len(state.fret_nodes) != 0:
        with solara.Column(style={"height": "100%"}):
            MainPage()
    else:
        solara.Text("Loading fret nodes...")
