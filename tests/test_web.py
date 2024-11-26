import asyncio
import time
from pathlib import Path
from typing import List

import ipyvuetify as v
import numpy as np
import plotly.graph_objects as go
import polars as pl
import pytest
import solara

import dont_fret.web.state as state
from dont_fret import PhotonData, PhotonFile
from dont_fret.config import cfg
from dont_fret.web.bursts.components import BurstFigure
from dont_fret.web.datamanager import ThreadedDataManager
from dont_fret.web.home.methods import task_burst_search

# from dont_fret.web.methods import burst_search, create_file_items
from dont_fret.web.models import BurstNode, FRETNode, ListStore, PhotonNode

cwd = Path(__file__).parent
input_data_dir = cwd / "test_data" / "input"
output_data_dir = cwd / "test_data" / "output"

DCBS_TEST = {"DD + DA": {"L": 50, "M": 35, "T": 0.0005}, "AA": {"L": 50, "M": 35, "T": 0.0005}}
APBS_TEST = {"DD + DA + AA": {"L": 50, "M": 35, "T": 0.0005}}


@pytest.fixture
def ph_ds1() -> PhotonData:
    return PhotonData.from_file(PhotonFile(input_data_dir / "ds1" / "datafile_1.ptu"))


@pytest.fixture
def photon_nodes() -> List[PhotonNode]:
    pth = input_data_dir / "ds1"
    return [PhotonNode(file_path=ptu_pth) for ptu_pth in pth.glob("*.ptu")]


@pytest.fixture
def burst_nodes(photon_nodes: List[PhotonNode]) -> List[BurstNode]:
    dm = ThreadedDataManager()
    burst_settings = ["DCBS", "APBS"]

    burst_nodes = []
    for name in burst_settings:
        node = asyncio.run(dm.get_burst_node(photon_nodes, cfg.burst_search[name], name=name))
        burst_nodes.append(node)

    return burst_nodes


@pytest.mark.asyncio
async def test_burst_search():
    ph_item = PhotonNode(file_path=input_data_dir / "ds1" / "datafile_1.ptu")

    node = FRETNode(
        name=solara.Reactive("FRET NOT"),
        photons=ListStore([ph_item]),
    )

    state.fret_nodes.set([])
    assert len(state.fret_nodes) == 0
    state.fret_nodes.append(node)
    assert len(state.fret_nodes) == 1

    await task_burst_search.function("DCBS", [ph_item], node.bursts)  # type: ignore

    new_node = state.fret_nodes.get_node(node.id)
    assert new_node.bursts
    burst_item = new_node.bursts[0]
    assert burst_item.name == "DCBS"
    assert burst_item.df.shape == (72, 24)
    assert burst_item.df["filename"].unique()[0] == "datafile_1.ptu"

    await asyncio.sleep(0)


def test_burst_figure(photon_nodes, burst_nodes):
    node = FRETNode(
        name=solara.Reactive("FRET NOT"),
        photons=ListStore(photon_nodes),
        bursts=ListStore(burst_nodes),
    )

    fig = BurstFigure([node])
    box, rc = solara.render(fig)

    locator = rc.find(v.Select, label="Bursts")
    locator.wait_for(timeout=3)
    assert locator.widget.v_model == burst_nodes[0].id.hex
    assert locator.widget.items == [
        {"text": "DCBS", "value": burst_nodes[0].id.hex},
        {"text": "APBS", "value": burst_nodes[1].id.hex},
    ]

    find_figure = rc.find(go.FigureWidget).wait_for(timeout=5)
    find_figure.assert_single()
    img_test = find_figure.widget.data[0].z  # type: ignore
    img_ref = np.load(output_data_dir / "ds1" / "z_img_dcbs.npy")
    assert np.allclose(img_test, img_ref)

    locator.widget.v_model = burst_nodes[1].id.hex
    time.sleep(0.5)  # wait for the redraw to start
    find_figure = rc.find(go.FigureWidget).wait_for(timeout=5)
    find_figure.assert_single()
    img_test = find_figure.widget.data[0].z  # type: ignore
    img_ref = np.load(output_data_dir / "ds1" / "z_img_apbs.npy")
    assert np.allclose(img_test, img_ref)
