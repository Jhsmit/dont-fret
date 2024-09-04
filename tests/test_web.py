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
from dont_fret.web.bursts.components import BurstFigure
from dont_fret.web.home.methods import task_burst_search
from dont_fret.web.methods import burst_search, create_file_items
from dont_fret.web.models import BurstItem, FRETNode, PhotonFileItem

cwd = Path(__file__).parent
input_data_dir = cwd / "test_data" / "input"
output_data_dir = cwd / "test_data" / "output"

DCBS_TEST = {"DD + DA": {"L": 50, "M": 35, "T": 0.0005}, "AA": {"L": 50, "M": 35, "T": 0.0005}}
APBS_TEST = {"DD + DA + AA": {"L": 50, "M": 35, "T": 0.0005}}


@pytest.fixture
def ph_ds1() -> PhotonData:
    return PhotonData.from_file(PhotonFile(input_data_dir / "ds1" / "datafile_1.ptu"))


@pytest.fixture
def photon_file_items() -> List[PhotonFileItem]:
    file_items = create_file_items(input_data_dir / "ds1")
    return file_items


@pytest.fixture
def burst_items(photon_file_items: List[PhotonFileItem]) -> List[BurstItem]:
    burst_settings = ["DCBS", "APBS"]
    dtype = pl.Enum([item.name for item in photon_file_items])

    burst_items = []
    for name in burst_settings:
        # TODO there should be a function list[PhotonFileItem] -> BurstItem
        dfs = [burst_search(f, name, dtype) for f in photon_file_items]
        df = pl.concat(dfs, how="vertical_relaxed")

        burst_items.append(BurstItem(name=name, df=df))
    return burst_items


@pytest.mark.asyncio
async def test_burst_search(ph_ds1):
    ph_item = PhotonFileItem(file_path=input_data_dir / "ds1" / "datafile_1.ptu")

    node = FRETNode(
        name="FRET NOT",
        photons=[ph_item],
    )

    state.fret_nodes.set([])
    assert len(state.fret_nodes.value) == 0
    state.fret_nodes.append(node)
    assert len(state.fret_nodes.value) == 1

    await task_burst_search.function("DCBS", node.id)  # type: ignore

    new_node = state.fret_nodes.get_node(node.id)
    assert new_node.bursts
    burst_item = new_node.bursts[0]
    assert burst_item.name == "DCBS"
    assert burst_item.df.shape == (72, 15)
    assert burst_item.df["filename"].unique()[0] == "datafile_1.ptu"

    await asyncio.sleep(0)


def test_burst_figure(photon_file_items, burst_items):
    state.fret_nodes.set([])
    node = FRETNode(
        name="FRET NOT",
        photons=photon_file_items,
        bursts=burst_items,
    )
    state.fret_nodes.append(node)

    fig = BurstFigure(state.fret_nodes, state.filters)
    box, rc = solara.render(fig)

    locator = rc.find(v.Select, label="Burst item")
    assert locator.widget.v_model == 0
    assert locator.widget.items == [{"text": "DCBS", "value": 0}, {"text": "APBS", "value": 1}]

    find_figure = rc.find(go.FigureWidget).wait_for(timeout=5)
    find_figure.assert_single()
    img_test = find_figure.widget.data[0].z  # type: ignore
    img_ref = np.load(output_data_dir / "ds1" / "z_img_dcbs.npy")
    assert np.allclose(img_test, img_ref)

    locator.widget.v_model = 1
    time.sleep(0.5)  # wait for the redraw to start
    find_figure = rc.find(go.FigureWidget).wait_for(timeout=5)
    find_figure.assert_single()
    img_test = find_figure.widget.data[0].z  # type: ignore
    img_ref = np.load(output_data_dir / "ds1" / "z_img_apbs.npy")
    assert np.allclose(img_test, img_ref)
