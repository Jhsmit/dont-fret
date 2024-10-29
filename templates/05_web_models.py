# %%
"""template for creating web models and using them to launch solara parts
for interactive use only

"""

from __future__ import annotations

import dataclasses
import threading
import time
import uuid
from pathlib import Path
from typing import Literal, Optional, TypeVar

import numpy as np
import polars as pl
import solara

import dont_fret.web.state as state
from dont_fret.config import cfg
from dont_fret.config.config import BurstColor
from dont_fret.web.bursts.components import BurstFigure
from dont_fret.web.methods import batch_burst_search, get_duration
from dont_fret.web.models import BurstNode, PhotonNode
from dont_fret.web.new_models import (
    FRETNode,
    FRETStore,
    ListStore,
    SelectorNode,
    SyncDataManager,
    ThreadedDataManager,
)
from dont_fret.web.utils import find_object

# %%
ROOT = Path(__file__).parent.parent
pth = ROOT / "tests" / "test_data" / "input" / "ds2"
photon_nodes = [PhotonNode(file_path=ptu_pth) for ptu_pth in pth.glob("*.ptu")]

data_manager = ThreadedDataManager()
sync_manager = SyncDataManager()

fret_store = FRETStore([])

# %%
node_1 = FRETNode(
    name=solara.Reactive("my_node"),
)

burst_settings = ["DCBS", "APBS"]
nodes = [
    sync_manager.get_burst_node(photon_nodes, cfg.burst_search[name], name=name)
    for name in burst_settings
]

node_1.bursts.extend(nodes)
node_1.photons.extend(photon_nodes)
fret_store.append(node_1)


# %%

node_2 = FRETNode(
    name=solara.Reactive("my_node_2"),
)
node_2.photons.extend(photon_nodes[2:])
fret_store.append(node_2)
# %%


fret_node = node_1

# %%


df = fret_node.bursts.items[0].df

# %%

source = df

import altair as alt

chart = (
    alt.Chart(df)
    .mark_bar()
    .encode(
        alt.X(
            "n_photons:Q",
            bin=alt.Bin(step=22.3, nice=False),
            axis=alt.Axis(),  # This removes the bin-aligned ticks
        ),
        y="count()",
    )
)
spec = chart.to_dict()
spec["encoding"]


# %%
def fd_bin_width(data):
    """
    Calculate bin width using the Freedman-Diaconis rule:
    bin width = 2 * IQR * n^(-1/3)
    where IQR is the interquartile range and n is the number of observations
    """
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    n = len(data)
    return 2 * iqr * (n ** (-1 / 3))


# %%
field = "nanotimes_AA"
fd_bin_width(df[field].drop_nans())

# %%

filtered_df = df.filter(pl.col(field).is_null())
filtered_df

# %%


# %%

q75, q25 = np.percentile(data, [75, 25])
iqr = q75 - q25
iqr

# %%
np.any(np.isnan(np.asarray(data)))
# %%


# %%

f_item = state.filters.items[0]
f_item.as_expr()

# %%

df_f = df.filter(f_item.as_expr())
len(df), len(df_f)

# %%
field = "nanotimes_AA"
selection = alt.selection_interval(name="range", encodings=["x"])


def make_chart(df, color="blue", opacity=1.0):
    chart = (
        alt.Chart(df.select(pl.col(field)))
        .mark_rect(opacity=opacity)
        .transform_bin(as_=["x", "x2"], field=field, bin=alt.Bin(step=fd_bin_width(df[field])))
        .encode(
            x=alt.X(
                "x:Q",
                scale={"zero": False},
                title=field,
            ),
            x2="x2:Q",
            y=alt.Y("count():Q", title="count"),
            tooltip=[
                alt.Tooltip("x:Q", title="Center", format=".2f"),
                alt.Tooltip("x2:Q", title="Start", format=".2f"),
                alt.Tooltip("bin_center:Q", title="End", format=".2f"),
                alt.Tooltip("count():Q", title="Count", format=","),
            ],
            color=alt.value(color),
        )
        .transform_calculate(
            bin_center="(datum.x + datum.x2) / 2"  # Calculate bin center
        )
        # .add_params(selection)
    )
    return chart


chart1 = make_chart(df, color="#1f77b4", opacity=0.5)
chart1

chart2 = make_chart(df_f, color="#1f77b4").add_params(selection)

chart = chart1 + chart2
chart

chart
# %%
type(df.select(pl.col(field)))

type(df[field])

# %%

jchart = alt.JupyterChart(chart, embed_options={"actions": False})
jchart

# %%


def on_select(value):
    print(value)


jchart.selections.observe(on_select, "range")


# %%
None.value

# %%

spec.keys()
r_spec = spec.copy()
r_spec.pop("datasets", None)

r_spec
# %%
jchart = alt.JupyterChart(chart, spec=spec, embed_options={"actions": False})
jchart

jchart
