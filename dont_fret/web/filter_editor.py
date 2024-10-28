from dataclasses import replace
from functools import reduce
from itertools import chain
from operator import and_
from pathlib import Path
from typing import Callable, Dict, List, Optional, Type, Union

import altair as alt
import numpy as np
import solara
import solara.lab
import yaml

import dont_fret.web.state as state
from dont_fret.config.config import BurstFilterItem, cfg
from dont_fret.web.bursts import BurstPage
from dont_fret.web.bursts.components import BurstFigure, BurstFigureSelection
from dont_fret.web.home import HomePage
from dont_fret.web.main import Page as MainPage
from dont_fret.web.models import BurstNode, PhotonNode
from dont_fret.web.new_models import FRETNode, ListStore, SyncDataManager
from dont_fret.web.trace import TracePage
from dont_fret.web.utils import find_index

data = yaml.safe_load(Path("default_testing.yaml").read_text())
cfg.update(data)

sync_manager = SyncDataManager()


ROOT = Path(__file__).parent.parent
pth = ROOT / "tests" / "test_data" / "input" / "ds2"
photon_file_items = [PhotonNode(file_path=ptu_pth) for ptu_pth in pth.glob("*.ptu")]

# TODO default burst searches from config
DCBS_TEST = {"DD + DA": {"L": 50, "M": 35, "T": 0.0005}, "AA": {"L": 50, "M": 35, "T": 0.0005}}
APBS_TEST = {"DD + DA + AA": {"L": 50, "M": 35, "T": 0.0005}}

pth = "ds2"

burst_settings = ["DCBS", "APBS"]
bursts = []
for setting in burst_settings:
    colors = cfg.burst_search[setting]
    burst_node = sync_manager.get_burst_node(photon_file_items, colors, name=setting)
    bursts.append(burst_node)


node_1 = FRETNode(
    name=solara.reactive("FRET TOO"),
    photons=ListStore(photon_file_items),
    bursts=ListStore(bursts),
)

# %%

node_2 = FRETNode(
    name=solara.reactive("TWO"),
    photons=ListStore(photon_file_items[2:]),
)

nodes = [node_1, node_2]

df = node_1.bursts.items[0].df

# %%

filters = state.filters.items
active_filters = ListStore([f.name for f in filters])

# %%
import polars as pl


def fd_bin_width(data: pl.Series):
    """
    Calculate bin width using the Freedman-Diaconis rule:
    bin width = 2 * IQR * n^(-1/3)
    where IQR is the interquartile range and n is the number of observations
    """
    q75, q25 = data.quantile(0.75), data.quantile(0.25)
    iqr = q75 - q25
    n = len(data)
    return 2 * iqr * (n ** (-1 / 3))


def make_chart(df: pl.DataFrame, field: str, opacity: float = 1.0):
    chart = (
        alt.Chart(df.select(pl.col(field)))
        .mark_rect(opacity=opacity)
        .transform_bin(
            as_=["x", "x2"],
            field=field,
            bin=alt.Bin(step=fd_bin_width(df[field])),
        )
        .encode(
            x=alt.X("x:Q", scale={"zero": False}, title=field),
            x2="x2:Q",
            y=alt.Y("count():Q", title="count"),
            tooltip=[
                alt.Tooltip("bin_center:Q", title="Bin center", format=".3g"),
                alt.Tooltip("count():Q", title="Count", format=","),
            ],
        )
        .transform_calculate(
            bin_center="(datum.x + datum.x2) / 2"  # Calculate bin center
        )
        # .add_params(selection)  # Add the selection to the chart
        .properties(width="container")
    )
    return chart


def make_overlay_chart(
    df: pl.DataFrame, field: str, filters: list[BurstFilterItem]
) -> alt.LayerChart:
    if filters:
        f_expr = reduce(and_, chain(*[f.as_expr() for f in filters]))
        df_f = df.filter(f_expr)
    else:
        df_f = df

    base_chart = make_chart(df, field, opacity=0.5)

    selection = alt.selection_interval(name="range", encodings=["x"])
    filtered_chart = make_chart(df_f, field).add_params(selection)

    chart = base_chart + filtered_chart
    return chart


# initial_chart = make_chart(field)
# inital_spec = initial_chart.to_dict()
column_options = [col_name for col_name in df.columns if df[col_name].dtype.is_numeric()]


@solara.component
def RangeInputField(
    label: str,
    value: float | int | None,
    on_value: Callable[[float | int | None], None],
    vtype: Type[float] | Type[int],
    vmin: Optional[float | int] = None,
    vmax: Optional[float | int] = None,
    disabled: bool = False,
):
    """Input field for a float value with a range slider."""

    if vmin is not None and vmax is not None and vmin >= vmax:
        raise ValueError("vmin must be smaller than vmax")

    error, set_error = solara.use_state(False)
    message, set_message = solara.use_state("")

    def inputtext_cb(new_value: str):
        print(new_value)
        if not new_value:
            on_value(None)
            return
        try:
            value = vtype(new_value)

        except ValueError:
            if vtype == int:
                set_message("Input must be an integer")
            else:
                set_message("Input must be a number")
            set_error(True)
            return
        if vmin is not None and value < vmin:
            set_message(f"Input must be >= {vmin}")
            set_error(True)
            return
        if vmax is not None and value > vmax:
            set_message(f"Input must be <= {vmax}")
            set_error(True)
            return
        set_error(False)
        set_message("")
        on_value(value)

    solara.InputText(
        label=label,
        value=str(value),
        error=error,
        message=message,
        disabled=disabled,
        on_value=inputtext_cb,
    )


style = """
/* Make sure action menu isn't cut off */
.vega-embed {
    overflow: visible;
    width: 100% !important;
}

"""

# TODO selectors to choose the node ?

field_options = [col_name for col_name in df.columns if df[col_name].dtype.is_numeric()]

initial_field = "n_photons"
initial_chart = make_chart(df, initial_field)
initial_chart = make_overlay_chart(df, initial_field, state.filters.items)

import reacton.ipyvuetify as v
from solara.util import _combine_classes


@solara.component
def MyCard(
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    elevation: int = 2,
    margin=2,
    children: List[solara.Element] = [],
    classes: List[str] = [],
    style: Union[str, Dict[str, str], None] = None,
):
    class_ = _combine_classes([f"ma-{margin}", *classes])
    style_flat = solara.util._flatten_style(style)
    children_actions = []
    children_text = []
    for child in children:
        if isinstance(child, solara.Element) and child.component == solara.CardActions:
            children_actions.extend(child.kwargs.get("children", []))
        else:
            children_text.append(child)
    with v.Card(elevation=elevation, class_=class_, style_=style_flat) as main:
        if title:
            with v.CardTitle(
                children=title,
            ):
                pass
        if subtitle:
            with v.CardSubtitle(
                children=[subtitle],
            ):
                pass
        with v.CardText(children=children_text):
            pass
        if children_actions:
            with v.CardActions(children=children_actions):
                pass
    return main


@solara.component
def SelectionChart(chart: alt.Chart, on_selection):
    jchart = alt.JupyterChart.element(chart=chart, embed_options={"actions": False})

    def bind():
        widget = solara.get_widget(jchart)
        widget.selections.observe(on_selection, "range")

    solara.use_effect(bind, [chart])


@solara.component
def Page():
    field = solara.use_reactive(initial_field)
    # chart = solara.use_reactive(initial_chart)
    selection = solara.use_reactive(None)

    if selection.value is not None:
        interval_selection = selection.value["new"].value
        if interval_selection:
            vmin, vmax = interval_selection["x"]
            val = (vmin, vmax)
        else:
            val = "removed selection"
    else:
        val = None

    solara.Text(str(val))

    existing_filter_fields = [f.name for f in state.filters]

    solara.Style(style)

    def on_field(value):
        # new_chart = make_chart(df, value)

        new_chart = make_overlay_chart(df, value, state.filters.items)

        # chart.set(new_chart)
        field.set(value)

    def make_chart():
        new_chart = make_overlay_chart(df, field.value, state.filters.items)
        return new_chart

    task_chart = solara.lab.use_task(make_chart, dependencies=[field.value, state.filters.items])

    def filter_from_selection():
        try:
            vmin, vmax = selection.value["new"].value["x"]
        except (KeyError, TypeError):
            vmin, vmax = None, None

        if field.value in existing_filter_fields:
            current_item_idx = find_index(state.filters.items, name=field.value)
            current_item = state.filters.items[current_item_idx]
            new_item = replace(current_item, min=vmin, max=vmax)
            state.filters.set_item(current_item_idx, new_item)
        else:
            filter_item = BurstFilterItem(name=field.value, min=vmin, max=vmax)
            state.filters.append(filter_item)

    solara.Text(str(state.filters.items))

    with solara.ColumnsResponsive([8, 4]):
        with solara.Card("Histogram"):
            with solara.Row(style={"align-items": "center"}):
                solara.Select(
                    label="Field", value=field.value, values=field_options, on_value=on_field
                )
                solara.IconButton("mdi-filter-plus", on_click=filter_from_selection)
            # todo wrap in component as FigureFromTask
            if task_chart.latest is None:
                solara.Text("loading...")
            else:
                chart = task_chart.value if task_chart.finished else task_chart.latest
                assert chart is not None
                with solara.Div(style="opacity: 0.3" if task_chart.pending else None):
                    SelectionChart(chart, on_selection=selection.set)

            # if task_chart.finished:?

            # SelectionChart(chart.value, on_selection=selection.set)
            # alt.JupyterChart.element(chart=chart.value, embed_options={"actions": False})
        with solara.Column():
            # with solara.Card("Filters"):
            #     with solara.Row():
            #         solara.Select(
            #             label="Field",
            #             value=new_field.value,
            #             on_value=new_field.set,
            #             values=new_field_values,
            #         )

            #         def new_filter():
            #             filter_item = BurstFilterItem(name=new_field.value)
            #             state.filters.append(filter_item)

            #         solara.Button("Add", on_click=new_filter)

            text_style = {
                "fontWeight": "bold",  # Can use numbers 100-900 for weight
                "fontSize": "16px",
                "padding": "8px 0",  # Optional: add some padding
            }

            for filter_item in state.filters:
                with solara.Card(title=None):
                    with solara.Row(style={"align-items": "end"}, justify="space-between"):
                        solara.Text(filter_item.name, style=text_style)
                        with solara.Row(style={"align-items": "end"}, gap="1px"):
                            solara.Checkbox(label="", style="marginBottom: -16px")
                            # solara.Switch(
                            #     value=filter_item.name in active_filters.items,
                            # )

                            def on_delete(filter_item=filter_item):
                                state.filters.remove(filter_item)

                            solara.IconButton(
                                icon_name="delete",
                                on_click=on_delete,
                            )

                    with solara.Row():
                        solara.InputFloat(label="Lower bound", value=filter_item.min)
                        solara.InputFloat(label="Upper bound", value=filter_item.max)

            # with solara.Card():
            # text_style = {
            #     "fontWeight": "500",  # Can use numbers 100-900 for weight
            #     "fontSize": "16px",
            #     # "fontFamily": "'Segoe UI', sans-serif",
            #     # "letterSpacing": "0.5px",  # Optional: adjust letter spacing
            #     # "lineHeight": "1.5",  # Optional: adjust line height
            #     # "padding": "8px 0",  # Optional: add some padding
            # }

            # solara.Text("hellow", style=text_style)
            # solara.Text("hellow")
            # with solara.Div(
            #     style={
            #         "display": "flex",
            #         "alignItems": "center",  # Vertical alignment
            #         "gap": "12px",  # Space between items
            #         "justifyContent": "flex-start",  # Horizontal alignment
            #         #     "fontWeight": "900",  # Can use numbers 100-900 for weight
            #         #     "fontSize": "16px",
            #         # "fontFamily": "'Segoe UI', sans-serif",
            #         # "letterSpacing": "0.5px",  # Optional: adjust letter spacing
            #         # "lineHeight": "1.5",  # Optional: adjust line height
            #         # "padding": "8px 0",  # Optional: add some padding
            #     }
            # ):
            #     solara.Text("sometext", style=text_style)
            #     # with solara.Row(style={"align-items": "end"}, gap="1px"):
            #     solara.Checkbox(label="", style="marginBottom: -16px")
            #     # solara.Switch(
            #     #     value=filter_item.name in active_filters.items,
            #     # )
            #     solara.IconButton(
            #         icon_name="delete",
            #         # on_click=lambda: state.filters.remove(filter_item),
            #     )

    # def on_field(value):

    # values = [f.name for f in state.filters]
    # for filter_item in state.filters:
    #     with solara.Row():
    #         solara.Switch(
    #             value=filter_item.name in active_filters.items,
    #         )
    #         solara.Select(
    #             label="Field",
    #             value=filter_item.name,
    #             values=[filter_item.name] + values,
    #         )
    #     with solara.Row():
    #         solara.InputFloat(label="Lower bound", value=filter_item.min)
    #         solara.InputFloat(label="Upper bound", value=filter_item.max)

    # print(selection.is_set)
    # BurstFigure(selection, state.filters)
    # if len(state.fret_nodes) != 0:
    #     with solara.Column(style={"height": "100%"}):
    #         solara.DataFrame(df)
    # else:
    #     solara.Text("Loading fret nodes...")


# %%
