from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from typing import Callable, Literal, Optional, TypeVar, cast

import altair as alt
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import solara
import solara.lab
from plotly.subplots import make_subplots
from solara.alias import rv

import dont_fret.web.state as state
from dont_fret.web.components import FigureFromTask, RangeInputField, RegexSelectDialog
from dont_fret.web.methods import chain_filters
from dont_fret.web.models import BinnedImage, BurstFilterItem, BurstNode, BurstPlotSettings
from dont_fret.web.new_models import FRETNode, FRETStore, ListStore
from dont_fret.web.utils import (
    NestedSelectors,
    find_index,
    find_object,
    get_bursts,
    make_selector_nodes,
)

N_STEP = 1000  # number slider steps
CMAP = px.colors.sequential.Viridis
WRATIO = 3.5

DEFAULT_FIELD = "n_photons"
# TODO this is hardcoded but it depends on cfg settings what fields burst item dataframes have
# -> deduce from selection
FIELD_OPTIONS = [
    "burst_index",
    "n_DD",
    "n_DA",
    "n_AA",
    "n_AD",
    "nanotimes_DD",
    "nanotimes_DA",
    "nanotimes_AA",
    "nanotimes_AD",
    "timestamps_mean",
    "timestamps_min",
    "timestamps_max",
    "E_app",
    "S_app",
    "n_photons",
    "time_mean",
    "time_length",
    "tau_DD",
    "tau_DA",
    "tau_AA",
    "tau_AD",
]


def fd_bin_width(data: pl.Series) -> float:
    """
    Calculate bin width using the Freedman-Diaconis rule:
    bin width = 2 * IQR * n^(-1/3)
    where IQR is the interquartile range and n is the number of observations
    """
    q75, q25 = data.quantile(0.75), data.quantile(0.25)
    assert q75 is not None
    assert q25 is not None
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
) -> alt.LayerChart | alt.Chart:
    f = chain_filters(filters)
    df_f = df.filter(f)

    base_chart = make_chart(df, field, opacity=0.5)
    selection = alt.selection_interval(name="range", encodings=["x"])

    if len(df_f) > 0:
        filtered_chart = make_chart(df_f, field).add_params(selection)
        chart = base_chart + filtered_chart
    else:
        chart = base_chart.add_params(selection)

    return chart


@solara.component
def SelectionChart(chart: alt.Chart | alt.LayerChart, on_selection):
    jchart = alt.JupyterChart.element(chart=chart, embed_options={"actions": False})  # type: ignore

    def bind():
        widget = cast(alt.JupyterChart, solara.get_widget(jchart))
        widget.selections.observe(on_selection, "range")

    solara.use_effect(bind, [chart])


@solara.component
def FilterItemCard(filter_item: BurstFilterItem):
    text_style = {
        "fontWeight": "bold",
        "fontSize": "16px",
        "padding": "8px 0",
    }
    with solara.Card(title=None):
        with solara.Row(style={"align-items": "end"}, justify="space-between"):
            solara.Text(filter_item.name, style=text_style)
            with solara.Row(style={"align-items": "end"}, gap="1px"):

                def on_checkbox(value: bool, filter_item=filter_item):
                    idx = state.filters.index(filter_item)
                    state.filters.update(idx, active=value)

                solara.Checkbox(label="", style="marginBottom: -16px", on_value=on_checkbox)

                def on_delete(filter_item=filter_item):
                    state.filters.remove(filter_item)

                solara.IconButton(
                    icon_name="delete",
                    on_click=on_delete,
                )

        with solara.Row():

            def on_vmin(value, filter_item=filter_item):
                idx = state.filters.index(filter_item)
                state.filters.update(idx, min=value)

            RangeInputField(
                label="Lower bound",
                value=filter_item.min,
                on_value=on_vmin,
                vmax=filter_item.max,
                vtype=float,
                allow_none=True,
            )

            def on_vmax(value, filter_item=filter_item):
                idx = state.filters.index(filter_item)
                state.filters.update(idx, max=value)

            RangeInputField(
                label="Upper bound",
                value=filter_item.max,
                on_value=on_vmax,
                vmin=filter_item.min,
                vtype=float,
                allow_none=True,
            )


@solara.component
def FilterEditDialog():
    field = solara.use_reactive(DEFAULT_FIELD)
    chart_selection = solara.use_reactive(None)
    existing_filter_fields = [f.name for f in state.filters]
    selector_nodes = make_selector_nodes(state.fret_nodes.items, attr="bursts")

    def make_store():
        return ListStore[str]([selector_nodes[0].value, selector_nodes[0].children[0].value])

    burst_node_choice = solara.use_memo(make_store, [])

    def make_chart():
        burst_node = get_bursts(state.fret_nodes.items, burst_node_choice.items)
        new_chart = make_overlay_chart(burst_node.df, field.value, state.filters.items)
        return new_chart

    # note: cant be 100% sure but you could get some panicking threads when using
    # vegafusion in a use_task hook with prefer_threaded=True
    task_chart = solara.lab.use_task(  # type: ignore
        make_chart,
        dependencies=[field.value, state.filters.items, burst_node_choice.items],
        prefer_threaded=False,
    )

    def filter_from_selection():
        """adds or edits a filter based on the current selection in the chart"""
        try:
            vmin, vmax = chart_selection.value["new"].value["x"]  # type: ignore
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

    with solara.ColumnsResponsive([8, 4]):
        with solara.Card("Histogram"):
            labels = ["Measurement", "Bursts"]  # TODO move elsewhere
            selectors = NestedSelectors(
                nodes=selector_nodes, selection=burst_node_choice, labels=labels
            )
            with solara.Row():
                for level in selectors:
                    solara.Select(**level)

            with solara.Row(style={"align-items": "center"}):

                def on_field(value):
                    chart_selection.set(None)
                    field.set(value)

                solara.Select(
                    label="Field", value=field.value, values=FIELD_OPTIONS, on_value=on_field
                )
                solara.IconButton("mdi-filter-plus", on_click=filter_from_selection)
            if task_chart.latest is None:
                solara.Text("loading...")
            else:
                chart = task_chart.value if task_chart.finished else task_chart.latest
                assert chart is not None
                with solara.Div(style="opacity: 0.3" if task_chart.pending else None):
                    SelectionChart(chart, on_selection=chart_selection.set)
        with solara.Column():
            for filter_item in state.filters:
                FilterItemCard(filter_item)


@solara.component
def PlotSettingsEditDialog(
    plot_settings: solara.Reactive[BurstPlotSettings],
    df: pl.DataFrame,
    on_close: Callable[[], None],
    duration: Optional[float] = None,
):
    copy = solara.use_reactive(plot_settings.value)
    img, set_img = solara.use_state(cast(Optional[BinnedImage], None))
    items = list(df.columns)

    drop_cols = ["filename", "burst_index"]
    for col in drop_cols:
        if col in items:
            items.remove(col)

    def on_save():
        plot_settings.value = copy.value
        on_close()

    def autolimit_xy(field: str, axis: Literal["x"] | Literal["y"]):
        if field in ["E_app", "S_app"]:
            update = {f"{axis}_min": 0.0, f"{axis}_max": 1.0}
        elif field.startswith("time") and duration is not None:
            update = {f"{axis}_min": 0.0, f"{axis}_max": duration}
        # elif field in time .. (autolimit to 0, aquisition duration)
        else:
            update = {f"{axis}_min": df[field].min(), f"{axis}_max": df[field].max()}
        copy.update(**update)

    def autolimit_z():
        if img is not None:
            copy.update(z_min=0.0, z_max=img.img_data.max())

    # TODO z limits should only update when:
    # they were `None` (first redraw)
    # any of the 'redraw' attributes changed
    def rebin():
        if (
            copy.value == plot_settings.value
            and copy.value.z_min is not None
            and copy.value.z_max is not None
        ):
            return
        img = BinnedImage.from_settings(df, copy.value)
        set_img(img)
        copy.update(z_min=0.0, z_max=img.img_data.max())

    # only *some* of copy's attributes should trigger the redraw (specifically not z_min, z_max)
    redraw_attrs = ["x_name", "y_name", "x_min", "x_max", "y_min", "y_max", "nbinsx", "nbinsy"]

    # TODO use_task
    bin_result = solara.use_thread(
        rebin,  # type: ignore
        dependencies=[getattr(copy.value, attr) for attr in redraw_attrs],
    )
    disabled = bin_result.state == solara.ResultState.RUNNING
    with solara.Card("Plot Settings"):
        solara.Select(
            label="X value",
            value=copy.value.x_name,
            values=items,
            on_value=lambda val: copy.update(x_name=val),
        )

        solara.Select(
            label="Y value",
            value=copy.value.y_name,
            values=items,
            on_value=lambda val: copy.update(y_name=val),
        )

        with solara.Row():
            RangeInputField(
                label="X min",
                value=copy.value.x_min,
                on_value=lambda val: copy.update(x_min=val),
                vtype=float,
                vmax=copy.value.x_max,
            )

            RangeInputField(
                label="X max",
                value=copy.value.x_max,
                on_value=lambda val: copy.update(x_max=val),
                vtype=float,
                vmin=copy.value.x_min,
            )
            solara.IconButton(
                icon_name="mdi-auto-fix",
                on_click=lambda *args: autolimit_xy(copy.value.x_name, "x"),
            )

        with solara.Row():
            RangeInputField(
                label="Y min",
                value=copy.value.y_min,
                on_value=lambda val: copy.update(y_min=val),
                vtype=float,
                vmax=copy.value.y_max,
            )

            RangeInputField(
                label="Y max",
                value=copy.value.y_max,
                on_value=lambda val: copy.update(y_max=val),
                vtype=float,
                vmin=copy.value.y_min,
            )

            solara.IconButton(
                icon_name="mdi-auto-fix",
                on_click=lambda *args: autolimit_xy(copy.value.y_name, "y"),
            )

        # this should never be `True` after rebin thread finished for the first time
        if copy.value.z_min is not None and copy.value.z_max is not None:
            with solara.Row():
                RangeInputField(
                    label="Z min",
                    value=copy.value.z_min,
                    on_value=lambda val: copy.update(z_min=val),
                    vtype=float,
                    vmax=copy.value.z_max,
                    disabled=disabled,
                )
                RangeInputField(
                    label="Z max",
                    value=copy.value.z_max,
                    on_value=lambda val: copy.update(z_max=val),
                    vtype=float,
                    vmin=copy.value.z_min,
                    disabled=disabled,
                )

                solara.IconButton(
                    icon_name="mdi-auto-fix",
                    on_click=lambda *args: autolimit_z(),
                    disabled=disabled,
                )

        with solara.GridFixed(columns=2):
            solara.InputInt(
                label="N bins X",
                value=copy.value.nbinsx,
                on_value=lambda val: copy.update(nbinsx=val),
            )
            solara.InputInt(
                label="N bins Y",
                value=copy.value.nbinsy,
                on_value=lambda val: copy.update(nbinsy=val),
            )

        with solara.CardActions():
            rv.Spacer()
            solara.Button("Save", icon_name="mdi-content-save", on_click=on_save, disabled=disabled)
            solara.Button("Close", icon_name="mdi-window-close", on_click=on_close)


@solara.component_vue("select_count.vue")
def SelectCount(label, value, on_value, items, item_name="item"):
    pass


T = TypeVar("T")


@dataclass
class BurstFigureSelection:
    fret_store: FRETStore

    # TODO make sure it can not be None -> always some uuid
    # maybe we should make a class which solves the interdependent select for n states
    # and it should also be able to deal with deletions in the tree
    # and it takes the full tree as input
    fret_id: solara.Reactive[uuid.UUID | None] = field(
        default_factory=lambda: solara.reactive(None)
    )
    burst_id: solara.Reactive[uuid.UUID | None] = field(
        default_factory=lambda: solara.reactive(None)
    )

    # keys are tuple of (fret_id, burst_id)
    # values are hex values of the photon item uuids
    selection_store: solara.Reactive[dict[tuple[uuid.UUID, uuid.UUID], list[str]]] = field(
        default_factory=lambda: solara.reactive({})
    )

    def __post_init__(self):
        self.fret_store._items.subscribe(self.on_fret_store)
        self.reset()

    def on_fret_store(self, new_value: list[FRETNode]):
        options = [fret_node.id for fret_node in new_value]
        if self.fret_id.value not in options:
            self.reset()

    def reset(self):
        if self.has_bursts:
            fret_node = self.fret_nodes_with_bursts[0]
            self.set_fret_id(fret_node.id.hex)
        else:
            self.fret_id.set(None)
            self.burst_id.set(None)

    @property
    def is_set(self) -> bool:
        return bool(self.fret_id.value) and bool(self.burst_id.value)

    @property
    def fret_nodes_with_bursts(self) -> list[FRETNode]:
        return [node for node in self.fret_store if node.bursts]

    @property
    def has_bursts(self) -> bool:
        return bool(self.fret_nodes_with_bursts)

    @property
    def fret_values(self) -> list[dict]:
        return [
            {"text": node.name.value, "value": node.id.hex} for node in self.fret_nodes_with_bursts
        ]

    @property
    def burst_values(self) -> list[dict]:
        return [{"text": node.name, "value": node.id.hex} for node in self.fret_node.bursts]

    @property
    def fret_node(self) -> FRETNode:
        return find_object(self.fret_nodes_with_bursts, id=self.fret_id.value)

    @property
    def burst_node(self) -> BurstNode:
        return find_object(self.fret_node.bursts.items, id=self.burst_id.value)

    @property
    def selected_files_values(self) -> list[dict]:
        return [{"text": node.name, "value": node.id.hex} for node in self.burst_node.photon_nodes]

    @property
    def selected_files(self) -> list[str]:
        assert self.fret_id.value is not None
        assert self.burst_id.value is not None
        stored = self.selection_store.value.get((self.fret_id.value, self.burst_id.value), [])

        if stored:
            return stored

        return [item["value"] for item in self.selected_files_values]

    def set_fret_id(self, value: str):
        self.fret_id.set(uuid.UUID(value))

        # set the first burst node as the default value
        burst_node = self.fret_node.bursts[0]
        self.burst_id.set(burst_node.id)

    def set_burst_id(self, value: str):
        self.burst_id.set(uuid.UUID(value))

    def set_selected_files(self, values: list[str]):
        new_value = self.selection_store.value.copy()
        assert self.fret_id.value is not None
        assert self.burst_id.value is not None
        new_value[(self.fret_id.value, self.burst_id.value)] = values
        self.selection_store.value = new_value


def generate_figure(
    df: pl.DataFrame,
    plot_settings: BurstPlotSettings,
    binned_image: BinnedImage,
    dark: bool = False,
) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=2,
        shared_yaxes="rows",  # type: ignore
        shared_xaxes="columns",  # type: ignore
        horizontal_spacing=0.02,
        vertical_spacing=0.02,
        column_widths=[WRATIO, 1],
        row_heights=[1, WRATIO],
    )

    if sum([plot_settings.z_min is None, plot_settings.z_max is None]) == 1:
        raise ValueError("z_min and z_max must be both None or both not None")

    hist2d = go.Heatmap(
        x=binned_image.x,
        y=binned_image.y,
        z=binned_image.img_data.T,
        zmin=plot_settings.z_min,
        zmax=plot_settings.z_max,
        colorscale=CMAP,
        colorbar={"title": "Counts"},
    )

    fig.add_trace(hist2d, row=2, col=1)
    fig.update_xaxes(row=2, col=1, range=plot_settings.x_range, title=plot_settings.x_name)
    fig.update_yaxes(row=2, col=1, range=plot_settings.y_range, title=plot_settings.y_name)

    if dark:
        hist_settings = {
            "marker_color": "#adadad",
        }
    else:
        hist_settings = {
            "marker_color": "#330C73",
        }

    histx = go.Histogram(
        x=df[plot_settings.x_name],
        xbins=plot_settings.xbins,
        name=plot_settings.x_name,
        **hist_settings,
    )
    fig.add_trace(histx, row=1, col=1)
    fig.update_yaxes(row=1, col=1, title="counts")

    histy = go.Histogram(
        y=df[plot_settings.y_name],
        ybins=plot_settings.ybins,
        name=plot_settings.y_name,
        **hist_settings,
    )
    fig.add_trace(
        histy,
        row=2,
        col=2,
    )
    fig.update_xaxes(row=2, col=2, title="counts")
    fig.update_layout(
        width=700,
        height=700,
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20),
        template="plotly_dark" if dark else "plotly_white",
    )

    return fig


@solara.component
def BurstFigure(
    selection: ListStore[str],
    file_selection: dict[uuid.UUID, ListStore[str]],
):
    settings_dialog = solara.use_reactive(False)
    file_filter_dialog = solara.use_reactive(False)
    plot_settings = solara.use_reactive(
        BurstPlotSettings()
    )  # -> these reset to default, move to global state?

    dark_effective = solara.lab.use_dark_effective()

    labels = ["Measurement", "Bursts"]  # TODO move elsewhere
    selector_nodes = make_selector_nodes(state.fret_nodes.items, "bursts")
    # making the levels populates the selection
    levels = list(NestedSelectors(nodes=selector_nodes, selection=selection, labels=labels))
    burst_node = get_bursts(state.fret_nodes.items, selection.items)
    filenames = sorted(burst_node.df["filename"].unique())
    if burst_node.id in file_selection:
        file_store = file_selection[burst_node.id]
    else:
        file_store = ListStore(filenames)
        file_selection[burst_node.id] = file_store

    # file_store = file_selection[
    #     burst_node.id
    # ]  # we make a new one here if it doestn exist yet, but should be OK

    # # it should never be empty, only after making a new one, thus we set it to all selected if its emtp

    # solara.Text(str(burst_node.id))

    file_filter = pl.col("filename").is_in(file_store.items)
    f_expr = chain_filters(state.filters.items) & file_filter

    # this is triggered twice ? -> known plotly bug, use .key(...)
    def redraw():
        filtered_df = burst_node.df.filter(f_expr)
        img = BinnedImage.from_settings(filtered_df, plot_settings.value)
        figure = generate_figure(
            filtered_df, plot_settings.value, binned_image=img, dark=dark_effective
        )
        return figure

    figure_task = solara.lab.use_task(  # type: ignore
        redraw,
        dependencies=[burst_node.id, plot_settings.value, file_store.items, state.filters.items],
    )

    with solara.Card():
        with solara.Row():
            for level in levels:
                solara.Select(**level)

            solara.IconButton(
                icon_name="mdi-file-star", on_click=lambda: file_filter_dialog.set(True)
            )
            solara.IconButton(icon_name="mdi-settings", on_click=lambda: settings_dialog.set(True))

        FigureFromTask(figure_task)

    with solara.v.Dialog(
        v_model=settings_dialog.value, max_width=750, on_v_model=settings_dialog.set
    ):
        PlotSettingsEditDialog(
            plot_settings,
            burst_node.df.filter(f_expr),  # = filtered dataframe by global filter
            on_close=lambda: settings_dialog.set(False),
            duration=burst_node.duration,
        )

    with solara.v.Dialog(
        v_model=file_filter_dialog.value, max_width=750, on_v_model=file_filter_dialog.set
    ):
        RegexSelectDialog(
            title="File Filter",
            value=file_store.items,
            on_value=file_store.set,
            values=filenames,
            on_close=lambda: file_filter_dialog.set(False),
        )


@solara.component
def FilterListItem(filter_item):
    with rv.ListItem():
        rv.ListItemTitle(children=[filter_item.name])

        def fmt(v):
            if v is None:
                return "None"
            else:
                return f"{v:.3g}"

        rv.ListItemSubtitle(children=[f"{fmt(filter_item.min)} - {fmt(filter_item.max)}"])

        def on_check(value: bool, filter_item=filter_item):
            idx = state.filters.index(filter_item)
            state.filters.update(idx, active=value)

        solara.Checkbox(
            value=filter_item.active,
            on_value=on_check,
        )
