from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, TypeVar, cast

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import solara
import solara.lab
from plotly.subplots import make_subplots
from solara.alias import rv
from solara.toestand import Ref

from dont_fret.web.bursts.methods import create_histogram
from dont_fret.web.components import RangeInputField
from dont_fret.web.methods import chain_filters
from dont_fret.web.models import BinnedImage, BurstFilterItem, BurstNode, BurstPlotSettings
from dont_fret.web.new_models import FRETNode, FRETStore
from dont_fret.web.reactive import ReactiveFRETNodes
from dont_fret.web.utils import not_none

N_STEP = 1000  # number slider steps
CMAP = px.colors.sequential.Viridis
WRATIO = 3.5


@solara.component
def FigurePlotlyShapes(
    fig: go.Figure,
    shapes: dict,
    dependencies=None,
):
    from plotly.graph_objs._figurewidget import FigureWidget

    fig_element = FigureWidget.element()  # type: ignore

    def update_data():
        fig_widget: FigureWidget = solara.get_widget(fig_element)  # type: ignore
        fig_widget.layout = fig.layout

        length = len(fig_widget.data)  # type: ignore
        fig_widget.add_traces(fig.data)
        data = list(fig_widget.data)
        fig_widget.data = data[length:]

    def update_shapes():
        if shapes:
            fig_widget: FigureWidget = solara.get_widget(fig_element)  # type: ignore
            fig_widget.update_shapes(**shapes)

    solara.use_effect(update_data, dependencies or fig)
    solara.use_effect(update_shapes, shapes)

    return fig_element


@solara.component
def EditFilterDialog(
    filter_item: solara.Reactive[BurstFilterItem],  # should be a reactive
    data: np.ndarray,
    on_close: Callable[[], None],
):
    def bin_data():
        data_f = data[~np.isnan(data)]
        counts, binspace = np.histogram(data_f, bins="fd")
        xbins = {"start": binspace[0], "end": binspace[-1], "size": binspace[1] - binspace[0]}
        arange = 2 * binspace[0] - 0.05 * binspace[-1], 1.05 * binspace[-1] - binspace[0]

        return data_f, xbins, arange

    data_f, xbins, arange = solara.use_memo(bin_data, [])

    xr_default = (
        not_none(filter_item.value.min, arange[0]),
        not_none(filter_item.value.max, arange[1]),
    )

    xrange, set_xrange = solara.use_state(xr_default)
    shapes, set_shapes = solara.use_state({})

    def make_figure():
        return create_histogram(data_f, xbins, arange, xrange)

    fig = solara.use_memo(make_figure, [])

    show_slider, set_show_slider = solara.use_state(True)

    def update_xmin(value):
        set_xrange((value, xrange[1]))
        d = {"patch": dict(x0=arange[0], x1=value), "selector": 0}
        set_shapes(d)

    def update_xmax(value):
        set_xrange((xrange[0], value))
        d = {"patch": dict(x0=value, x1=arange[1]), "selector": 1}
        set_shapes(d)

    def on_slider(value: tuple[float, float]):
        if value[0] != xrange[0]:
            d = {"patch": dict(x0=arange[0], x1=value[0]), "selector": 0}
        elif value[1] != xrange[1]:
            d = {"patch": dict(x0=value[1], x1=arange[1]), "selector": 1}
        else:
            return
        set_shapes(d)
        set_xrange(value)

    with solara.Card(f"Filter: {filter_item.value.name}"):
        FigurePlotlyShapes(fig, shapes=shapes)
        step = (arange[1] - arange[0]) / N_STEP
        with solara.Row():
            with solara.Tooltip(
                "Disable slider to prevent threshold value rounding."  # type: ignore
            ):
                rv.Switch(v_model=show_slider, on_v_model=set_show_slider)
            if show_slider:
                solara.SliderRangeFloat(
                    label="",
                    min=arange[0],
                    max=arange[1],
                    value=xrange,
                    step=step,
                    on_value=on_slider,
                )
        with solara.Row():
            RangeInputField(
                label="Min",
                value=xrange[0],
                vtype=float,
                on_value=update_xmin,
                vmin=arange[0],
            )
            RangeInputField(
                label="Max",
                value=xrange[1],
                vtype=float,
                on_value=update_xmax,
                vmax=arange[1],
            )

        def on_save():
            new_filter = BurstFilterItem(
                filter_item.value.name,
                min=xrange[0],
                max=xrange[1],
            )
            filter_item.set(new_filter)
            on_close()

        with solara.CardActions():
            rv.Spacer()
            solara.Button("Save", icon_name="mdi-content-save", on_click=on_save)
            solara.Button("Cancel", icon_name="mdi-window-close", on_click=on_close)


@solara.component
def FilterListItem(
    filter_item: solara.Reactive[BurstFilterItem], data: np.ndarray, on_delete: Callable[[], None]
):
    edit, set_edit = solara.use_state(False)
    with rv.ListItem():
        rv.ListItemAvatar(children=[rv.Icon(children=["mdi-filter"])])
        rv.ListItemTitle(children=[filter_item.value.name])

        # TODO multi line
        def fmt(v):
            if v is None:
                return "None"
            else:
                return f"{v:.5g}"

        rv.ListItemSubtitle(
            children=[f"{fmt(filter_item.value.min)} - {fmt(filter_item.value.max)}"]
        )

        solara.IconButton(
            color="secondary",
            small=True,
            rounded=True,
            icon_name="mdi-delete",
            on_click=on_delete,
        )

        solara.IconButton(
            color="secondary",
            small=True,
            rounded=True,
            icon_name="mdi-pencil",
            on_click=lambda: set_edit(True),
        )

    with rv.Dialog(v_model=edit, max_width=750, on_v_model=set_edit):
        if edit:
            EditFilterDialog(
                filter_item,
                data,
                on_close=lambda: set_edit(False),
            )


DTYPES = {
    "E_app": float,
    "S_app": float,
    "n_photons": int,
    "time_length": float,
    "time_mean": float,
    "time_min": float,
    "time_max": float,
    "n_DD": int,
    "n_DA": int,
    "n_AA": int,
    "n_AD": int,
    "tau_DD": float,
    "tau_DA": float,
    "tau_AA": float,
    "tau_AD": float,
}


@solara.component
def BurstFilters(filters: solara.Reactive[list[BurstFilterItem]], dataframe: pl.DataFrame):
    f_names = [f.name for f in filters.value]
    attrs = [k for k in DTYPES if k not in f_names]
    new_filter_name, set_new_filter_name = solara.use_state(attrs[0])

    with solara.Card(title="Global filters"):
        with rv.List(dense=False):
            for idx, flt in enumerate(filters.value):

                def on_delete(idx=idx):
                    new_filters = filters.value.copy()
                    del new_filters[idx]
                    filters.set(new_filters)

                arr = dataframe[flt.name].to_numpy()
                FilterListItem(Ref(filters.fields[idx]), arr, on_delete)

        def add_filter():
            item = BurstFilterItem(name=new_filter_name)
            new_filters = filters.value.copy()
            new_filters.append(item)
            filters.set(new_filters)

        solara.Select(
            label="Filter attribute",
            value=new_filter_name,
            values=attrs,
            on_value=set_new_filter_name,
        )
        solara.Button("Add filter", on_click=lambda: add_filter(), block=True)


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
def FileFilterDialog(
    burst_item: solara.Reactive[BurstNode],
    on_close: Callable[[], None],
):
    all_files = sorted(burst_item.value.df["filename"].unique())
    local_selected_files = solara.use_reactive(cast(list[str], burst_item.value.selected_files))
    error, set_error = solara.use_state("")
    regex, set_regex = solara.use_state("")

    def on_input(value: str):
        try:
            pattern = re.compile(value)
            set_error("")
        except Exception:
            set_error("Invalid regex")
            set_regex(value)
            return
        new_selected = [f for f in all_files if pattern.search(f)]
        local_selected_files.set(new_selected)

    def on_save():
        if not local_selected_files.value:
            return
        burst_item.update(selected_files=local_selected_files.value)
        # selected_files.set(local_selected_files.value)
        on_close()

    with solara.Card("File Filter"):
        with solara.Row(style="align-items: center;"):
            solara.InputText(
                label="regex", value=regex, on_value=on_input, continuous_update=True, error=error
            )
            solara.Button(label="Select All", on_click=lambda: local_selected_files.set(all_files))
            solara.Button(label="Select None", on_click=lambda: local_selected_files.set([]))
        with solara.v.List(nav=True):
            with solara.v.ListItemGroup(
                v_model=local_selected_files.value,
                on_v_model=local_selected_files.set,
                multiple=True,
            ):
                for filename in all_files:
                    with solara.v.ListItem(value=filename):
                        with solara.v.ListItemAction():
                            solara.Checkbox(value=filename in local_selected_files.value)
                        solara.v.ListItemTitle(children=[filename])

        with solara.CardActions():
            solara.v.Spacer()
            solara.Button(
                "Save",
                icon_name="mdi-content-save",
                on_click=on_save,
                disabled=not local_selected_files.value,
            )
            solara.Button("Close", icon_name="mdi-window-close", on_click=on_close)


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
    bin_result = solara.use_thread(
        rebin, dependencies=[getattr(copy.value, attr) for attr in redraw_attrs]
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
                enable_restore=False,
            )

            RangeInputField(
                label="X max",
                value=copy.value.x_max,
                on_value=lambda val: copy.update(x_max=val),
                vtype=float,
                vmin=copy.value.x_min,
                enable_restore=False,
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
                enable_restore=False,
            )

            RangeInputField(
                label="Y max",
                value=copy.value.y_max,
                on_value=lambda val: copy.update(y_max=val),
                vtype=float,
                vmin=copy.value.y_min,
                enable_restore=False,
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
                    enable_restore=False,
                    disabled=disabled,
                )
                RangeInputField(
                    label="Z max",
                    value=copy.value.z_max,
                    on_value=lambda val: copy.update(z_max=val),
                    vtype=float,
                    vmin=copy.value.z_min,
                    enable_restore=False,
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


def find_object(items: list[T], **kwargs) -> T:
    for item in items:
        if all(getattr(item, key) == value for key, value in kwargs.items()):
            return item
    raise ValueError("Object not found")


@dataclass
class BurstFigureSelection:
    fret_store: FRETStore

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


# = VIEW
@solara.component
def BurstFigure(
    selection: BurstFigureSelection,
    global_filters: solara.Reactive[list[BurstFilterItem]],
):
    figure, set_figure = solara.use_state(cast(Optional[go.Figure], None))
    # edit_filter, set_edit_filter = solara.use_state(False)
    edit_settings, set_edit_settings = solara.use_state(False)
    plot_settings = solara.use_reactive(
        BurstPlotSettings()
    )  # -> these reset to default, combine with burstfigureselection?

    dark_effective = solara.lab.use_dark_effective()

    # this is triggered twice ? -> known plotly bug
    def redraw():
        # get the names of the files
        selected_file_names = [
            node.name
            for node in selection.burst_node.photon_nodes
            if node.id.hex in selection.selected_files
        ]
        file_filter = pl.col("filename").is_in(selected_file_names)
        f_expr = chain_filters(global_filters.value) & file_filter
        filtered_df = selection.burst_node.df.filter(f_expr)
        img = BinnedImage.from_settings(filtered_df, plot_settings.value)
        figure = generate_figure(
            filtered_df, plot_settings.value, binned_image=img, dark=dark_effective
        )
        set_figure(figure)

    # todo upgrade to use_task
    fig_result = solara.use_thread(
        redraw,
        dependencies=[
            selection.burst_id.value,
            selection.selected_files,
            plot_settings.value,
            global_filters.value,
            dark_effective,
        ],
        intrusive_cancel=False,  # is much faster
    )

    with solara.Card():
        with solara.Row():
            solara.Select(
                label="Measurement",
                value=selection.fret_id.value.hex,
                on_value=selection.set_fret_id,  # type: ignore
                values=selection.fret_values,  # type: ignore
            )

            solara.Select(
                label="Burst item",
                value=selection.burst_id.value.hex,
                on_value=selection.set_burst_id,
                values=selection.burst_values,  # type: ignore
            )
            SelectCount(
                label="Files",
                value=selection.selected_files,
                on_value=selection.set_selected_files,
                items=selection.selected_files_values,  # type: ignore
                item_name="file",
            )
            # solara.IconButton(icon_name="mdi-file-star", on_click=lambda: set_edit_filter(True))
            solara.IconButton(icon_name="mdi-settings", on_click=lambda: set_edit_settings(True))

        solara.ProgressLinear(fig_result.state == solara.ResultState.RUNNING)
        if figure is not None:
            with solara.Div(
                style="opacity: 0.3" if fig_result.state == solara.ResultState.RUNNING else None
            ):
                solara.FigurePlotly(figure)

        # dedent this and figure will flicker/be removed when opening the dialog
        if edit_settings:
            with rv.Dialog(v_model=edit_settings, max_width=750, on_v_model=set_edit_settings):
                PlotSettingsEditDialog(
                    plot_settings,
                    selection.burst_node.df.filter(f_expr),  # = filtered dataframe by global filter
                    on_close=lambda: set_edit_settings(False),
                    duration=selection.burst_node.duration,
                )

        # if edit_filter:
        #     pass
        # with rv.Dialog(v_model=edit_filter, max_width=750, on_v_model=set_edit_filter):
        #     FileFilterDialog(
        #         burst_item=burst_ref,
        #         on_close=lambda: set_edit_filter(False),
        #     )


# = VIEW
@solara.component
def BurstFigureDepr(
    fret_nodes: list[FRETNode],  # can be only the value since it doesnt edit the nodes
    global_filters: solara.Reactive[list[BurstFilterItem]],
    node_idx: Optional[solara.Reactive[int]] = None,
    burst_idx: Optional[solara.Reactive[int]] = None,
):
    # we are only interested in nodes with bursts
    # TODO move to controller object
    has_nodes = [node for node in fret_nodes if node.bursts.items]
    node_idx = solara.use_reactive(node_idx if node_idx is not None else 0)
    burst_idx = solara.use_reactive(burst_idx if burst_idx is not None else 0)

    fret_node = has_nodes[node_idx.value]
    burst_node = fret_node.bursts[burst_idx.value]

    # selector options
    fret_node_values = [{"text": node.name.value, "value": i} for i, node in enumerate(has_nodes)]
    burst_node_values = [
        {"text": burst_node.name, "value": i} for i, burst_node in enumerate(fret_node.bursts.items)
    ]

    figure, set_figure = solara.use_state(cast(Optional[go.Figure], None))
    edit_filter, set_edit_filter = solara.use_state(False)
    edit_settings, set_edit_settings = solara.use_state(False)
    plot_settings = solara.use_reactive(BurstPlotSettings())

    dark_effective = solara.lab.use_dark_effective()

    #
    f_expr = chain_filters(global_filters.value)  # & file_filter

    # this is triggered twice ? -> known plotly bug
    def redraw():
        # file_filter = pl.col("filename").is_in(burst_ref.value.selected_files)

        filtered_df = burst_node.df.filter(f_expr)
        img = BinnedImage.from_settings(filtered_df, plot_settings.value)
        figure = generate_figure(
            filtered_df, plot_settings.value, binned_image=img, dark=dark_effective
        )
        set_figure(figure)

    # -> use_task
    fig_result = solara.use_thread(
        redraw,
        dependencies=[
            node_idx.value,
            burst_idx.value,
            plot_settings.value,
            global_filters.value,
            dark_effective,
        ],
        intrusive_cancel=False,  # is much faster
    )

    def on_fret_node(value: int):
        node_idx.set(value)
        burst_idx.set(0)

    print(fret_node_values)
    with solara.Card():
        with solara.Row():
            solara.Select(
                label="Measurement",
                value=node_idx.value,
                on_value=on_fret_node,  # type: ignore
                values=fret_node_values,  # type: ignore
            )

            solara.Select(
                label="Burst item",
                value=burst_idx.value,
                on_value=burst_idx.set,
                values=burst_node_values,  # type: ignore
            )

            solara.IconButton(icon_name="mdi-file-star", on_click=lambda: set_edit_filter(True))
            solara.IconButton(icon_name="mdi-settings", on_click=lambda: set_edit_settings(True))

        solara.ProgressLinear(fig_result.state == solara.ResultState.RUNNING)
        if figure is not None:
            with solara.Div(
                style="opacity: 0.3" if fig_result.state == solara.ResultState.RUNNING else None
            ):
                solara.FigurePlotly(figure)

        # dedent this and figure will flicker/be removed when opening the dialog
        if edit_settings:
            with rv.Dialog(v_model=edit_settings, max_width=750, on_v_model=set_edit_settings):
                PlotSettingsEditDialog(
                    plot_settings,
                    burst_node.df.filter(f_expr),  # = filtered dataframe by global filter
                    on_close=lambda: set_edit_settings(False),
                    duration=burst_node.duration,
                )

        if edit_filter:
            pass
            # with rv.Dialog(v_model=edit_filter, max_width=750, on_v_model=set_edit_filter):
            #     FileFilterDialog(
            #         burst_item=burst_ref,
            #         on_close=lambda: set_edit_filter(False),
            #     )
