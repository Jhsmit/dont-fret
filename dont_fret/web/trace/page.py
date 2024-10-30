import uuid
from dataclasses import dataclass, field
from io import BytesIO
from typing import Optional, cast

import pandas as pd
import plotly.graph_objects as go
import polars as pl
import solara
import solara.lab
from solara.alias import rv

import dont_fret.web.state as state
from dont_fret import BinnedPhotonData, PhotonData
from dont_fret.formatting import TRACE_COLORS, TRACE_SIGNS
from dont_fret.web.components import FigureFromTask
from dont_fret.web.methods import generate_traces
from dont_fret.web.models import (
    BurstNode,
    PhotonNode,
    TCSPCSettings,
    TraceSettings,
)
from dont_fret.web.new_models import FRETNode, FRETStore, ListStore
from dont_fret.web.trace.methods import create_tcspc_histogram
from dont_fret.web.utils import (
    NestedSelectors,
    find_object,
    get_photons,
    make_selector_nodes,
    wrap_callback,
)

# TODO move fret node / photon file reactives to module level
TCSPC_SETTINGS = solara.Reactive(TCSPCSettings())
MAX_DATAPOINTS = 100_000


@solara.component
def TCSPCFigureDEPR(photons: PhotonData, title: str):
    settings = TCSPC_SETTINGS.value  # not really need in the global state
    figure, set_figure = solara.use_state(cast(Optional[go.Figure], None))
    dark_effective = solara.lab.use_dark_effective()

    def redraw():
        fig = create_tcspc_histogram(photons, settings, dark=dark_effective)
        fig.update_layout(title=title)
        set_figure(fig)

    fig_result = solara.use_thread(redraw, dependencies=[photons, settings, dark_effective])
    # TODO style="opacity: 0.3" if fig_result.state == solara.ResultState.RUNNING else None
    with solara.Card():
        solara.ProgressLinear(fig_result.state == solara.ResultState.RUNNING)
        if figure:
            solara.FigurePlotly(figure)
        else:
            solara.SpinnerSolara(size="100px")


def to_csv(traces: dict[str, BinnedPhotonData], name: str) -> BytesIO:
    """Return a CSV string for the traces."""
    # TODO this name does not do anything
    df = pd.DataFrame(
        index=next(iter(traces.values())).time,
        data={tr_name: trace.photons for tr_name, trace in traces.items()},
    )

    bio = BytesIO()
    df.to_csv(bio)
    bio.name = name
    bio.seek(0)
    return bio


# TODO remove
@dataclass
class PhotonNodeSelection:
    fret_store: FRETStore

    # TODO make sure it can not be None
    fret_id: solara.Reactive[uuid.UUID | None] = field(
        default_factory=lambda: solara.reactive(None)
    )
    photon_id: solara.Reactive[uuid.UUID | None] = field(
        default_factory=lambda: solara.reactive(None)
    )

    def __post_init__(self):
        print("deprecate")
        # i want to remove this, we can check if the current id is in selection when accessing the property
        self.fret_store._items.subscribe(self.on_fret_store)
        self.reset()

    def on_fret_store(self, new_value: list[FRETNode]):
        options = [fret_node.id for fret_node in new_value]
        if self.fret_id.value not in options:
            self.reset()

    def reset(self):
        if self.has_photons:
            fret_node = self.fret_nodes_with_photons[0]
            self.set_fret_id(fret_node.id.hex)
        else:
            self.fret_id.set(None)
            self.photon_id.set(None)

    @property
    def is_set(self) -> bool:
        return bool(self.fret_id.value) and bool(self.photon_id.value)

    @property
    def fret_nodes_with_photons(self) -> list[FRETNode]:
        return [node for node in self.fret_store if node.photons]

    @property
    def has_photons(self) -> bool:
        return bool(self.fret_nodes_with_photons)

    @property
    def fret_node(self) -> FRETNode:
        return find_object(self.fret_nodes_with_photons, id=self.fret_id.value)

    def set_fret_id(self, value: str):
        self.fret_id.set(uuid.UUID(value))

        # set the first photon node as the default value
        photon_node = self.fret_node.photons[0]
        self.photon_id.set(photon_node.id)

    @property
    def fret_values(self) -> list[dict]:
        return [
            {"text": node.name.value, "value": node.id.hex} for node in self.fret_nodes_with_photons
        ]

    @property
    def photon_node(self) -> PhotonNode:
        return find_object(self.fret_node.photons.items, id=self.photon_id.value)

    def set_photon_id(self, value: str):
        self.photon_id.set(uuid.UUID(value))

    @property
    def photon_values(self) -> list[dict]:
        return [{"text": node.name, "value": node.id.hex} for node in self.fret_node.photons]


@solara.component
def TracePage():
    solara.Title(f"{state.APP_TITLE} / Trace")

    selector_nodes = make_selector_nodes(state.fret_nodes.items)
    labels = ["Measurement", "Photons"]  # TODO move elsewhere
    TRACE_SETTINGS: solara.Reactive[TraceSettings] = solara.use_reactive(TraceSettings())

    selectors = NestedSelectors(
        nodes=selector_nodes, selection=state.trace_selection, labels=labels
    )
    with solara.Sidebar():
        for level in selectors:
            solara.Select(**level)

    photon_node = get_photons(state.fret_nodes.items, state.trace_selection.items)
    TraceFigure(photon_node, TRACE_SETTINGS.value)
    TCSPCFigure(photon_node)


@solara.component
def TraceFigure(photon_node: PhotonNode, settings: TraceSettings):
    dark_effective = solara.lab.use_dark_effective()

    async def redraw():
        photons = await state.data_manager.get_photons(photon_node)

        # also probably it would be nice to make the getting photons etc more functional?
        # we might want to move binning to data manager as well such that is is async
        t_bin = settings.t_bin * 1e-3
        bounds = (settings.t_min, settings.t_max)

        traces = {}
        for stream in TRACE_COLORS:
            stream_data = PhotonData(
                photons.data.filter(pl.col("stream") == stream),
                metadata=photons.metadata,
                cfg=photons.cfg,
            )

            traces[stream] = BinnedPhotonData(stream_data, binning_time=t_bin, bounds=bounds)

        fig = go.Figure()
        for tr in TRACE_COLORS:
            line = dict(color=TRACE_COLORS[tr])
            fig.add_trace(
                go.Scatter(
                    x=traces[tr].time,
                    y=TRACE_SIGNS[tr] * traces[tr].photons,
                    mode="lines",
                    line=line,
                    name=tr,
                )
            )

            fig.update_layout(
                # title="Time trace",
                xaxis_title="Time (s)",
                yaxis_title="Photons per bin",
                template="plotly_dark" if dark_effective else "plotly_white",
            )

        return fig

    figure_task = solara.lab.use_task(redraw, dependencies=[photon_node, settings, dark_effective])
    FigureFromTask(figure_task)


@solara.component
def TCSPCFigure(photon_node: PhotonNode):
    # todo move settings
    settings = TCSPC_SETTINGS.value
    dark_effective = solara.lab.use_dark_effective()

    # TODO make binning step also async / threaded
    async def redraw():
        photons = await state.data_manager.get_photons(photon_node)
        if photons.nanotimes_unit is None:
            raise ValueError("Provided photon data does not have TCSPC information.")
        fig = go.Figure()
        for stream in TRACE_COLORS:
            x, y = (
                photons.data.filter(pl.col("stream") == stream)["nanotimes"]
                .value_counts()
                .sort(pl.col("nanotimes"))
            )

            line = dict(color=TRACE_COLORS[stream])
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=line, name=stream))

        fig.update_layout(
            xaxis_title="Time (ns)",
            yaxis_title="Photons per bin",
            template="plotly_dark" if dark_effective else "plotly_white",
        )

        if settings.log_y:
            fig.update_yaxes(type="log")

        return fig

    figure_task = solara.lab.use_task(redraw, dependencies=[photon_node, settings, dark_effective])
    FigureFromTask(figure_task)


@solara.component
def TracePageDepr():
    solara.Title(f"{state.APP_TITLE} / Trace")

    # photons, set_photons = solara.use_state(cast(Optional[ChannelPhotonData], None))
    TRACE_SETTINGS: solara.Reactive[TraceSettings] = solara.use_reactive(TraceSettings())

    # only fret nodes with photon file items added are shown
    nodes_with_photons = [node for node in state.fret_nodes.value if node.photons]
    fret_node, set_fret_node = solara.use_state(nodes_with_photons[0])

    photon_file, set_photon_file = solara.use_state(fret_node.photons[0])
    dark_effective = solara.lab.use_dark_effective()

    def on_fret_node(node_id: str):
        new_node = state.fret_nodes.get_node(node_id)
        set_fret_node(new_node)
        set_photon_file(new_node.photons[0])

    def on_photon_file(photon_file_name: str):
        photon_file_item = state.fret_nodes.get_item(fret_node.id, "photons", photon_file_name)
        set_photon_file(photon_file_item)

    def load_photons() -> PhotonData:
        photons = photon_file.get_photons()
        return photons

    ph_result = solara.lab.use_task(load_photons, dependencies=[photon_file], prefer_threaded=True)

    def traces_memo():
        if ph_result.value is None:
            return {}
        if TRACE_SETTINGS.value.num_dpts > MAX_DATAPOINTS:
            return {}

        return generate_traces(ph_result.value, TRACE_SETTINGS.value)

    traces: dict[str, BinnedPhotonData] = solara.use_memo(
        traces_memo, dependencies=[ph_result.value, TRACE_SETTINGS.value]
    )
    with solara.Sidebar():
        with solara.Card("Controls"):
            solara.Select(
                label="Measurement",
                value=fret_node.id,
                on_value=on_fret_node,  # type: ignore
                values=[
                    {"text": fret_node.name, "value": fret_node.id}
                    for fret_node in state.fret_nodes.value
                    if fret_node.photons  # type: ignore
                ],
            )

            solara.Select(
                label="Photon file",
                value=photon_file.name,
                on_value=on_photon_file,
                values=[ph.name for ph in fret_node.photons],
            )

            solara.ProgressLinear(ph_result.pending)
            if ph_result.finished is not None:
                solara.Text("Trace settings")
                rv.Divider()  # noqa
                solara.InputFloat(
                    label="Time start",
                    # hint="Start time point of the trace in seconds.",
                    value=TRACE_SETTINGS.value.t_min,
                    on_value=lambda x: TRACE_SETTINGS.update(t_min=x),
                )
                solara.InputFloat(
                    label="Time end",
                    # hint="Ending time point of the trace in seconds.",
                    value=TRACE_SETTINGS.value.t_max,
                    on_value=lambda x: TRACE_SETTINGS.update(t_max=x),
                )
                solara.InputFloat(
                    label="Binning time",
                    # hint="Binning time in milliseconds.",
                    value=TRACE_SETTINGS.value.t_bin,
                    on_value=lambda x: TRACE_SETTINGS.update(t_bin=x),
                )

                solara.Div(style="height: 25px")

                solara.Text("TCSPC histogram settings")
                rv.Divider()

                solara.Checkbox(
                    label="logy",
                    value=TCSPC_SETTINGS.value.log_y,
                    on_value=lambda x: TCSPC_SETTINGS.update(log_y=x),
                )
                rv.Divider()

            solara.Div(style="height: 25px")

            if TRACE_SETTINGS.value.num_dpts > MAX_DATAPOINTS:
                solara.Warning(
                    "Too many datapoints to plot. Please increase the binning time or reduce the interval."
                )

            if ph_result.finished and ph_result.value is not None:
                solara.Info(f"Loaded {len(ph_result.value)} photons.")
                solara.Info(f"Duration: {ph_result.value.metadata['acquisition_duration']} s")

            if traces:

                def download_cb():
                    return to_csv(traces, "filename.txt")

                solara.FileDownload(
                    download_cb,
                    filename=f"{photon_file.name}_traces.csv",
                    children=[
                        solara.Button(
                            "Download binned data",
                            block=True,
                        )
                    ],
                )

    with solara.Column():
        # move to component
        if traces:  # this doesnt work like this? use use_state?
            fig = go.Figure()

            for tr in TRACE_COLORS:
                line = dict(color=TRACE_COLORS[tr])
                fig.add_trace(
                    go.Scatter(
                        x=traces[tr].time,
                        y=TRACE_SIGNS[tr] * traces[tr].photons,
                        mode="lines",
                        line=line,
                        name=tr,
                    )
                )

            fig.update_layout(
                title=f"Time trace: {fret_node.name}/{photon_file.name}",
                xaxis_title="Time (s)",
                yaxis_title="Photons per bin",
                template="plotly_dark" if dark_effective else "plotly_white",
            )
            with solara.Card():
                solara.FigurePlotly(fig, dependencies=[traces, dark_effective])

        if ph_result.finished:
            assert ph_result.value is not None
            TCSPCFigure(
                ph_result.value, title=f"TCSPC histogram: {fret_node.name}/{photon_file.name}"
            )
