from io import BytesIO

import pandas as pd
import plotly.graph_objects as go
import polars as pl
import solara
import solara.lab

import dont_fret.web.state as state
from dont_fret import BinnedPhotonData, PhotonData
from dont_fret.formatting import TRACE_COLORS, TRACE_SIGNS
from dont_fret.web.components import FigureFromTask
from dont_fret.web.models import (
    PhotonNode,
    TCSPCSettings,
    TraceSettings,
)
from dont_fret.web.utils import (
    NestedSelectors,
    get_photons,
    make_selector_nodes,
)

# TODO move fret node / photon file reactives to module level
TCSPC_SETTINGS = solara.Reactive(TCSPCSettings())
MAX_DATAPOINTS = 100_000


def to_csv(traces: dict[str, BinnedPhotonData], name: str) -> BytesIO:
    """Return a CSV string for the traces."""
    # TODO this name does not do anything
    # TODO not used, saving data needs to be reimplemented
    df = pd.DataFrame(
        index=next(iter(traces.values())).time,
        data={tr_name: trace.photons for tr_name, trace in traces.items()},
    )

    bio = BytesIO()
    df.to_csv(bio)
    bio.name = name
    bio.seek(0)
    return bio


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

    figure_task = solara.lab.use_task(  # type: ignore  # noqa: SH101
        redraw, dependencies=[photon_node, settings, dark_effective], prefer_threaded=False
    )
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

    figure_task = solara.lab.use_task(
        redraw, dependencies=[photon_node, settings, dark_effective], prefer_threaded=False
    )  # type: ignore  # noqa: SH101
    FigureFromTask(figure_task)
