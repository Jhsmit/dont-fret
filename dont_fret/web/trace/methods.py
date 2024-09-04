import plotly.graph_objects as go
import polars as pl
import solara
import solara.lab

from dont_fret import PhotonData
from dont_fret.formatting import TRACE_COLORS
from dont_fret.web.models import TCSPCSettings

# TODO move fret node / photon file reactives to module level
TCSPC_SETTINGS = solara.Reactive(TCSPCSettings())
MAX_DATAPOINTS = 100_000


def create_tcspc_histogram(
    photons: PhotonData, settings: TCSPCSettings, dark: bool = False
) -> go.Figure:
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
        template="plotly_dark" if dark else "plotly_white",
    )

    if settings.log_y:
        fig.update_yaxes(type="log")

    return fig
