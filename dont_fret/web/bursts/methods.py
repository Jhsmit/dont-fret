from typing import Optional

import plotly.graph_objs as go


def create_histogram(
    data, xbins, arange: tuple[float, float], xrange: tuple[Optional[float], Optional[float]]
) -> go.Figure:
    hist_trace = go.Histogram(x=data, xbins=xbins)

    layout = go.Layout(
        modebar_remove=[
            "lasso2d",
            "select2d",
            "zoom2d",
            "pan2d",
            "zoomIn2d",
            "zoomOut2d",
            "autoScale2d",
            "resetScale2d",
        ],
        xaxis=dict(
            range=arange,
            type="linear",
        ),
        yaxis=dict(
            type="linear",
        ),
    )

    # Create a figure with the histogram trace and layout
    fig = go.Figure(data=[hist_trace], layout=layout)
    fig.update_layout(dragmode=False, selectdirection="h")

    fig.add_vrect(
        editable=False,
        x0=arange[0],
        x1=xrange[0] if xrange[0] is not None else arange[0],
        fillcolor="gray",
        opacity=0.5,
        layer="above",
        line_width=0,
    )
    fig.add_vrect(
        editable=False,
        x0=xrange[1] if xrange[1] is not None else arange[1],
        x1=arange[1],
        fillcolor="gray",
        opacity=0.5,
        layer="above",
        line_width=0,
    )

    return fig
