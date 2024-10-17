from __future__ import annotations

import polars as pl
import solara

import dont_fret.web.state as state
from dont_fret.web.bursts.components import BurstFigure, BurstFilters
from dont_fret.web.methods import chain_filters


@solara.component
def BurstPage():
    solara.Title(f"{state.APP_TITLE} / Plot 2D")
    with solara.Sidebar():
        with solara.Card():
            solara.Text("download button")
            # solara.FileDownload(
            #     filtered_df.write_csv,
            #     filename=f"{burst_item.name}{'_filtered' if state.filters.value else ''}_bursts.csv",
            #     children=[
            #         solara.Button(
            #             "Download bursts csv",
            #             block=True,
            #         )
            #     ],
            # )

    # todo make them when needed,
    # (delete when not anymore (no bursts) unlikely to happen)
    # store in state module
    with solara.GridFixed(columns=2):
        BurstFigure(
            state.burst_figure_selection[0],
            state.filters,
        )
        BurstFigure(
            state.burst_figure_selection[1],
            state.filters,
        )
