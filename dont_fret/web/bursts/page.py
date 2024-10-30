from __future__ import annotations

import solara

import dont_fret.web.state as state
from dont_fret.web.bursts.components import BurstFigure, FilterEditDialog, FilterListItem


@solara.component
def BurstPage():
    open_filter_dialog = solara.use_reactive(False)
    solara.Title(f"{state.APP_TITLE} / Bursts")
    with solara.Sidebar():
        with solara.Card("Filters"):
            solara.Button("Edit filters", on_click=lambda: open_filter_dialog.set(True), block=True)
            with solara.v.List(dense=False):
                for filter_item in state.filters.items:
                    FilterListItem(filter_item)

    with solara.v.Dialog(
        v_model=open_filter_dialog.value, max_width=1200, on_v_model=open_filter_dialog.set
    ):
        with solara.Card(style={"width": "1190px"}):
            # only place the dialog if its open otherwise altair chart won't show
            if open_filter_dialog.value:
                FilterEditDialog()
            else:
                pass

    with solara.GridFixed(columns=2):
        BurstFigure(
            state.burst_figure_selection[0],
            state.burst_figure_file_selection[0],
        )
        BurstFigure(
            state.burst_figure_selection[1],
            state.burst_figure_file_selection[1],
        )
