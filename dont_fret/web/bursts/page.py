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

    if open_filter_dialog:
        with solara.v.Dialog(
            v_model=open_filter_dialog.value, max_width=1200, on_v_model=open_filter_dialog.set
        ):
            with solara.Card(style={"width": "1000px"}):
                FilterEditDialog(
                    # plot_settings,
                    # selection.burst_node.df.filter(f_expr),  # = filtered dataframe by global filter
                    # on_close=lambda: set_edit_settings(False),
                    # duration=selection.burst_node.duration,
                )

    with solara.GridFixed(columns=2):
        BurstFigure(
            state.burst_figure_selection[0],
        )
        BurstFigure(
            state.burst_figure_selection[1],
        )
