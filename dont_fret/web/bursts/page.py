from __future__ import annotations

import polars as pl
import solara

import dont_fret.web.state as state
from dont_fret.web.bursts.components import BurstFigure, BurstFilters
from dont_fret.web.methods import chain_filters


@solara.component
def BurstPage():
    solara.Title(f"{state.APP_TITLE} / Plot 2D")
    # sidebar selected node and burst for filters
    has_bursts = [node for node in state.fret_nodes.value if node.bursts]
    fret_node, set_fret_node = solara.use_state(has_bursts[0])
    burst_item, set_burst_item = solara.use_state(fret_node.bursts[0])

    def on_fret_node(node_id: str):
        new_node = state.fret_nodes.get_node(node_id)
        set_fret_node(new_node)
        set_burst_item(new_node.bursts[0])

    def on_burst_item(item_name: str):
        new_item = state.fret_nodes.get_item(fret_node.id, "bursts", item_name)
        set_burst_item(new_item)

    def get_dataframe() -> tuple[pl.DataFrame, pl.DataFrame]:
        return burst_item.df.filter(chain_filters(state.filters.value)), burst_item.df

    filtered_df, original_df = solara.use_memo(
        get_dataframe,
        dependencies=[
            burst_item,
            state.filters.value,
        ],  # TODO do they need id's for faster hashing?
    )
    with solara.Sidebar():
        with solara.Card("Burst plot settings", margin=0, elevation=0):
            solara.Select(
                label="Measurement",
                value=fret_node.id,
                on_value=on_fret_node,  # type: ignore
                values=[
                    {"text": fret_node.name, "value": fret_node.id}
                    for fret_node in state.fret_nodes.value
                    if fret_node.bursts  # type: ignore
                ],
            )

            # check what happens if fret node is changed; should make sure to select a new burst item
            solara.Select(
                label="Burst item",
                value=burst_item.name,
                on_value=on_burst_item,
                values=[ph.name for ph in fret_node.bursts],
            )
            solara.Info(f"Number of bursts: {len(original_df)}", dense=True)
            solara.Info(f"After filtering: {len(filtered_df)}", dense=True)
            items = list(filtered_df.columns)
            try:
                items.remove("filename")
            except ValueError:
                pass

        BurstFilters(state.filters, original_df)
        with solara.Card():
            solara.FileDownload(
                filtered_df.write_csv,
                filename=f"{burst_item.name}{'_filtered' if state.filters.value else ''}_bursts.csv",
                children=[
                    solara.Button(
                        "Download bursts csv",
                        block=True,
                    )
                ],
            )

    with solara.GridFixed(columns=2):
        BurstFigure(
            state.fret_nodes,
            state.filters,
            node_idx=state.burst_figure_selection[0][0],
            burst_idx=state.burst_figure_selection[0][1],
        )
        BurstFigure(
            state.fret_nodes,
            state.filters,
            node_idx=state.burst_figure_selection[1][0],
            burst_idx=state.burst_figure_selection[1][1],
        )
