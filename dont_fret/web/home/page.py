from typing import cast

import solara
import solara.lab
import solara.toestand

import dont_fret.web.state as state
from dont_fret.web.home.info_cards import (
    BurstInfoCard,
    BurstItemInfoCard,
    FRETNodeInfoCard,
    PhotonFileInfoCard,
    PhotonInfoCard,
)
from dont_fret.web.methods import to_treeview

welcome_text = """# Don't FRET!

Add measurements on the left hand side. \n
Explore the nodes in the tree view to add photon files, perform burst search, inspect photon and burst files, 
or reload burst .csv files. 

Then proceed to the `BURSTS` and `TRACE` pages to view bursts and photon files.
"""


@solara.component_vue("fret_tree.vue")
def FRETTreeView(items, value, active, open_):
    pass


# Treeview reactives
_val = solara.reactive(None)
open_ = solara.Reactive(cast(list[str], []))
active_id = solara.Reactive(cast(list[str], []))


@solara.component
def HomePage():
    if len(state.fret_nodes.value) == 0:
        state.fret_nodes.add_node()

    with solara.Columns([4, 8], style={"height": "100%"}):
        with solara.Card("FRET Measurements"):
            with solara.Column():
                FRETTreeView(
                    items=to_treeview(state.fret_nodes.value),
                    active=active_id.value,
                    on_active=active_id.set,  # type: ignore
                    open_=open_.value,
                    on_open_=open_.set,  # type: ignore
                    value=_val.value,
                    on_value=_val.set,  # type: ignore
                )

            with solara.CardActions():
                solara.Button(
                    "Add new measurement",
                    text=True,
                    on_click=lambda *args: state.fret_nodes.add_node(),
                )

        split = active_id.value[0].split(":") if active_id.value else []
        if len(split) == 0:
            with solara.Card():
                solara.Markdown(welcome_text, style="font-size: 18px")
        elif len(split) == 1:

            def on_delete_node(node_id=split[0]):
                # set the active id to the parent node before deleting
                active_id.set([])
                state.fret_nodes.remove_node(node_id)

            def on_name(value: str, node_id=split[0]):
                idx = state.fret_nodes.node_idx(node_id)
                ref = solara.toestand.Ref(state.fret_nodes.fields[idx])
                ref.update(name=value)

            def on_description(value: str, node_id=split[0]):
                idx = state.fret_nodes.node_idx(node_id)
                ref = solara.toestand.Ref(state.fret_nodes.fields[idx])
                ref.update(description=value)

            fret_node = state.fret_nodes.get_node(split[0])
            FRETNodeInfoCard(
                fret_node, on_name=on_name, on_description=on_description, on_delete=on_delete_node
            )
        elif len(split) == 2:
            node_id, dtype = split
            if dtype == "photons":
                PhotonInfoCard(
                    state.fret_nodes, node_id, state.filebrowser_folder, state.burst_settings, open_
                )
            elif dtype == "bursts":
                BurstInfoCard(state.fret_nodes, node_id, state.filebrowser_folder, open_)
        elif len(split) == 3:
            node_id, dtype, file_item_id = split
            fret_node = state.fret_nodes.get_node(node_id)

            if dtype not in ["photons", "bursts"]:
                raise ValueError(f"Invalid dtype {dtype}")

            item = state.fret_nodes.get_item(node_id, dtype, file_item_id)  # type: ignore
            if dtype == "photons":

                def on_delete(item_id=file_item_id):
                    # set the active id to the parent node before deleting
                    active_id.set([f"{node_id}:{dtype}"])
                    state.fret_nodes.remove_item(node_id, "photons", item_id)

                PhotonFileInfoCard(item, fret_node.name, on_delete)
            elif dtype == "bursts":

                def on_delete(item_id=file_item_id):
                    # set the active id to the parent node before deleting
                    active_id.set([f"{node_id}:{dtype}"])
                    state.fret_nodes.remove_item(node_id, "bursts", item_id)

                BurstItemInfoCard(item, fret_node.name, on_delete)
