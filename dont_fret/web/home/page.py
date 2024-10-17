from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, cast

import solara
import solara.lab
import solara.toestand

import dont_fret.web.state as state
from dont_fret.web.home.info_cards import (
    BurstInfoCard,
    BurstItemInfoCard,
    FRETNodeInfoCard,
    PhotonInfoCard,
    PhotonNodeInfoCard,
)
from dont_fret.web.methods import to_treeview
from dont_fret.web.models import BurstNode, PhotonNode
from dont_fret.web.new_models import FRETNode

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


# TODO there are now 3 selection classes
# perhaps they can be generalized
@dataclass
class TreeSelection:
    f_idx: Optional[int] = None
    node_type: Literal["bursts", "photons", None] = None
    d_idx: Optional[int] = None

    @classmethod
    def from_string(cls, input_string: str) -> TreeSelection:
        parts = input_string.split(":")

        def parse_node_type(nt: str) -> Literal["bursts", "photons", None]:
            if nt in ("bursts", "photons"):
                return nt
            elif nt.lower() == "none":
                return None
            else:
                raise ValueError(f"Invalid node_type: {nt}. Must be 'bursts', 'photons', or None.")

        if not input_string:
            return cls()
        elif len(parts) == 1:
            return cls(f_idx=int(parts[0]))
        elif len(parts) == 2:
            f_idx, node_type = parts
            return cls(f_idx=int(f_idx), node_type=parse_node_type(node_type))
        elif len(parts) == 3:
            f_idx, node_type, d_idx = parts
            return cls(f_idx=int(f_idx), node_type=parse_node_type(node_type), d_idx=int(d_idx))
        else:
            raise ValueError("Invalid input string format")

    @property
    def fret_node(self) -> FRETNode:
        assert self.f_idx is not None
        return state.fret_nodes.items[self.f_idx]

    @property
    def data_node(self) -> PhotonNode | BurstNode:
        assert self.d_idx is not None
        if self.node_type == "photons":
            return self.fret_node.photons.items[self.d_idx]
        elif self.node_type == "bursts":
            return self.fret_node.bursts.items[self.d_idx]
        else:
            raise ValueError(f"Invalid node_type: {self.node_type}")


@solara.component  # type: ignore
def HomePage():
    if len(state.fret_nodes) == 0:
        state.fret_nodes.new_node()

    with solara.Columns([4, 8], style={"height": "100%"}):
        with solara.Card("FRET Measurements"):
            with solara.Column():
                FRETTreeView(
                    items=to_treeview(state.fret_nodes.items),
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
                    on_click=lambda *args: state.fret_nodes.new_node(),
                )

        selection = TreeSelection.from_string(active_id.value[0] if active_id.value else "")
        # TODO this selection suses indices while later we use uuid's
        match selection:
            case TreeSelection(f_idx=None, node_type=None, d_idx=None):
                with solara.Card():
                    solara.Markdown(welcome_text, style="font-size: 18px")
            case TreeSelection(f_idx=int(), node_type=None, d_idx=None):
                node = selection.fret_node

                def on_delete(idx=selection.f_idx):
                    # set the active id to the parent node before deleting
                    active_id.set([])
                    state.fret_nodes.pop(idx)  # type: ignore

                FRETNodeInfoCard(name=node.name, description=node.description, on_delete=on_delete)
            case TreeSelection(f_idx=int(), node_type="photons", d_idx=None):
                # TODO open_ callable
                PhotonInfoCard(
                    selection.fret_node.name.value,
                    selection.fret_node.photons,
                    selection.fret_node.bursts,
                    state.filebrowser_folder,
                    state.burst_settings,
                )
            case TreeSelection(f_idx=int(), node_type="bursts", d_idx=None):
                # TODO open_
                BurstInfoCard(
                    selection.fret_node.name.value,
                    selection.fret_node.bursts,
                    state.filebrowser_folder,
                    # open_
                )
            case TreeSelection(f_idx=int(), node_type="photons", d_idx=int()):
                photon_node = selection.data_node
                assert isinstance(photon_node, PhotonNode)
                fret_node = selection.fret_node

                def delete_photon(p_node_idx=selection.d_idx):
                    # set the active id to the parent node before deleting
                    active_id.set([f"{selection.f_idx}:{selection.node_type}"])
                    fret_node.photons.pop(p_node_idx)  # type: ignore

                PhotonNodeInfoCard(fret_node.name.value, photon_node, delete_photon)

            case TreeSelection(f_idx=int(), node_type="bursts", d_idx=int()):
                burst_node = selection.data_node
                assert isinstance(burst_node, BurstNode)

                fret_node = selection.fret_node

                def delete_burst(b_node_idx=selection.d_idx):
                    # set the active id to the parent node before deleting
                    active_id.set([f"{selection.f_idx}:{selection.node_type}"])
                    fret_node.bursts.pop(b_node_idx)  # type: ignore

                BurstItemInfoCard(fret_node.name.value, burst_node, delete_burst)
