import time
import uuid
from typing import Literal, Optional, TypeVar

from dont_fret.web.models import BurstNode, PhotonNode
from dont_fret.web.new_models import FRETNode, SelectorNode

T = TypeVar("T")


def not_none(*args: Optional[T]) -> T:
    """Returns the first `not None` argument"""
    for a in args:
        if a is not None:
            return a
    raise ValueError("All arguments are `None`")


def some_task(param=3) -> int:
    time.sleep(param / 2)
    return param * 2


def find_object(items: list[T], **kwargs) -> T:
    for item in items:
        if all(getattr(item, key) == value for key, value in kwargs.items()):
            return item
    raise ValueError("Object not found")


def find_index(items: list, **kwargs) -> int:
    for i, item in enumerate(items):
        if all(getattr(item, key) == value for key, value in kwargs.items()):
            return i
    raise ValueError("Object not found")


def make_selector_nodes(
    fret_nodes: list[FRETNode], attr: Literal["photons", "bursts"] = "photons"
) -> list[SelectorNode]:
    selector_nodes = []
    for fret_node in fret_nodes:
        node = SelectorNode(
            value=fret_node.id.hex,
            text=fret_node.name.value,
            children=[
                SelectorNode(value=n.id.hex, text=n.name) for n in getattr(fret_node, attr).items
            ],
        )

        selector_nodes.append(node)

    return selector_nodes


def get_photons(fret_nodes: list[FRETNode], choice: list[str]) -> PhotonNode:
    assert len(choice) == 2
    fret_node = find_object(fret_nodes, id=uuid.UUID(choice[0]))
    photon_node = find_object(fret_node.photons.items, id=uuid.UUID(choice[1]))
    return photon_node


def get_bursts(fret_nodes: list[FRETNode], choice: list[str]) -> BurstNode:
    assert len(choice) == 2
    fret_node = find_object(fret_nodes, id=uuid.UUID(choice[0]))
    burst_node = find_object(fret_node.bursts.items, id=uuid.UUID(choice[1]))
    return burst_node
