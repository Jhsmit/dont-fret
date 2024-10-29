import time
import uuid
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional, TypeVar

from dont_fret.web.models import BurstNode, PhotonNode
from dont_fret.web.new_models import FRETNode, ListStore, SelectorNode

T = TypeVar("T")
V = TypeVar("V")


# TODO 1) it should be able not to take any args at all
# and just bind args and return a callable taking no arg
# and 2) you should be able to not give any value for value_arg
# and then the returned wrapper takes a single argument which is passed to the function
# as its only remaining non-wrapped argument
def wrap_callback(fn: Callable[..., Any], value_arg: str, **bound_args: Any) -> Callable[[V], None]:
    """
    Wraps a function to create a callback suitable for Solara component props.

    Args:
        fn: The base function to wrap (e.g., set_item)
        value_arg: Name of the argument that will receive the callback's value
        **bound_args: Fixed arguments to bind to the function

    Example:
        >>> store = ListStore()
        >>> on_value = wrap_callback(store.set_item, 'item', idx=0)
        >>> solara.Select(value='current', on_value=on_value, values=['a', 'b'])
    """
    pfunc = partial(fn, **bound_args)

    def wrapper(value: V) -> None:
        pfunc(**{value_arg: value})

    return wrapper


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
    fret_nodes: list[FRETNode],
    attr: Literal["photons", "bursts"] = "photons",
    require_children: bool = True,
) -> list[SelectorNode]:
    selector_nodes = []
    for fret_node in fret_nodes:
        children = [
            SelectorNode(value=n.id.hex, text=n.name) for n in getattr(fret_node, attr).items
        ]

        if require_children and not children:
            continue

        node = SelectorNode(
            value=fret_node.id.hex,
            text=fret_node.name.value,
            children=children,
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


@dataclass
class NestedSelectors:
    nodes: List[SelectorNode]
    selection: ListStore[str]
    labels: Optional[List[str]]

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        stack = self.nodes
        i = 0
        while stack:
            records = [node.record for node in stack]
            if not records:
                break

            on_value = wrap_callback(self.selection.set_item, "item", idx=i)

            val_stored = self.selection.get_item(i, None)
            if val_stored in {v["value"] for v in records}:
                value = val_stored
            else:
                value = records[0]["value"]
                on_value(value)

            if self.labels:
                label = self.labels[i]
            else:
                label = f"Level {i}"

            yield {"label": label, "value": value, "values": records, "on_value": on_value}

            selected_node = find_object(stack, value=value)
            stack = selected_node.children if selected_node else []
            i += 1
