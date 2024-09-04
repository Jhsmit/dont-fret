import dataclasses
import uuid
from typing import Dict, List, Literal, Optional, TypeVar, Union, overload

import solara
from solara import Reactive
from solara.lab import Ref

from dont_fret import cfg
from dont_fret.config.config import BurstColor
from dont_fret.web.models import (
    BurstColorList,
    BurstItem,
    FRETNode,
    PhotonFileItem,
    SnackbarMessage,
)

S = TypeVar("S")


class ReactiveFRETNodes(solara.Reactive[list[FRETNode]]):
    def add_node(self, name: Optional[str] = None) -> None:
        name = name or self.default_name()
        node = FRETNode(name=name, id=uuid.uuid4().hex)  # todo uuid as UUID type
        self.append(node)

    def append(self, item: FRETNode) -> None:
        new_value = self.value.copy()
        new_value.append(item)
        self.value = new_value

    def extend(self, items: list[FRETNode]) -> None:
        new_value = self.value.copy()
        new_value.extend(items)
        self.value = new_value

    def default_name(self) -> str:
        num_dic = {
            1: "TWO",
            2: "THREE",
            3: "FOUR",
            4: "FIVE",
            5: "SIX",
            6: "SEVEN",
            7: "EIGHT",
            8: "NINE",
            9: "TEN",
        }

        if len(self.value) == 0:
            return "FRET NOT"
        else:
            return f"FRET {num_dic.get(len(self.value), len(self.value) + 1)}"

    def remove_node(self, node_id: str) -> None:
        self.value = [node for node in self.value if node.id != node_id]

    def node_idx(self, node_id: str) -> int:
        return [node.id for node in self.value].index(node_id)

    def get_node(self, node_id: str) -> FRETNode:
        for node in self.value:
            if node.id == node_id:
                return node

        raise ValueError(f"Node with id {node_id} not found.")

    def remove_all_photon_files(self, node_id: str) -> None:
        idx = self.node_idx(node_id)
        ref = Ref(self.fields[idx].photons)
        ref.value = []

    def add_photon_files(self, node_id: str, photon_files: list[PhotonFileItem]) -> None:
        idx = self.node_idx(node_id)
        ref = Ref(self.fields[idx].photons)
        ref.value = ref.value + photon_files

    def add_burst_items(self, node_id: str, burst_items: list[BurstItem]) -> None:
        """
        adds a burst item. If the item already exists, it will be replaced
        """

        idx = self.node_idx(node_id)
        ref = Ref(self.fields[idx].bursts)
        new_names = [item.name for item in burst_items]
        previous_items = [item for item in ref.value if item.name not in new_names]
        ref.value = previous_items + burst_items

    @overload
    def get_item(
        self, node_id: str, item_type: Literal["photons"], item_name: str
    ) -> PhotonFileItem:
        ...

    @overload
    def get_item(self, node_id: str, item_type: Literal["bursts"], item_name: str) -> BurstItem:
        ...

    # overload for type checking
    def get_item(
        self, node_id: str, item_type: Literal["photons", "bursts"], item_name: str
    ) -> Union[PhotonFileItem, BurstItem]:
        idx = self.node_idx(node_id)
        ref = Ref(getattr(self.fields[idx], item_type))
        for item in ref.value:
            if item.name == item_name:
                return item

        raise ValueError(f"Item with name {item_name} not found.")

    def remove_item(
        self, node_id: str, item_type: Literal["photons", "bursts"], item_name: str
    ) -> None:
        idx = self.node_idx(node_id)
        ref = Ref(getattr(self.fields[idx], item_type))
        ref.value = [item for item in ref.value if item.name != item_name]


class ReactiveList(solara.Reactive[list[S]]):
    def add_item(self, item: S) -> None:
        self.value = self.value + [item]

    def remove_item(self, idx: int) -> None:
        new_items = self.value.copy()
        del new_items[idx]
        self.value = new_items

        # self.value = [item for i, item in enumerate(self.value) if i != idx]


class PhotonFileReactiveList(ReactiveList[PhotonFileItem]):
    def __init__(self, *args, **kwargs):
        raise DeprecationWarning("deprecated in favour of ReactiveFRETNodes")

    def add_file_items(self, file_items: list[PhotonFileItem]) -> None:
        new_items = self.value + [f for f in file_items if f.file_info["name"] not in self.names]
        if new_items:
            self.value = new_items

    def remove_file_item(self, idx: int) -> None:
        new_items = self.value.copy()
        del new_items[idx]
        self.value = new_items

    def remove_all(self) -> None:
        self.value = []

    def select_index(self, idx: list[int]) -> None:
        self.value = [dataclasses.replace(f, selected=i in idx) for i, f in enumerate(self.value)]

    def select_none(self) -> None:
        self.value = [dataclasses.replace(f, selected=False) for f in self.value]

    def select_all(self) -> None:
        self.value = [dataclasses.replace(f, selected=True) for f in self.value]

    @property
    def names(self) -> list[str]:
        """Returns names of all files"""
        return [item.file_info["name"] for item in self.value]

    @property
    def selected(self) -> list[int]:
        """Returns indices of selected files"""
        return [i for i, f in enumerate(self.value) if f.selected]


class BurstItemsReactiveList(ReactiveList[BurstItem]):
    @property
    def names(self) -> list[str]:
        """Returns names of all files"""
        return [item.name for item in self.value]


class BurstSettingsReactive(Reactive[Dict[str, List[BurstColor]]]):
    def reset(self) -> None:
        self.value = {k: BurstColorList(v) for k, v in cfg.burst_search.items()}

    def add_settings(self, setting_name: str):
        """Adds a new burst settings name with default settings."""
        new_value = self.value.copy()
        new_value[setting_name] = [BurstColor()]
        self.value = new_value

    def remove_settings(self, setting_name: str):
        """Removes a burst settings name."""
        new_value = self.value.copy()
        new_value.pop(setting_name)
        self.value = new_value

    def remove_color(self, setting_name: str):
        """Removes the last color from the list of colors for a given burst settings name."""
        colors_ref = Ref(self.fields[setting_name])
        new_colors = colors_ref.get().copy()
        if len(new_colors) == 1:
            return
        new_colors.pop()
        colors_ref.set(new_colors)  # isnt this a copy?

    def update_color(self, settings_name: str, color_idx: int, **kwargs):
        # calling update / setter is fine because there are no listeners attached
        Ref(self.fields[settings_name][color_idx]).update(**kwargs)

    def get_color(self, settings_name: str, color_idx: int) -> BurstColor:
        # this is not allowed because it creates a listener with a too high idx
        # return Ref(self.fields[setting_name][color_idx]).get()

        # do this instead
        return self.value[settings_name][color_idx]

    def add_color(self, settings_name: str):
        colors_ref = Ref(self.fields[settings_name])
        colors_ref.set(colors_ref.get().copy() + [BurstColor()])

    @property
    def settings_names(self) -> List[str]:
        return list(self.value.keys())


class SnackbarReactive(solara.Reactive[SnackbarMessage]):
    def __init__(self, value: SnackbarMessage = SnackbarMessage(), default_timeout: int = 5000):
        self.default_timeout = default_timeout
        super().__init__(value)

    def set_message(self, msg: str, timeout: Optional[int] = None, **kwargs):
        timeout = timeout if timeout is not None else self.default_timeout
        self.update(message=msg, timeout=timeout, show=True, **kwargs)

    def info(self, msg: str, timeout: Optional[int] = None):
        self.set_message(msg, color="primary", btn_color="text-primary-color", timeout=timeout)

    def secondary(self, msg: str, timeout: Optional[int] = None):
        self.set_message(msg, color="secondary", btn_color="text-secondary-color", timeout=timeout)

    def warning(self, msg: str, timeout: Optional[int] = None):
        self.set_message(msg, color="warning", btn_color="text-warning-color", timeout=timeout)

    def error(self, msg: str, timeout: Optional[int] = None):
        self.set_message(msg, color="error", btn_color="text-error-color", timeout=timeout)

    def success(self, msg: str, timeout: Optional[int] = None):
        self.set_message(msg, color="success", btn_color="text-success-color", timeout=timeout)
