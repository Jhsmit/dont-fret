from typing import Dict, List, TypeVar

from solara import Reactive
from solara.lab import Ref

from dont_fret import cfg
from dont_fret.config.config import BurstColor
from dont_fret.web.models import (
    BurstColorList,
)

S = TypeVar("S")


# make composed instead of inherited
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
