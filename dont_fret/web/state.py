import copy
import dataclasses
import uuid
from pathlib import Path
from typing import Callable, Generic, Type, TypeVar

import solara
from attrs import define, field

from dont_fret.config import cfg
from dont_fret.web.bursts.components import BurstFigureSelection
from dont_fret.web.models import BurstColorList, BurstNode, PhotonNode
from dont_fret.web.new_models import FRETStore, ThreadedDataManager
from dont_fret.web.reactive import BurstSettingsReactive, ReactiveFRETNodes, SnackbarReactive

APP_TITLE = "Don't FRET!"
# fret_nodes = ReactiveFRETNodes([])

filebrowser_folder = solara.Reactive[Path](cfg.web.default_dir)
burst_settings = BurstSettingsReactive({k: BurstColorList(v) for k, v in cfg.burst_search.items()})

filters = solara.Reactive(copy.deepcopy(cfg.web.burst_filters))
snackbar = SnackbarReactive()

# selections for burst figures (node, burst)


# TODO reactive classes should have front end stuff ONLY
# no photons. just their names
# then we dont have to have burst search method for headless and web seperately


# TODO refactor fret_store
fret_nodes = FRETStore([])

burst_figure_selection = [
    BurstFigureSelection(fret_nodes),
    BurstFigureSelection(fret_nodes),
]

# cfg set to dask manager
data_manager = ThreadedDataManager()
