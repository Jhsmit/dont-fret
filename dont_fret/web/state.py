import copy
from pathlib import Path

import solara

from dont_fret.config import cfg
from dont_fret.web.models import BurstColorList
from dont_fret.web.new_models import FRETStore, ListStore, ThreadedDataManager
from dont_fret.web.reactive import BurstSettingsReactive, SnackbarReactive

APP_TITLE = "Don't FRET!"

filebrowser_folder = solara.Reactive[Path](cfg.web.default_dir)

# TODO as liststore
burst_settings = BurstSettingsReactive(
    {k: BurstColorList(v) for k, v in cfg.burst_search.items()}  # type ignore
)

filters = ListStore(copy.deepcopy(cfg.web.burst_filters))
snackbar = SnackbarReactive()

# from dont_fret.web.tmp_load import TEST_NODES
# fret_nodes = FRETStore(TEST_NODES)
fret_nodes = FRETStore([])

# TODO refactor fret_store

burst_figure_selection = [
    ListStore[str](),
    ListStore[str](),
]

burst_figure_file_selection = [{}, {}]

trace_selection = ListStore[str]()

# cfg set to dask manager
data_manager = ThreadedDataManager()
