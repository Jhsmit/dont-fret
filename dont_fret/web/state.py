import copy
from pathlib import Path

import solara

from dont_fret.config import cfg
from dont_fret.web.datamanager import ThreadedDataManager
from dont_fret.web.models import BurstColorList, FRETStore, ListStore, Snackbar
from dont_fret.web.reactive import BurstSettingsReactive

APP_TITLE = "Don't FRET!"

filebrowser_folder = solara.Reactive[Path](cfg.web.default_dir)

# TODO as liststore
burst_settings = BurstSettingsReactive(
    {k: BurstColorList(v) for k, v in cfg.burst_search.items()}  # type ignore
)

filters = ListStore(copy.deepcopy(cfg.web.burst_filters))
snackbar = Snackbar()

fret_nodes = FRETStore([])

burst_figure_selection = [
    ListStore[str](),
    ListStore[str](),
]

burst_figure_file_selection = [{}, {}]

trace_selection = ListStore[str]()

# cfg set to dask manager
data_manager = ThreadedDataManager()

# maybe check this via subscribe on the liststores ?
# -> make sure we can use the classes in different contexts
disable_trace_page = solara.Reactive(True)
disable_burst_page = solara.Reactive(True)
