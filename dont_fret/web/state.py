import copy
from pathlib import Path

import solara

from dont_fret.config import cfg
from dont_fret.web.models import BurstColorList
from dont_fret.web.reactive import BurstSettingsReactive, ReactiveFRETNodes, SnackbarReactive

APP_TITLE = "Don't FRET!"
fret_nodes = ReactiveFRETNodes([])

filebrowser_folder = solara.Reactive[Path](cfg.web.default_dir)
burst_settings = BurstSettingsReactive({k: BurstColorList(v) for k, v in cfg.burst_search.items()})

filters = solara.Reactive(copy.deepcopy(cfg.web.burst_filters))
snackbar = SnackbarReactive()

# selections for burst figures (node, burst)
burst_figure_selection = [
    (solara.Reactive(0), solara.Reactive(0)),
    (solara.Reactive(0), solara.Reactive(0)),
]
